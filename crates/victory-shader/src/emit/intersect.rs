//! The console's ray-tracing node test, `image_bvh_intersect_ray` and
//! `image_bvh64_intersect_ray`, as inline DXIL: the host has no instruction
//! that tests one ray against one node of the title's own structures.
//!
//! [`crate::bvh`] says what the instruction computes and is the reference:
//! the code made here does the same float operations in the same order
//! (DXIL float operations carry no fast-math flags, so the host may not
//! reorder or fuse them), and the tests in `intersect_tests.rs` run it on
//! the card against the reference.
//!
//! The title's nodes are in title memory, named by address: they are read
//! through the host's copies of title memory (`SLOT_MEMORY` / `SLOT_FAULTS`),
//! one directory lookup per readable node (with a second lookup for a 128-byte
//! box node only when the node straddles a 64 KiB page boundary).
//!
//! Shape of the code, under the instruction's lane guard:
//!
//! ```text
//! kind = pointer & 7
//! other (6, 7): a triangle miss (no directory lookup, no fault recorded)
//! readable (0-5): resolve node copy once
//!   box (4, 5): 4 child pointers; stream each box's slab test as its 6 floats
//!               load/widen (kind 5 checks page crossing for bytes 64..112);
//!               the sorting network
//!   triangle (0-3): the id dword, three vertices at the bytes it names; the
//!                   watertight test
//! ```
//!
//! Each lane reads only the dwords its node's kind needs: 28 for a 32-bit
//! box node, 16 for a 16-bit one, 10 for a triangle.

use super::{Attr, Bin, Cast, Fields, Inst, Lower, Operand, Reg, RootType, Tag, Term, V, Why, pred};
use crate::bvh;
use crate::resource::BvhNodes;

#[cfg(test)]
thread_local! {
    /// The tests' stand-in for title memory ([`Lower::bvh_memory`]): a
    /// heap slot holding a raw buffer, and the title address of its first
    /// byte. (The card tests have no host to keep copies of title memory
    /// and their directory.)
    pub(in crate::emit) static TEST_MEMORY: std::cell::Cell<Option<(u32, u64)>> =
        const { std::cell::Cell::new(None) };
}

/// Where the instruction's node is.
#[derive(Clone, Copy)]
enum NodeAt {
    /// At this byte of a raw buffer of the host.
    Buffer { handle: V, byte: V },
    /// In title memory: the raw view of the host copy holding the node's
    /// first 64 bytes, its view-heap `slot` and `byte` offset
    /// ([`Lower::bvh_memory_slot`]), and the node's address for the second
    /// half of a 128-byte node when it crosses a 64 KiB page boundary.
    Memory {
        handle: V,
        slot: V,
        byte: V,
        low: V,
        high: V,
    },
}

/// How box sorting is controlled for this node test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SortMode {
    Always,
    Never,
    /// Runtime bit 31 of descriptor dword 1 (`i1`).
    Dynamic(V),
}

/// The ray operands, as floats.
struct Ray {
    extent: V,
    origin: [V; 3],
    direction: [V; 3],
    inverse: [V; 3],
}

impl<'a> Lower<'a> {
    /// `image_bvh_intersect_ray` and `image_bvh64_intersect_ray`.
    pub(super) fn bvh(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            rsrc,
            dmask,
            a16,
            d16,
            tfe,
            lwe,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *a16 || *d16 || *tfe || *lwe || *dmask == 0 || *dmask > 0xf {
            return Err("a ray-tracing node test with 16-bit operands or an unsupported mask");
        }
        let wide = inst.name() == "image_bvh64_intersect_ray";
        // Operand VGPRs in order: the node pointer (two dwords in the
        // 64-bit form), the ray's extent, origin, direction and inverse
        // direction.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        if regs.len() != if wide { 12 } else { 11 } {
            return Err("a ray-tracing node test's operands");
        }
        let Operand::Vector { reg: dst, .. } = *data else {
            return Err("a ray-tracing node test's result");
        };
        let site = (b, i);
        let nodes = self
            .x
            .res
            .bvh
            .iter()
            .find(|n| n.0 == site)
            .map(|n| n.1)
            .ok_or("an untracked ray-tracing structure")?;
        if let BvhNodes::Memory { word3, .. } = nodes
            && word3 & bvh::RETURN_BARYCENTRICS == 0
        {
            // Mode 0 returns a triangle id and a hit status in the last two
            // dwords; no program of the title asks for it.
            return Err("a ray-tracing structure that returns triangle ids");
        }
        let guard = self.guard(b, i)?;
        let mut x = Vec::with_capacity(regs.len());
        for &r in &regs {
            let v = inst.read(Reg::V(r)).ok_or("an unread operand")?;
            x.push(self.i32_of(v)?);
        }
        let pointer = x[0];
        let pointer_high = wide.then(|| x[1]);
        let f: Vec<V> = x[1 + usize::from(wide)..]
            .iter()
            .map(|&v| self.as_float(v))
            .collect();
        let ray = Ray {
            extent: f[0],
            origin: [f[1], f[2], f[3]],
            direction: [f[4], f[5], f[6]],
            inverse: [f[7], f[8], f[9]],
        };
        // Prepare uniform descriptor inputs before entering the lane guard.
        enum Source {
            Resource(V),
            Memory {
                word0: V,
                word1: V,
                sort: Option<bool>,
                zero_base: bool,
            },
        }
        let source = match nodes {
            BvhNodes::Resource => {
                let src = self
                    .descriptors
                    .get(&(site, false))
                    .copied()
                    .ok_or("an untracked ray-tracing structure")?;
                Source::Resource(self.handle(site, src, false, (11, 0))?)
            }
            BvhNodes::Memory {
                sort, zero_base, ..
            } => {
                let Operand::Scalar { code, .. } = *rsrc else {
                    return Err("a ray-tracing structure descriptor not in SGPRs");
                };
                let mut words = [pointer; 2];
                for (k, w) in words.iter_mut().enumerate() {
                    let v = inst
                        .read(Reg::S(code + k as u8))
                        .ok_or("an unread operand")?;
                    *w = self.i32_of(v)?;
                }
                Source::Memory {
                    word0: words[0],
                    word1: words[1],
                    sort,
                    zero_base,
                }
            }
        };
        let zero = self.s.m.i32(0);
        let gate = self.begin_guarded(guard);
        let head = self.cur;
        let (k4, k6, k7) = (self.s.m.i32(4), self.s.m.i32(6), self.s.m.i32(7));
        let kind = self.s.m.bin(head, Bin::And, pointer, k7);
        // Kinds 6 (instance) and 7 (procedural) are not read by the
        // instruction; skip the directory lookup and fault path for them.
        let readable = self.s.m.cmp(head, pred::IULT, kind, k6);
        let (fetch_block, box_block) = (self.s.m.new_block(), self.s.m.new_block());
        let (triangle_block, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s
            .m
            .terminate(head, Term::CondBr(readable, fetch_block, join));

        self.cur = fetch_block;
        let (at, sort_mode) = match source {
            Source::Resource(handle) => {
                // (pointer >> 3) << 6 from the buffer's start.
                let (kind_bits, k3) = (self.s.m.i32(!7), self.s.m.i32(3));
                let byte = self.s.m.bin(fetch_block, Bin::And, pointer, kind_bits);
                let byte = self.s.m.bin(fetch_block, Bin::Shl, byte, k3);
                (NodeAt::Buffer { handle, byte }, SortMode::Always)
            }
            Source::Memory {
                word0,
                word1,
                sort,
                zero_base,
            } => {
                let (low, high) =
                    self.bvh_address(word0, word1, pointer, pointer_high, zero_base);
                // One lookup for the node's first 64 bytes: a node is 64 bytes
                // on 64, so its first half lies in one 64 KiB page whatever
                // its kind.
                let (slot, byte, nonuniform) = self.bvh_memory_slot(low, high);
                let handle = self.bvh_memory_handle(slot, nonuniform);
                let mode = match sort {
                    Some(true) => SortMode::Always,
                    Some(false) => SortMode::Never,
                    None => {
                        let box_sort = self.s.m.i32(bvh::BOX_SORT);
                        let bit = self.s.m.cmp(self.cur, pred::IUGE, word1, box_sort);
                        SortMode::Dynamic(bit)
                    }
                };
                (
                    NodeAt::Memory {
                        handle,
                        slot,
                        byte,
                        low,
                        high,
                    },
                    mode,
                )
            }
        };
        let fetch_end = self.cur;
        let is_box = self.s.m.cmp(fetch_end, pred::IUGE, kind, k4);
        self.s
            .m
            .terminate(fetch_end, Term::CondBr(is_box, box_block, triangle_block));

        self.cur = box_block;
        let boxes = self.bvh_boxes(at, kind, &ray, sort_mode);
        let box_end = self.cur;
        self.s.m.terminate(box_end, Term::Br(join));

        self.cur = triangle_block;
        let triangle = self.bvh_triangle(at, kind, &ray);
        let triangle_end = self.cur;
        self.s.m.terminate(triangle_end, Term::Br(join));

        self.cur = join;
        // Kinds 6 and 7 (an instance, a procedural primitive): not known
        // what the console returns; the title's programs do not read it.
        let other = bvh::MISS.map(|f| self.s.m.i32(f.to_bits()));
        let i32t = self.s.t.i32;
        let channels: Vec<usize> = (0..4).filter(|ch| (*dmask >> ch) & 1 != 0).collect();
        let mut results = Vec::with_capacity(channels.len());
        for &ch in &channels {
            let phi = self.s.m.phi(join, i32t);
            self.s.m.set_phi(
                phi,
                vec![
                    (boxes[ch], box_end),
                    (triangle[ch], triangle_end),
                    (other[ch], head),
                ],
            );
            results.push((phi, zero));
        }
        let results = self.end_guarded(gate, &results);
        for (k, &r) in results.iter().enumerate() {
            self.write(inst, Reg::V(dst + k as u8), r, guard);
        }
        Ok(())
    }

    /// Title memory at `low`/`high` for a readable node's bytes: the view-heap
    /// `slot`, byte offset in that view, and whether the slot index may be
    /// non-uniform across lanes.
    ///
    /// Unlike the branchless [`Lower::memory`], the fault record (`URem` +
    /// `rawBufferStore` to `SLOT_FAULTS`) sits behind a cold `copy == 0`
    /// branch: mapped pages (the common case) perform no integer remainder and
    /// never contend on the global `SLOT_FAULTS[0]` sink dword. Only called
    /// inside the lane guard for readable node kinds (`0..=5`).
    fn bvh_memory_slot(&mut self, low: V, high: V) -> (V, V, bool) {
        #[cfg(test)]
        if let Some((slot, base)) = TEST_MEMORY.get() {
            // The buffer at heap slot `slot` holds the bytes from title
            // address `base` on; any other address reads past its end,
            // which is zeros.
            let c = self.cur;
            let slot = self.s.m.i32(slot);
            let (base_low, base_high) =
                (self.s.m.i32(base as u32), self.s.m.i32((base >> 32) as u32));
            let at = self.s.m.bin(c, Bin::Sub, low, base_low);
            let in_low = self.s.m.cmp(c, pred::IUGE, low, base_low);
            let in_high = self.s.m.cmp(c, pred::EQ, high, base_high);
            let inside = self.s.m.bin(c, Bin::And, in_low, in_high);
            let outside = self.s.m.i32(0xffff_ff00);
            let at = self.s.m.select(c, inside, at, outside);
            return (slot, at, false);
        }
        use super::super::shader::{
            MEMORY_FAULTS, MEMORY_PAGES, SLOT_FAULTS, SLOT_MEMORY, USER_MEMORY_BASE,
        };
        let c = self.cur;
        let (i32t, res, void) = (self.s.t.i32, self.s.t.res_ret, self.s.t.void);
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (k16, k8) = (k(self, 16), k(self, 0xFF));
        let lo_page = self.s.m.bin(c, Bin::LShr, low, k16);
        let hi_page = self.s.m.bin(c, Bin::And, high, k8);
        let hi_page = self.s.m.bin(c, Bin::Shl, hi_page, k16);
        let page = self.s.m.bin(c, Bin::Or, lo_page, hi_page);
        let last = k(self, MEMORY_PAGES - 1);
        let page = self
            .s
            .op(c, "binary.i32", Attr::ReadNone, i32t, 40, &[page, last]);
        self.s.m.tag(page, Tag::Clamp);
        let k3 = k(self, 3);
        let at = self.s.m.bin(c, Bin::Shl, page, k3);
        let directory = self.s.root_srv_handle(SLOT_MEMORY);
        let (iu, both, align4, align8) = (
            self.s.m.undef(i32t),
            self.s.m.i8(3),
            k(self, 4),
            k(self, 8),
        );
        let entry = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            res,
            139,
            &[directory, at, iu, both, align8],
        );
        let (copy, start) = (self.s.m.extract(c, entry, 0), self.s.m.extract(c, entry, 1));
        let (copy, start) = (self.as_int(copy), self.as_int(start));
        // Hot path only checks `copy == 0` (1 instruction); the cold fault
        // path skips unmappable page 0 and the out-of-range sentinel
        // `MEMORY_PAGES - 1` so it never erases `faults[1]` or clobbers a slot.
        let (zero, one) = (k(self, 0), k(self, 1));
        let missing = self.s.m.cmp(c, pred::EQ, copy, zero);
        let (fault_block, mapped_block) = (self.s.m.new_block(), self.s.m.new_block());
        self.s
            .m
            .terminate(c, Term::CondBr(missing, fault_block, mapped_block));

        self.cur = fault_block;
        let valid_limit = k(self, MEMORY_PAGES - 2);
        let page_m1 = self.s.m.bin(fault_block, Bin::Sub, page, one);
        let mappable = self.s.m.cmp(fault_block, pred::IULT, page_m1, valid_limit);
        let store_fault = self.s.m.new_block();
        self.s
            .m
            .terminate(fault_block, Term::CondBr(mappable, store_fault, mapped_block));

        self.cur = store_fault;
        let places = k(self, MEMORY_FAULTS - 1);
        let place = self.s.m.bin(store_fault, Bin::URem, page, places);
        let place = self.s.m.bin(store_fault, Bin::Add, place, one);
        let k2 = k(self, 2);
        let place = self.s.m.bin(store_fault, Bin::Shl, place, k2);
        let faults = self.s.root_uav_handle(SLOT_FAULTS);
        let mask = self.s.m.i8(1);
        self.s.op(
            store_fault,
            "rawBufferStore.i32",
            Attr::NoUnwind,
            void,
            140,
            &[faults, place, iu, page, iu, iu, iu, mask, align4],
        );
        self.s.m.terminate(store_fault, Term::Br(mapped_block));

        self.cur = mapped_block;
        let base = self.host_word(RootType::Other(USER_MEMORY_BASE), 0, 0);
        let slot = self.s.m.bin(mapped_block, Bin::Add, base, copy);
        let k64k = k(self, 0xFFFF);
        let within = self.s.m.bin(mapped_block, Bin::And, low, k64k);
        let byte = self.s.m.bin(mapped_block, Bin::Add, start, within);
        (slot, byte, true)
    }

    /// Creates a raw-buffer view-heap handle in the current block for `slot`.
    fn bvh_memory_handle(&mut self, slot: V, nonuniform: bool) -> V {
        let props = if nonuniform { (11 | 0x1000, 0) } else { (11, 0) };
        self.s.heap_handle(self.cur, slot, false, nonuniform, props)
    }

    /// Reads `count` raw dwords (as `f32` when `as_float`, else `i32`) from
    /// `handle` starting at `byte + base + offset`, four at a time.
    fn bvh_load_dwords(
        &mut self,
        handle: V,
        byte: V,
        base: Option<V>,
        offset: u32,
        count: u32,
        as_float: bool,
    ) -> Vec<V> {
        let c = self.cur;
        let undef = self.s.m.undef(self.s.t.i32);
        let (align4, align16) = (self.s.m.i32(4), self.s.m.i32(16));
        let base_byte = match base {
            Some(b) => self.s.m.bin(c, Bin::Add, byte, b),
            None => byte,
        };
        let mut out = Vec::with_capacity(count as usize);
        for start in (0..count).step_by(4) {
            let n = (count - start).min(4);
            let add = offset + 4 * start;
            let addr = if add == 0 {
                base_byte
            } else {
                let k = self.s.m.i32(add);
                self.s.m.bin(c, Bin::Add, base_byte, k)
            };
            let mask = self.s.m.i8((1u8 << n) - 1);
            // Nodes are 64-byte aligned in memory and buffers; aligned 16-byte
            // `dwordx4` loads let host compilers emit 128-bit vector loads.
            let align = if base.is_none() && n == 4 && add.is_multiple_of(16) {
                align16
            } else {
                align4
            };
            let r = self.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                139,
                &[handle, addr, undef, mask, align],
            );
            for k in 0..n {
                let f = self.s.m.extract(c, r, k);
                out.push(if as_float { f } else { self.as_int(f) });
            }
        }
        out
    }

    /// Reads `count` dwords from the node's first 64 bytes (guaranteed to sit
    /// in the already-resolved first page for `NodeAt::Memory`).
    fn bvh_node_page0(
        &mut self,
        at: NodeAt,
        base: Option<V>,
        offset: u32,
        count: u32,
        as_float: bool,
    ) -> Vec<V> {
        let (handle, byte) = match at {
            NodeAt::Buffer { handle, byte } | NodeAt::Memory { handle, byte, .. } => (handle, byte),
        };
        self.bvh_load_dwords(handle, byte, base, offset, count, as_float)
    }

    /// The node's title address, [`bvh::node_address`]: the descriptor's
    /// base (dword 0 and the low byte of dword 1, in 256-byte units) plus
    /// `(pointer >> 3) << 6`, computed in pure 32-bit arithmetic (no `i64`
    /// shader feature flag required).
    fn bvh_address(
        &mut self,
        word0: V,
        word1: V,
        pointer: V,
        pointer_high: Option<V>,
        zero_base: bool,
    ) -> (V, V) {
        let c = self.cur;
        let (kind_mask, k3, k29) = (self.s.m.i32(!7), self.s.m.i32(3), self.s.m.i32(29));
        // (pointer >> 3) << 6 == (pointer & !7) << 3 in low 32 bits;
        // high 32 bits == (pointer >> 29) | (pointer_high << 3).
        let p_lo = self.s.m.bin(c, Bin::And, pointer, kind_mask);
        let p_lo = self.s.m.bin(c, Bin::Shl, p_lo, k3);
        let p_hi_lo = self.s.m.bin(c, Bin::LShr, pointer, k29);
        let p_hi = match pointer_high {
            Some(ph) => {
                let ph_sh = self.s.m.bin(c, Bin::Shl, ph, k3);
                self.s.m.bin(c, Bin::Or, p_hi_lo, ph_sh)
            }
            None => p_hi_lo,
        };
        if zero_base {
            return (p_lo, p_hi);
        }
        let (k8, k24, byte_mask) = (self.s.m.i32(8), self.s.m.i32(24), self.s.m.i32(0xff));
        let base_lo = self.s.m.bin(c, Bin::Shl, word0, k8);
        let base_hi_lo = self.s.m.bin(c, Bin::LShr, word0, k24);
        let base_hi_hi = self.s.m.bin(c, Bin::And, word1, byte_mask);
        let base_hi_hi = self.s.m.bin(c, Bin::Shl, base_hi_hi, k8);
        let base_hi = self.s.m.bin(c, Bin::Or, base_hi_lo, base_hi_hi);
        let low = self.s.m.bin(c, Bin::Add, base_lo, p_lo);
        let carry = self.s.m.cmp(c, pred::IULT, low, base_lo);
        let carry = self.s.m.cast(c, Cast::ZExt, carry, self.s.t.i32);
        let high = self.s.m.bin(c, Bin::Add, base_hi, p_hi);
        let high = self.s.m.bin(c, Bin::Add, high, carry);
        (low, high)
    }

    /// Tests one box (`bx = [min_x, min_y, min_z, max_x, max_y, max_z]`)
    /// against `ray` using pre-clamped `inverse`, returning `(id, distance)`.
    #[allow(clippy::too_many_arguments)]
    fn bvh_slab_test(
        &mut self,
        bx: [V; 6],
        child: V,
        ray: &Ray,
        inverse: [V; 3],
        none: V,
        infinity: V,
        fzero: V,
        abs_mask: V,
        inf_bits: V,
    ) -> (V, V) {
        let c = self.cur;
        let (mut near, mut far) = ([fzero; 3], [fzero; 3]);
        for a in 0..3 {
            let low = self.s.m.bin(c, Bin::Sub, bx[a], ray.origin[a]);
            let low = self.s.m.bin(c, Bin::Mul, low, inverse[a]);
            let high = self.s.m.bin(c, Bin::Sub, bx[3 + a], ray.origin[a]);
            let high = self.s.m.bin(c, Bin::Mul, high, inverse[a]);
            near[a] = self.binary(36, low, high);
            far[a] = self.binary(35, low, high);
        }
        let enter = self.binary(35, near[0], near[1]);
        let enter = self.binary(35, enter, near[2]);
        let leave = self.binary(36, far[0], far[1]);
        let leave = self.binary(36, leave, far[2]);
        let from = self.binary(35, enter, fzero);
        // Ordered compares: a NaN anywhere is a miss.
        let through = self.s.m.cmp(c, pred::OGE, leave, from);
        let before = self.s.m.cmp(c, pred::OLT, enter, ray.extent);
        // Integer NaN test (`!bx[0].is_nan()`): avoids drivers folding `fcmp`
        // away under fast-math while saving the `xor` of `!is_nan_f32`.
        let bx0_bits = self.as_int(bx[0]);
        let bx0_abs = self.s.m.bin(c, Bin::And, bx0_bits, abs_mask);
        let on = self.s.m.cmp(c, pred::IULE, bx0_abs, inf_bits);
        let hit = self.s.m.bin(c, Bin::And, through, before);
        let hit = self.s.m.bin(c, Bin::And, hit, on);
        let id = self.s.m.select(c, hit, child, none);
        let dist = self.s.m.select(c, hit, enter, infinity);
        (id, dist)
    }

    /// The box test ([`bvh::boxes`]): four result dwords. Starts in the
    /// current block and leaves `self.cur` at the block it ends in.
    fn bvh_boxes(&mut self, at: NodeAt, kind: V, ray: &Ray, sort_mode: SortMode) -> [V; 4] {
        let head = self.cur;
        // Both 16-bit (64 B) and 32-bit (128 B) box nodes read the exact same
        // first 64 bytes (4 child pointers + 12 box dwords) from page 0:
        // fetch them once before branching on `kind == 5` so divergent waves
        // never issue those three loads twice.
        let first64 = self.bvh_node_page0(at, None, 0, 16, true);
        let children: [V; 4] = std::array::from_fn(|n| self.as_int(first64[n]));
        let mid_floats = &first64[4..16];
        let k5 = self.s.m.i32(5);
        let wide = self.s.m.cmp(head, pred::EQ, kind, k5);
        let (wide_block, half_block) = (self.s.m.new_block(), self.s.m.new_block());
        let join = self.s.m.new_block();
        self.s
            .m
            .terminate(head, Term::CondBr(wide, wide_block, half_block));

        // 32-bit floats: first 12 floats are already loaded from bytes 16..64;
        // bytes 64..112 only cross a 64 KiB page boundary when
        // `(low & 0xFFFF) >= 0xFFC0` (<0.1% of 64B-aligned nodes, 0% of
        // 128B-aligned nodes).
        self.cur = wide_block;
        let mut floats = mid_floats.to_vec();
        let second12 = match at {
            NodeAt::Buffer { handle, byte } => {
                self.bvh_load_dwords(handle, byte, None, 64, 12, true)
            }
            NodeAt::Memory {
                slot,
                byte,
                low,
                high,
                ..
            } => {
                let k_ffff = self.s.m.i32(0xFFFF);
                let k_ffc0 = self.s.m.i32(0xFFC0);
                let page_off = self.s.m.bin(wide_block, Bin::And, low, k_ffff);
                let crosses = self.s.m.cmp(wide_block, pred::IUGE, page_off, k_ffc0);
                let (cross_block, wide_done) = (self.s.m.new_block(), self.s.m.new_block());
                self.s
                    .m
                    .terminate(wide_block, Term::CondBr(crosses, cross_block, wide_done));

                self.cur = cross_block;
                let (low2, high2) = self.address_plus(low, high, 64);
                let (slot2, byte2, nonuniform) = self.bvh_memory_slot(low2, high2);
                let cross_end = self.cur;
                let k64 = self.s.m.i32(64);
                let byte2_adj = self.s.m.bin(cross_end, Bin::Sub, byte2, k64);
                self.s.m.terminate(cross_end, Term::Br(wide_done));

                self.cur = wide_done;
                let i32t = self.s.t.i32;
                let s_phi = self.s.m.phi(wide_done, i32t);
                self.s
                    .m
                    .set_phi(s_phi, vec![(slot, wide_block), (slot2, cross_end)]);
                let b_phi = self.s.m.phi(wide_done, i32t);
                self.s
                    .m
                    .set_phi(b_phi, vec![(byte, wide_block), (byte2_adj, cross_end)]);
                let h2 = self.bvh_memory_handle(s_phi, nonuniform);
                self.bvh_load_dwords(h2, b_phi, None, 64, 12, true)
            }
        };
        floats.extend(second12);
        let wide_end = self.cur;
        self.s.m.terminate(wide_end, Term::Br(join));

        // 16-bit floats: 24 halves in the 12 dwords already loaded from bytes
        // 16..64, the low half of a dword first (zero extra memory loads).
        self.cur = half_block;
        let words: Vec<V> = mid_floats.iter().map(|&f| self.as_int(f)).collect();
        let k16 = self.s.m.i32(16);
        let f32t = self.s.t.f32;
        let mut halves = Vec::with_capacity(24);
        for n in 0..24 {
            let mut h = words[n / 2];
            if n % 2 == 1 {
                h = self.s.m.bin(half_block, Bin::LShr, h, k16);
            }
            // The conversion reads the low 16 bits.
            halves.push(self.s.op(
                half_block,
                "legacyF16ToF32",
                Attr::ReadNone,
                f32t,
                131,
                &[h],
            ));
        }
        self.s.m.terminate(half_block, Term::Br(join));

        // Converged slab test: executed once for both 16-bit and 32-bit box
        // nodes so divergent waves share the ALU and icache footprint.
        self.cur = join;
        let mut coords = Vec::with_capacity(24);
        for n in 0..24 {
            let phi = self.s.m.phi(join, f32t);
            self.s
                .m
                .set_phi(phi, vec![(floats[n], wide_end), (halves[n], half_block)]);
            coords.push(phi);
        }
        // The inverse direction held finite ([`bvh::boxes`]).
        let (most, least) = (
            self.s.m.f32(f32::MAX.to_bits()),
            self.s.m.f32((-f32::MAX).to_bits()),
        );
        let inverse = ray.inverse.map(|x| {
            let x = self.binary(35, x, least);
            self.binary(36, x, most)
        });
        let (none, infinity, fzero) = (
            self.s.m.i32(bvh::NONE),
            self.s.m.f32(f32::INFINITY.to_bits()),
            self.s.m.f32(0),
        );
        let (abs_mask, inf_bits) = (self.s.m.i32(0x7FFF_FFFF), self.s.m.i32(0x7F80_0000));
        let mut ids = [none; 4];
        let mut distances = [infinity; 4];
        for n in 0..4 {
            let bx = [
                coords[6 * n],
                coords[6 * n + 1],
                coords[6 * n + 2],
                coords[6 * n + 3],
                coords[6 * n + 4],
                coords[6 * n + 5],
            ];
            (ids[n], distances[n]) = self.bvh_slab_test(
                bx,
                children[n],
                ray,
                inverse,
                none,
                infinity,
                fzero,
                abs_mask,
                inf_bits,
            );
        }
        if sort_mode != SortMode::Never {
            let pairs = [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)];
            for (step, &(a, b)) in pairs.iter().enumerate() {
                let mut swap = self.s.m.cmp(join, pred::OLT, distances[b], distances[a]);
                if let SortMode::Dynamic(sort_bit) = sort_mode {
                    swap = self.s.m.bin(join, Bin::And, swap, sort_bit);
                }
                if step + 1 < pairs.len() {
                    let (da, db) = (distances[a], distances[b]);
                    distances[a] = self.s.m.select(join, swap, db, da);
                    distances[b] = self.s.m.select(join, swap, da, db);
                }
                let (ia, ib) = (ids[a], ids[b]);
                ids[a] = self.s.m.select(join, swap, ib, ia);
                ids[b] = self.s.m.select(join, swap, ia, ib);
            }
        }
        ids
    }

    /// The triangle test ([`bvh::triangle`] of [`bvh::triangle_vertices`]):
    /// four result dwords, made in the current block.
    fn bvh_triangle(&mut self, at: NodeAt, kind: V, ray: &Ray) -> [V; 4] {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        // The triangle's byte of the dword at byte 60: which of its
        // vertices barycentric I and J weigh.
        let id = self.bvh_node_page0(at, None, 60, 1, false)[0];
        let (k2, k3, k12, k15) = (k(self, 2), k(self, 3), k(self, 12), k(self, 15));
        let shift = self.s.m.bin(c, Bin::Shl, kind, k3);
        let id = self.s.m.bin(c, Bin::LShr, id, shift);
        let i = self.s.m.bin(c, Bin::And, id, k3);
        let j = self.s.m.bin(c, Bin::LShr, id, k2);
        let j = self.s.m.bin(c, Bin::And, j, k3);
        let first = self.s.m.bin(c, Bin::Sub, k3, i);
        let first = self.s.m.bin(c, Bin::Sub, first, j);
        let first = self.s.m.bin(c, Bin::And, first, k3);
        // The node's vertices the triangle is made of, four bits each.
        let (k0, k1) = (k(self, 0), k(self, 1));
        let maps: [V; 4] = std::array::from_fn(|n| k(self, bvh::vertex_map(n as u32)));
        let (is0, is1, is2) = (
            self.s.m.cmp(c, pred::EQ, kind, k0),
            self.s.m.cmp(c, pred::EQ, kind, k1),
            self.s.m.cmp(c, pred::EQ, kind, k2),
        );
        let map = self.s.m.select(c, is2, maps[2], maps[3]);
        let map = self.s.m.select(c, is1, maps[1], map);
        let map = self.s.m.select(c, is0, maps[0], map);
        let mut p = Vec::with_capacity(3);
        for n in [first, i, j] {
            let nibble = self.s.m.bin(c, Bin::Shl, n, k2);
            let vertex = self.s.m.bin(c, Bin::LShr, map, nibble);
            let vertex = self.s.m.bin(c, Bin::And, vertex, k15);
            let byte = self.s.m.bin(c, Bin::Mul, vertex, k12);
            let floats = self.bvh_node_page0(at, Some(byte), 0, 3, true);
            // Relative to the ray's origin.
            let r: [V; 3] =
                std::array::from_fn(|a| self.s.m.bin(c, Bin::Sub, floats[a], ray.origin[a]));
            p.push(r);
        }
        // The ray's largest axis z, and x and y after it in an order that
        // keeps the winding.
        let fzero = self.s.m.f32(0);
        let size = ray.direction.map(|d| self.unary(6, d));
        let xy = self.s.m.cmp(c, pred::OGE, size[0], size[1]);
        let xz = self.s.m.cmp(c, pred::OGE, size[0], size[2]);
        let yz = self.s.m.cmp(c, pred::OGE, size[1], size[2]);
        let z0 = self.s.m.bin(c, Bin::And, xy, xz);
        let on = self.s.m.bool(true);
        let not_xy = self.s.m.bin(c, Bin::Xor, xy, on);
        let z1 = self.s.m.bin(c, Bin::And, not_xy, yz);
        // Component (kz + n) % 3 of `q`.
        let pick = |l: &mut Self, q: [V; 3], n: usize| {
            let rest = l.s.m.select(c, z1, q[(1 + n) % 3], q[(2 + n) % 3]);
            l.s.m.select(c, z0, q[n % 3], rest)
        };
        let dz = pick(self, ray.direction, 0);
        let backwards = self.s.m.cmp(c, pred::OLT, dz, fzero);
        let (dx0, dy0) = (pick(self, ray.direction, 1), pick(self, ray.direction, 2));
        let dx = self.s.m.select(c, backwards, dy0, dx0);
        let dy = self.s.m.select(c, backwards, dx0, dy0);
        // (x, y, z) of `q`: the two after the largest axis, swapped for a
        // ray that runs backwards along it.
        let axes = |l: &mut Self, q: [V; 3]| {
            let (x, y) = (pick(l, q, 1), pick(l, q, 2));
            [
                l.s.m.select(c, backwards, y, x),
                l.s.m.select(c, backwards, x, y),
                pick(l, q, 0),
            ]
        };
        let sz = pick(self, ray.inverse, 0);
        let sx = self.s.m.bin(c, Bin::Mul, dx, sz);
        let sy = self.s.m.bin(c, Bin::Mul, dy, sz);
        let mut sheared = Vec::with_capacity(3);
        for r in p {
            let r = axes(self, r);
            let x = self.s.m.bin(c, Bin::Mul, sx, r[2]);
            let x = self.s.m.bin(c, Bin::Sub, r[0], x);
            let y = self.s.m.bin(c, Bin::Mul, sy, r[2]);
            let y = self.s.m.bin(c, Bin::Sub, r[1], y);
            let z = self.s.m.bin(c, Bin::Mul, sz, r[2]);
            sheared.push((x, y, z));
        }
        let [(ax, ay, az), (bx, by, bz), (cx, cy, cz)] = sheared[..] else {
            unreachable!()
        };
        // p * q - r * s.
        let cross = |l: &mut Self, p: V, q: V, r: V, s: V| {
            let a = l.s.m.bin(c, Bin::Mul, p, q);
            let b = l.s.m.bin(c, Bin::Mul, r, s);
            l.s.m.bin(c, Bin::Sub, a, b)
        };
        let u = cross(self, cx, by, cy, bx);
        let v = cross(self, ax, cy, ay, cx);
        let w = cross(self, bx, ay, by, ax);
        let any = |l: &mut Self, p: u8| {
            let a = l.s.m.cmp(c, p, u, fzero);
            let b = l.s.m.cmp(c, p, v, fzero);
            let d = l.s.m.cmp(c, p, w, fzero);
            let a = l.s.m.bin(c, Bin::Or, a, b);
            l.s.m.bin(c, Bin::Or, a, d)
        };
        let behind = any(self, pred::OLT);
        let before = any(self, pred::OGT);
        let outside = self.s.m.bin(c, Bin::And, behind, before);
        let inside = self.s.m.bin(c, Bin::Xor, outside, on);
        let det = self.s.m.bin(c, Bin::Add, v, w);
        let det = self.s.m.bin(c, Bin::Add, u, det);
        let t = self.s.m.bin(c, Bin::Mul, u, az);
        let t2 = self.s.m.bin(c, Bin::Mul, v, bz);
        let t = self.s.m.bin(c, Bin::Add, t, t2);
        let t2 = self.s.m.bin(c, Bin::Mul, w, cz);
        let t = self.s.m.bin(c, Bin::Add, t, t2);
        // The distance is not negative: the numerator has the
        // denominator's sign (or is 0). Ordered compares: a denominator
        // of 0 or a NaN is a miss.
        let front = self.s.m.cmp(c, pred::OGT, det, fzero);
        let front_ahead = self.s.m.cmp(c, pred::OGE, t, fzero);
        let front = self.s.m.bin(c, Bin::And, front, front_ahead);
        let back = self.s.m.cmp(c, pred::OLT, det, fzero);
        let back_ahead = self.s.m.cmp(c, pred::OLE, t, fzero);
        let back = self.s.m.bin(c, Bin::And, back, back_ahead);
        let ahead = self.s.m.bin(c, Bin::Or, front, back);
        let hit = self.s.m.bin(c, Bin::And, inside, ahead);
        let miss = bvh::MISS.map(|f| self.s.m.f32(f.to_bits()));
        let out = [t, det, v, w];
        std::array::from_fn(|n| {
            let f = self.s.m.select(c, hit, out[n], miss[n]);
            self.as_int(f)
        })
    }
}
