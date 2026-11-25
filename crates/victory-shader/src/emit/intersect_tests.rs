//! The ray-tracing node test end to end: a synthetic compute program that
//! runs `image_bvh64_intersect_ray` (or the 32-bit form) once per thread,
//! lowered to DXIL, validated by `dxil.dll` when present, run on the host
//! GPU and WARP, and compared dword for dword with the CPU reference
//! ([`crate::bvh`]).
//!
//! The nodes are in a raw buffer. Two ways to name it are run: the
//! buffer's descriptor from the program's table (a resource of the host,
//! [`crate::resource::BvhNodes::Resource`]), and the title's way, a
//! descriptor of numbers over title memory with the structure's address in
//! the node pointer or in the descriptor. These tests have no host that
//! keeps copies of title memory and their directory, so for the second
//! they put a stand-in under the node reader
//! (`lower::intersect::TEST_MEMORY`); without it the same program reads
//! through the directory, which is only checked to lower and validate.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::lower::intersect::TEST_MEMORY;
use super::{HostShader, lower, validator};
use crate::bvh::{self, Ray, tests as nodes};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::resource::BvhNodes;
use crate::{ShaderError, io, ir, resource, structure};

/// Threads of one group, each with its own record.
const THREADS: u32 = 32;
/// Bytes of a thread's record: the node pointer (two dwords), the ray's
/// extent, origin, direction and inverse direction, and whether the
/// thread runs the instruction.
const RECORD: u32 = 64;
/// Where the nodes start in the buffer.
const NODES: u32 = THREADS * RECORD;
/// What the result registers hold before the instruction.
const BEFORE: [u32; 4] = [1, 2, 3, 4];

/// How the program names the nodes' memory.
#[derive(Clone, Copy, PartialEq)]
enum Descriptor {
    /// The buffer's descriptor from the table.
    Table,
    /// Numbers, as the title builds it: dwords 0 and 1 (the base address
    /// in 256-byte units and the box sorting bit); the size and the rest of
    /// the flags as the title sets them.
    Numbers(u32, u32),
}

/// The descriptor the title builds for the 64-bit form: the whole address
/// space, sorted boxes, barycentrics returned.
const WHOLE: Descriptor = Descriptor::Numbers(0, bvh::BOX_SORT);

/// The program: per thread, the record's twelve operand dwords in v1-v12,
/// the instruction's result in v13-v16 (under a lane mask of the threads
/// whose record says so), stored to the UAV at 16 bytes a thread.
fn program(wide: bool, descriptor: Descriptor) -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    for k in 0..4u32 {
        // v_mov_b32 v13+k, 1+k
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k));
    }
    let rsrc = match descriptor {
        Descriptor::Table => 8,
        Descriptor::Numbers(word0, word1) => {
            // s_mov_b32 s24, word0
            w.extend([0xBE98_03FF, word0]);
            if word1 == bvh::BOX_SORT {
                // As the title makes it.
                w.push(0xBE99_0B81); // s_brev_b32 s25, 1
            } else {
                w.extend([0xBE99_03FF, word1]); // s_mov_b32 s25, word1
            }
            w.extend([
                0xBE9A_03C1, // s_mov_b32 s26, -1
                0xBE9B_03FF,
                0x8900_03FF, // s_mov_b32 s27, 0x890003ff
            ]);
            24
        }
    };
    w.extend([
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
    ]);
    let first = 0x0000_0D01 | (rsrc >> 2) << 16;
    if wide {
        // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[rsrc:rsrc+3]
        w.extend([0xF19C_9F07, first, 0x0504_0302, 0x0908_0706, 0x000C_0B0A]);
    } else {
        // image_bvh_intersect_ray v[13:16], [v1, v3 .. v12], s[rsrc:rsrc+3]
        w.extend([0xF198_9F07, first, 0x0605_0403, 0x0A09_0807, 0x0000_0C0B]);
    }
    w.extend([
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0x342A_0084, // v_lshlrev_b32 v21, 4, v0
        0xE078_1000,
        0x8003_0D15, // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn native(words: &[u32]) -> NativeProgram {
    let table = |kind, at| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: at,
        lower_bound: 0,
        stride: 4,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 22,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![table(ResourceType::Srv, 0), table(ResourceType::Uav, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [THREADS, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// Lowers the program; the DXIL is validated when the validator is there.
fn build(
    wide: bool,
    descriptor: Descriptor,
) -> Result<(HostShader, resource::Resources), ShaderError> {
    let native = native(&program(wide, descriptor));
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    Ok((shader, res))
}

/// One thread's test: a node (by its index in the node list), the kind its
/// pointer carries, and the ray; `None` leaves the thread masked out.
type Case = Option<(usize, u32, Ray)>;

/// The scene: node dwords (a 32-bit box node takes 32, the others 16) and
/// where each node starts, in dwords from the first node.
struct Scene {
    words: Vec<u32>,
    starts: Vec<usize>,
}

impl Scene {
    fn new() -> Self {
        Self {
            words: Vec::new(),
            starts: Vec::new(),
        }
    }

    /// Adds a node of `dwords` (16 or 32) and returns its index.
    fn add(&mut self, node: [u32; 28], dwords: usize) -> usize {
        self.starts.push(self.words.len());
        let mut padded = [0u32; 32];
        padded[..28].copy_from_slice(&node);
        // Bytes the instruction must not read as part of the node: what
        // follows a 64-byte node, and a box node's last 16 bytes.
        for w in padded.iter_mut().skip(if dwords == 16 { 16 } else { 28 }) {
            *w = 0x7fc0_dead;
        }
        self.words.extend(&padded[..dwords]);
        self.starts.len() - 1
    }

    /// The pointer of node `n` with `kind`, counted from the first node.
    fn pointer(&self, n: usize, kind: u32) -> u32 {
        (4 * self.starts[n] as u32) >> 3 | kind
    }

    /// What the instruction reads of node `n`.
    fn node(&self, n: usize) -> [u32; 28] {
        std::array::from_fn(|k| self.words.get(self.starts[n] + k).copied().unwrap_or(0))
    }
}

/// A small generator of repeatable numbers.
struct Numbers(u64);

impl Numbers {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    /// A float in -1..1 in steps of 1/4096 (exactly representable).
    fn unit(&mut self) -> f32 {
        (self.next() % 8193) as f32 / 4096.0 - 1.0
    }

    fn vector(&mut self, scale: f32) -> [f32; 3] {
        [
            self.unit() * scale,
            self.unit() * scale,
            self.unit() * scale,
        ]
    }
}

/// The scene and batches of cases over it.
fn scene() -> (Scene, Vec<Vec<Case>>) {
    let mut s = Scene::new();
    let mut g = Numbers(0x5eed);
    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved =
        |b: [f32; 6], by: [f32; 3]| -> [f32; 6] { std::array::from_fn(|k| b[k] + by[k % 3]) };
    let off = [f32::NAN; 6];
    let mut box_nodes = Vec::new();
    // The boxes of the reference's own tests, in both widths.
    let lined = [
        moved(UNIT, [30.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 5.0, 0.0]),
        moved(UNIT, [20.0, 0.0, 0.0]),
    ];
    let ids = [0x105, 0x20c, 0x304, 0x400];
    box_nodes.push((s.add(nodes::box32(ids, lined), 32), 5));
    box_nodes.push((s.add(nodes::box16(ids, lined), 16), 4));
    // Boxes around the origin, one switched off, one child missing.
    let around = [
        UNIT,
        moved(UNIT, [0.5, 0.5, 0.5]),
        off,
        [-4.0, -0.25, -0.25, 4.0, 0.25, 0.25],
    ];
    let ids2 = [8, bvh::NONE, 24, 0x1fff_fffc];
    box_nodes.push((s.add(nodes::box32(ids2, around), 32), 5));
    box_nodes.push((s.add(nodes::box16(ids2, around), 16), 4));
    // Random boxes.
    for n in 0..6 {
        let boxes: [[f32; 6]; 4] = std::array::from_fn(|_| {
            let (c, h) = (g.vector(8.0), g.vector(2.0).map(f32::abs));
            std::array::from_fn(|k| c[k % 3] + if k < 3 { -h[k] } else { h[k - 3] })
        });
        let ids: [u32; 4] = std::array::from_fn(|k| 0x1000 * (n + 1) + 8 * k as u32 + 5);
        if n % 2 == 0 {
            box_nodes.push((s.add(nodes::box32(ids, boxes), 32), 5));
        } else {
            box_nodes.push((s.add(nodes::box16(ids, boxes), 16), 4));
        }
    }
    // Triangles: one of the reference's tests, a pair as the title's
    // builder stores it, all four triangles of a full node with turned
    // ids, random ones.
    let mut triangle_nodes = Vec::new();
    let front = [[0.0, 2.0, 5.0], [2.0, -2.0, 5.0], [-2.0, -2.0, 5.0]];
    triangle_nodes.push((s.add(nodes::triangles(&front, 9), 16), 0));
    let pair = s.add(
        nodes::triangles(
            &[front[0], front[1], front[2], [2.0, 2.0, 5.0]],
            9 | 0x80 | 6 << 8,
        ),
        16,
    );
    triangle_nodes.extend([(pair, 0), (pair, 1)]);
    let five = [
        [-3.0, -1.0, 6.0],
        [-1.0, 2.0, 6.5],
        [1.0, -1.0, 6.0],
        [3.0, 2.0, 6.5],
        [5.0, -1.0, 7.0],
    ];
    // I and J per triangle: (1, 2), (2, 0), (0, 1), (1, 2).
    let full = s.add(nodes::triangles(&five, 9 | 2 << 8 | 4 << 16 | 9 << 24), 16);
    triangle_nodes.extend((0..4).map(|k| (full, k)));
    for _ in 0..5 {
        let c = g.vector(6.0);
        let v: [[f32; 3]; 3] = std::array::from_fn(|_| {
            let d = g.vector(3.0);
            [c[0] + d[0], c[1] + d[1], c[2] + d[2]]
        });
        triangle_nodes.push((s.add(nodes::triangles(&v, 9), 16), 0));
    }
    let mut cases: Vec<Case> = Vec::new();
    // Rays of the reference's tests against the lined-up boxes.
    for &(n, kind) in &box_nodes[..2] {
        for (origin, direction, extent) in [
            ([0.0; 3], [1.0, 0.0, 0.0], 1000.0),
            ([0.0; 3], [1.0, 0.0, 0.0], 19.5),
            ([0.0; 3], [1.0, 0.0, 0.0], 19.0),
            ([0.0; 3], [-1.0, 0.0, 0.0], 1000.0),
            ([40.0, 0.0, 0.0], [-2.0, 0.0, 0.0], 1000.0),
            ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], 1000.0),
            ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 1000.0),
            ([0.0, 0.5, 0.25], [1.0, -0.0, 0.0], 1000.0),
            ([0.0; 3], [0.0; 3], 1000.0),
        ] {
            cases.push(Some((n, kind, Ray::new(origin, direction, extent))));
        }
    }
    // Random rays at every box node, half of them aimed at a box.
    for &(n, kind) in &box_nodes {
        let words = s.node(n);
        let boxes = if kind == 5 {
            bvh::boxes32(&words)
        } else {
            bvh::boxes16(&words)
        };
        for k in 0..12 {
            let origin = g.vector(12.0);
            let direction = if k % 2 == 0 {
                let b = boxes[(k / 2) % 4];
                if b[0].is_nan() {
                    g.vector(1.0)
                } else {
                    std::array::from_fn(|a| (b[a] + b[3 + a]) / 2.0 - origin[a] + g.unit() / 2.0)
                }
            } else {
                g.vector(1.0)
            };
            let extent = if k % 3 == 0 { 6.0 } else { 100.0 };
            cases.push(Some((n, kind, Ray::new(origin, direction, extent))));
        }
    }
    // Rays at every triangle: aimed inside, at an edge, at a vertex, past
    // it, from both sides, along each axis.
    for &(n, kind) in &triangle_nodes {
        let p = bvh::triangle_vertices(&s.node(n), kind);
        let at = |w: [f32; 3]| -> [f32; 3] {
            std::array::from_fn(|a| w[0] * p[0][a] + w[1] * p[1][a] + w[2] * p[2][a])
        };
        for (k, w) in [
            [0.5, 0.25, 0.25],
            [0.125, 0.75, 0.125],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.75, 0.75, -0.5],
            [-0.25, 0.5, 0.75],
            [0.25, 0.25, 0.5],
        ]
        .into_iter()
        .enumerate()
        {
            let target = at(w);
            let origin = if k % 2 == 0 { [0.0; 3] } else { g.vector(16.0) };
            let mut direction: [f32; 3] = std::array::from_fn(|a| target[a] - origin[a]);
            if k == 7 {
                // The triangle lies behind.
                direction = direction.map(|d| -d);
            }
            cases.push(Some((n, kind, Ray::new(origin, direction, 100.0))));
        }
        for axis in 0..3 {
            for sign in [1.0f32, -1.0] {
                let target = at([0.25, 0.5, 0.25]);
                let mut origin = target;
                origin[axis] -= 8.0 * sign;
                let mut direction = [0.0; 3];
                direction[axis] = sign;
                cases.push(Some((n, kind, Ray::new(origin, direction, 100.0))));
            }
        }
    }
    // Kinds the instruction does not read, and threads masked out.
    let ray = Ray::new([0.0; 3], [0.0, 0.0, 1.0], 100.0);
    cases.extend([
        Some((triangle_nodes[0].0, 6, ray)),
        Some((triangle_nodes[0].0, 7, ray)),
        None,
        Some((
            box_nodes[0].0,
            5,
            Ray::new([0.0; 3], [1.0, 0.0, 0.0], 1000.0),
        )),
        None,
    ]);
    let batches = cases
        .chunks(THREADS as usize)
        .map(<[Case]>::to_vec)
        .collect();
    (s, batches)
}

/// The buffer for one batch: the threads' records, then the nodes. Node
/// pointers count from `base` before the first node, in bytes.
fn buffer(scene: &Scene, batch: &[Case], base: u64) -> Vec<u32> {
    let mut data = vec![0u32; (NODES / 4) as usize];
    for (t, case) in batch.iter().enumerate() {
        let Some((n, kind, ray)) = case else { continue };
        let r = &mut data[t * (RECORD / 4) as usize..][..13];
        let pointer = (base >> 3) + u64::from(scene.pointer(*n, *kind));
        (r[0], r[1]) = (pointer as u32, (pointer >> 32) as u32);
        r[2] = ray.extent.to_bits();
        for a in 0..3 {
            r[3 + a] = ray.origin[a].to_bits();
            r[6 + a] = ray.direction[a].to_bits();
            r[9 + a] = ray.inverse[a].to_bits();
        }
        r[12] = 1;
    }
    data.extend(&scene.words);
    data
}

fn expected(scene: &Scene, case: &Case, sort: bool) -> [u32; 4] {
    match case {
        Some((n, kind, ray)) => bvh::intersect(&scene.node(*n), *kind, ray, sort),
        None => BEFORE,
    }
}

/// Sets the tests' title memory for this thread until dropped.
struct Memory;

impl Memory {
    fn at(base: u64) -> Self {
        // Heap slot 0: the buffer of records and nodes.
        TEST_MEMORY.set(Some((0, base)));
        Self
    }
}

impl Drop for Memory {
    fn drop(&mut self) {
        TEST_MEMORY.set(None);
    }
}

/// Runs every batch on the hardware adapter and WARP and compares with
/// the reference. `pointers` is what node pointers count from, in bytes
/// before the first node.
fn run(wide: bool, descriptor: Descriptor, pointers: u64, sort: bool) {
    let (shader, res) = build(wide, descriptor).expect("the program lowers");
    assert_eq!(res.bvh.len(), 1);
    let (scene, batches) = scene();
    // The reference must see a bit of everything, or the comparison
    // proves little.
    let all: Vec<&Case> = batches.iter().flatten().collect();
    let results: Vec<[u32; 4]> = all.iter().map(|c| expected(&scene, c, sort)).collect();
    let boxes = |hits: usize| {
        all.iter()
            .zip(&results)
            .filter(|(c, r)| {
                matches!(c, Some((_, 4 | 5, _)))
                    && r.iter().filter(|&&x| x != bvh::NONE).count() == hits
            })
            .count()
    };
    let triangles = |hit: bool| {
        all.iter()
            .zip(&results)
            .filter(|(c, r)| {
                matches!(c, Some((_, 0..=3, _))) && (r[0] != f32::INFINITY.to_bits()) == hit
            })
            .count()
    };
    assert!(
        boxes(0) > 5 && boxes(1) > 5 && boxes(2) > 5 && boxes(3) > 0,
        "box cases"
    );
    assert!(
        triangles(true) > 30 && triangles(false) > 30,
        "triangle cases"
    );
    // Sorting must matter to some case, or the unsorted run proves nothing.
    assert!(
        all.iter()
            .any(|c| expected(&scene, c, true) != expected(&scene, c, false)),
        "no case tells sorted from unsorted"
    );
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (b, batch) in batches.iter().enumerate() {
            let data = buffer(&scene, batch, pointers);
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0; 5]))],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Raw,
                            },
                            View::RawUav(4 * THREADS),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            for t in 0..THREADS as usize {
                let case = batch.get(t).copied().flatten();
                let want = match batch.get(t) {
                    Some(c) => expected(&scene, c, sort),
                    // A thread past the batch: its record is zeros, so it
                    // is masked out.
                    None => BEFORE,
                };
                let got = &out[0][4 * t..4 * t + 4];
                assert_eq!(
                    got,
                    want,
                    "{label}, batch {b} thread {t}: {case:x?}\n got {:x?} = {:?}\nwant {:x?} = {:?}",
                    got,
                    got.iter().map(|&x| f32::from_bits(x)).collect::<Vec<_>>(),
                    want,
                    want.map(f32::from_bits),
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn the_64_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    // Pointers count from the buffer's start.
    run(true, Descriptor::Table, u64::from(NODES), true);
    assert_eq!(
        build(true, Descriptor::Table).unwrap().1.bvh[0].1,
        BvhNodes::Resource
    );
}

#[test]
fn the_32_bit_node_test_in_a_host_buffer_matches_the_reference_on_the_card() {
    run(false, Descriptor::Table, u64::from(NODES), true);
}

/// Where the tests' title memory puts the buffer: past 4 GiB, on 256 bytes.
const BASE: u64 = 0x12_3456_7800;

#[test]
fn the_64_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's way: one descriptor for all of memory, the structure's
    // address folded into a 64-bit pointer.
    let _memory = Memory::at(BASE);
    run(true, WHOLE, BASE + u64::from(NODES), true);
}

#[test]
fn the_32_bit_node_test_over_title_memory_matches_the_reference_on_the_card() {
    // The title's other way: the structure's address in the descriptor
    // (in 256-byte units, 40 bits), a 32-bit pointer from there. Here with
    // an address past 2^40 bytes, and without box sorting.
    let base = 0x312_3456_7800u64;
    let structure = base + u64::from(NODES);
    let _memory = Memory::at(base);
    let descriptor = Descriptor::Numbers((structure >> 8) as u32, (structure >> 40) as u32);
    run(false, descriptor, 0, false);
    // And sorted.
    let descriptor = Descriptor::Numbers(
        (structure >> 8) as u32,
        (structure >> 40) as u32 | bvh::BOX_SORT,
    );
    run(false, descriptor, 0, true);
}

#[test]
fn a_descriptor_of_numbers_is_title_memory_read_through_the_directory() {
    for wide in [true, false] {
        let native = native(&program(wide, WHOLE));
        let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("tracked");
        // The flags the title sets: sorted boxes, barycentrics, a
        // ray-tracing structure. No resource is taken for it: the buffer
        // loads' and the store's descriptors are all there is.
        assert_eq!(res.bvh.len(), 1);
        assert_eq!(
            res.bvh[0].1,
            BvhNodes::Memory {
                sort: Some(true),
                zero_base: true,
                word3: 0x8900_03ff
            }
        );
        assert_eq!(res.descriptors.len(), 5);
        // Without the tests' stand-in the nodes are read as the title's
        // programs will read them: through the host's directory of title
        // memory. The program lowers and validates, and asks the host for
        // the directory and for the list of pages it found no copy of.
        let (shader, _) = build(wide, WHOLE).expect("lowers through the directory");
        assert!(shader.root_srvs.contains(&super::SLOT_MEMORY));
        assert!(shader.root_uavs.contains(&super::SLOT_FAULTS));
    }
}

#[test]
fn a_structure_that_returns_triangle_ids_is_refused_by_name() {
    // Triangle return mode 0 (bit 24 of dword 3 clear): the last two
    // result dwords would be a triangle id and a hit status.
    let mut words = program(true, WHOLE);
    let flags = words.iter().position(|&w| w == 0x8900_03FF).unwrap();
    words[flags] = 0x8800_03FF;
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("tracked");
    let io = io::raise(&p, &native).expect("io::raise");
    let r = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    match r {
        Err(ShaderError::Unsupported { what, .. }) => {
            assert_eq!(what, "a ray-tracing structure that returns triangle ids");
        }
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// A looped BVH traversal compute program (32 threads/group, `iters` iterations
/// per thread) that runs `image_bvh64_intersect_ray` inside a loop with a
/// lane-mask guard, accumulating results into `v[18:21]` via `v_add_nc_u32` and
/// advancing the node pointer across a 16-node ring (128 bytes per slot =
/// pointer stride 16) so the driver cannot hoist or eliminate node tests.
fn looped_program(descriptor: Descriptor, iters: u32, ring_mask_ptr: u32) -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (records/nodes SRV)
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (output UAV)
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0              (64 * lane)
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    // Initialize accumulator v[18:21] = 0, and v[13:16] = BEFORE ([1,2,3,4]).
    for k in 0..4u32 {
        w.push(0x7E00_0280 | (18 + k) << 17); // v_mov_b32 v18+k, 0
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k)); // v_mov_b32 v13+k, 1+k
    }
    // v22 = v1 (base low pointer), v23 = 0 (ring offset)
    w.push(0x7E00_0301 | (22 << 17)); // v_mov_b32 v22, v1
    w.push(0x7E00_0280 | (23 << 17)); // v_mov_b32 v23, 0
    let rsrc = match descriptor {
        Descriptor::Table => 8,
        Descriptor::Numbers(word0, word1) => {
            w.extend([0xBE98_03FF, word0]);
            if word1 == bvh::BOX_SORT {
                w.push(0xBE99_0B81); // s_brev_b32 s25, 1
            } else {
                w.extend([0xBE99_03FF, word1]);
            }
            w.extend([
                0xBE9A_03C1, // s_mov_b32 s26, -1
                0xBE9B_03FF,
                0x8900_03FF, // s_mov_b32 s27, 0x890003ff
            ]);
            24
        }
    };
    // s22 = 0 (loop counter)
    w.push(0xBE96_0380); // s_mov_b32 s22, 0
    let loop_top = w.len();
    // v1 = v22 + v23 (step node pointer within ring)
    w.push(0x4A00_0000 | (1 << 17) | (23 << 9) | (0x100 | 22)); // v_add_nc_u32 v1, v22, v23
    w.extend([
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
    ]);
    let first = 0x0000_0D01 | (rsrc >> 2) << 16;
    // image_bvh64_intersect_ray v[13:16], [v1 .. v12], s[rsrc:rsrc+3]
    w.extend([0xF19C_9F07, first, 0x0504_0302, 0x0908_0706, 0x000C_0B0A]);
    w.push(0xBEFE_0314); // s_mov_b32 exec_lo, s20
    // Accumulate v[18:21] += v[13:16] (wrapping add so repeated hits never cancel)
    for k in 0..4u32 {
        let dst = 18 + k;
        let src = 13 + k;
        w.push(0x4A00_0000 | (dst << 17) | (dst << 9) | (0x100 | src));
    }
    // v23 = (v23 + 16) & ring_mask_ptr (16 pointer units = 128 bytes)
    w.push(0x4A00_0000 | (23 << 17) | (23 << 9) | (0x80 + 16)); // v_add_nc_u32 v23, 16, v23
    w.extend([
        0x3600_0000 | (23 << 17) | (23 << 9) | 0xFF, // v_and_b32 v23, lit, v23
        ring_mask_ptr,
    ]);
    // s22 += 1; if s22 < iters goto loop_top
    w.push(0x8016_8116); // s_add_u32 s22, s22, 1
    w.extend([0xBF0A_FF16, iters]); // s_cmp_lt_u32 s22, iters
    let br_idx = w.len();
    let delta = (loop_top as isize) - (br_idx as isize) - 1;
    w.push(0xBF85_0000 | (delta as i16 as u16 as u32)); // s_cbranch_scc1 loop_top
    // Store accumulated v[18:21] at 16 * lane (using v24 so v21 is preserved)
    w.extend([
        0x3430_0084, // v_lshlrev_b32 v24, 4, v0
        0xE078_1000,
        0x8003_1218, // buffer_store_dwordx4 v[18:21], v24, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

/// Runs a compute shader with optional real `SLOT_MEMORY` / `SLOT_FAULTS`
/// buffers on hardware D3D12, returning `(output_uav, faults_uav, median_us)`.
#[allow(clippy::too_many_arguments)]
fn dispatch_memory_timed(
    host: &Host,
    shader: &HostShader,
    records_and_nodes: &[u32],
    memory_pages: &[(u32, u32, u32)], // (page_idx, copy_id, byte_offset)
    extra_copy_buffers: &[&[u32]],    // copy 1 = records_and_nodes, copy 2+ = extras
    out_dwords: u32,
    groups: [u32; 3],
    warmup: usize,
    timed: usize,
) -> Result<(Vec<u32>, Vec<u32>, f64), String> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::core::Interface;

    let (root, pso) = host.compute_pipeline(shader)?;
    let dev = &host.device;
    let err = |e: windows::core::Error| e.message().to_string();
    unsafe {
        let queue: ID3D12CommandQueue = dev
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })
            .map_err(err)?;
        let freq = queue.GetTimestampFrequency().map_err(err)?;
        let alloc: ID3D12CommandAllocator = dev
            .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
            .map_err(err)?;
        let list: ID3D12GraphicsCommandList = dev
            .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &pso)
            .map_err(err)?;
        let make = |heap: D3D12_HEAP_TYPE, size: u64, flags, state| -> Result<ID3D12Resource, String> {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &desc,
                state,
                None,
                &mut r,
            )
            .map_err(err)?;
            r.ok_or_else(|| "no resource".to_string())
        };
        let upload_to_default = |list: &ID3D12GraphicsCommandList,
                                 data: &[u32],
                                 flags: D3D12_RESOURCE_FLAGS,
                                 after: D3D12_RESOURCE_STATES,
                                 keep: &mut Vec<ID3D12Resource>|
         -> Result<ID3D12Resource, String> {
            let bytes = (data.len() * 4).max(256) as u64;
            let up = make(
                D3D12_HEAP_TYPE_UPLOAD,
                bytes,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )?;
            let mut p = std::ptr::null_mut();
            up.Map(0, None, Some(&mut p)).map_err(err)?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), p.cast::<u32>(), data.len());
            up.Unmap(0, None);
            let dst = make(
                D3D12_HEAP_TYPE_DEFAULT,
                bytes,
                flags,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            list.CopyBufferRegion(&dst, 0, &up, 0, bytes);
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&dst),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                        StateAfter: after,
                    }),
                },
            };
            list.ResourceBarrier(&[barrier]);
            keep.push(up);
            Ok(dst)
        };

        let mut keep = Vec::new();
        // Heap layout:
        // slot 0: SRV (records_and_nodes)
        // slot 1: UAV (output)
        // slot 2: null raw UAV (USER_MEMORY_BASE)
        // slot 3: copy 1 raw UAV (records_and_nodes)
        // slot 4+: extra copy raw UAVs
        let num_views = 4 + extra_copy_buffers.len();
        let views: ID3D12DescriptorHeap = dev
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: num_views as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .map_err(err)?;
        let view_size =
            dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let cpu_at = |slot: usize| {
            let mut h = views.GetCPUDescriptorHandleForHeapStart();
            h.ptr += slot * view_size;
            h
        };
        let copy1 = upload_to_default(
            &list,
            records_and_nodes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            &mut keep,
        )?;
        // Also keep a dedicated SRV copy for slot 0 so D3D12 debug layer sees no SRV+UAV state conflict.
        let srv_buf = upload_to_default(
            &list,
            records_and_nodes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            &mut keep,
        )?;
        dev.CreateShaderResourceView(
            &srv_buf,
            Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV {
                        FirstElement: 0,
                        NumElements: records_and_nodes.len() as u32,
                        StructureByteStride: 0,
                        Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                    },
                },
            }),
            cpu_at(0),
        );
        let out_bytes = u64::from(out_dwords) * 4;
        let out_buf = make(
            D3D12_HEAP_TYPE_DEFAULT,
            out_bytes,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        dev.CreateUnorderedAccessView(
            &out_buf,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: out_dwords,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            cpu_at(1),
        );
        // slot 2: null raw UAV
        dev.CreateUnorderedAccessView(
            None,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        NumElements: 1,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            cpu_at(2),
        );
        // slot 3: copy 1 raw UAV
        dev.CreateUnorderedAccessView(
            &copy1,
            None,
            Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_TYPELESS,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV {
                        FirstElement: 0,
                        NumElements: records_and_nodes.len() as u32,
                        Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        ..Default::default()
                    },
                },
            }),
            cpu_at(3),
        );
        for (idx, extra) in extra_copy_buffers.iter().enumerate() {
            let eb = upload_to_default(
                &list,
                extra,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                &mut keep,
            )?;
            dev.CreateUnorderedAccessView(
                &eb,
                None,
                Some(&D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: extra.len() as u32,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                            ..Default::default()
                        },
                    },
                }),
                cpu_at(4 + idx),
            );
            keep.push(eb);
        }

        let uses_memory = shader.root_srvs.contains(&super::SLOT_MEMORY);
        let (dir_buf, faults_buf, faults_rb) = if uses_memory {
            let mut dir_data = vec![0u32; (super::MEMORY_PAGES as usize) * 2];
            for &(page, copy_id, off) in memory_pages {
                dir_data[(page as usize) * 2] = copy_id;
                dir_data[(page as usize) * 2 + 1] = off;
            }
            let dir = upload_to_default(
                &list,
                &dir_data,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                &mut keep,
            )?;
            let zeros = vec![0u32; super::MEMORY_FAULTS as usize];
            let faults = upload_to_default(
                &list,
                &zeros,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                &mut keep,
            )?;
            let frb = make(
                D3D12_HEAP_TYPE_READBACK,
                u64::from(super::MEMORY_FAULTS) * 4,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )?;
            (Some(dir), Some(faults), Some(frb))
        } else {
            (None, None, None)
        };

        let out_rb = make(
            D3D12_HEAP_TYPE_READBACK,
            out_bytes,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let qheap: ID3D12QueryHeap = {
            let mut q: Option<ID3D12QueryHeap> = None;
            dev.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut q,
            )
            .map_err(err)?;
            q.ok_or_else(|| "no query heap".to_string())?
        };
        let ts_rb = make(
            D3D12_HEAP_TYPE_READBACK,
            16,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

        // Prepare user_data constants buffer matching HostShader::user_data.
        let ud_dwords = shader.user_data_dwords().max(1);
        let mut ud = vec![0u32; ud_dwords as usize];
        for u in &shader.user_data {
            if u.root == RootType::Other(super::USER_MEMORY_BASE) {
                ud[u.index as usize] = 2; // slot 2 = null raw UAV
            }
        }
        // Finish upload list first.
        list.Close().map_err(err)?;
        let fence: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
        let mut fval = 1u64;
        queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
        queue.Signal(&fence, fval).map_err(err)?;
        while fence.GetCompletedValue() < fval {
            std::thread::yield_now();
        }

        let mut samples_us = Vec::with_capacity(timed);
        for iter in 0..(warmup + timed) {
            alloc.Reset().map_err(err)?;
            list.Reset(&alloc, &pso).map_err(err)?;
            list.SetDescriptorHeaps(&[Some(views.clone())]);
            list.SetComputeRootSignature(&root);
            list.SetComputeRoot32BitConstants(0, ud_dwords, ud.as_ptr().cast(), 0);
            if let (Some(f), Some(d)) = (&faults_buf, &dir_buf) {
                list.SetComputeRootUnorderedAccessView(1, f.GetGPUVirtualAddress());
                list.SetComputeRootShaderResourceView(2, d.GetGPUVirtualAddress());
            }
            list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list.Dispatch(groups[0], groups[1], groups[2]);
            list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list.ResolveQueryData(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &ts_rb, 0);
            let last = iter + 1 == warmup + timed;
            if last {
                let to_copy = |r: &ID3D12Resource| D3D12_RESOURCE_BARRIER {
                    Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                    Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                    Anonymous: D3D12_RESOURCE_BARRIER_0 {
                        Transition: std::mem::ManuallyDrop::new(
                            D3D12_RESOURCE_TRANSITION_BARRIER {
                                pResource: std::mem::transmute_copy(r),
                                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                            },
                        ),
                    },
                };
                list.ResourceBarrier(&[to_copy(&out_buf)]);
                list.CopyBufferRegion(&out_rb, 0, &out_buf, 0, out_bytes);
                if let (Some(f), Some(frb)) = (&faults_buf, &faults_rb) {
                    list.ResourceBarrier(&[to_copy(f)]);
                    list.CopyBufferRegion(
                        frb,
                        0,
                        f,
                        0,
                        u64::from(super::MEMORY_FAULTS) * 4,
                    );
                }
            }
            list.Close().map_err(err)?;
            fval += 1;
            queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            queue.Signal(&fence, fval).map_err(err)?;
            while fence.GetCompletedValue() < fval {
                if let Err(e) = dev.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
            if iter >= warmup {
                let mut p = std::ptr::null_mut();
                ts_rb.Map(0, None, Some(&mut p)).map_err(err)?;
                let ts = std::slice::from_raw_parts(p.cast::<u64>(), 2);
                let dt = ts[1].saturating_sub(ts[0]);
                ts_rb.Unmap(0, None);
                samples_us.push((dt as f64) * 1e6 / (freq as f64));
            }
        }
        samples_us.sort_by(|a, b| a.total_cmp(b));
        let median = samples_us[samples_us.len() / 2];

        let mut p = std::ptr::null_mut();
        out_rb.Map(0, None, Some(&mut p)).map_err(err)?;
        let out_words = std::slice::from_raw_parts(p.cast::<u32>(), out_dwords as usize).to_vec();
        out_rb.Unmap(0, None);

        let fault_words = if let Some(frb) = &faults_rb {
            let mut fp = std::ptr::null_mut();
            frb.Map(0, None, Some(&mut fp)).map_err(err)?;
            let fw = std::slice::from_raw_parts(
                fp.cast::<u32>(),
                super::MEMORY_FAULTS as usize,
            )
            .to_vec();
            frb.Unmap(0, None);
            fw
        } else {
            Vec::new()
        };
        drop((copy1, srv_buf, keep));
        Ok((out_words, fault_words, median))
    }
}

#[test]
#[ignore]
fn bench_bvh_intersect_gpu() {
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = Host::new(false).expect("hardware SM 6.6 GPU required");
    let build_looped = |desc: Descriptor, iters: u32, ring_mask: u32| {
        let mut nat = native(&looped_program(desc, iters, ring_mask));
        nat.vgprs = 26;
        let p = ir::build(&nat.code, nat.wave64, nat.vgprs).expect("ir::build");
        let st = structure::structure(&p).expect("structure");
        let res = resource::track(&p, &nat.layout, nat.stage, Some(&st)).expect("track");
        let io = io::raise(&p, &nat).expect("io::raise");
        let lowered = lower::lower(&lower::Parts {
            native: &nat,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        let shader = HostShader {
            stage: HwStage::Cs,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            root_uavs: lowered.root_uavs,
            root_indexed_views: res.root_indexed_views.clone(),
        };
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let census = super::census::take(&shader.dxil);
        (shader, census)
    };

    // Build a 16-slot ring of 128-byte-spaced nodes per workload (starting at NODES = 2048).
    const ITERS: u32 = 64;
    const RING_NODES: usize = 16;
    const RING_MASK: u32 = (RING_NODES as u32 - 1) << 4; // 16-unit steps -> 0xF0
    const GROUPS: [u32; 3] = [4096, 1, 1]; // 131,072 rays x 64 iters = 8.39M node tests
    const TITLE_BASE: u64 = 0x1_0000; // page 1

    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved = |b: [f32; 6], by: [f32; 3]| -> [f32; 6] {
        std::array::from_fn(|k| b[k] + by[k % 3])
    };
    let lined = [
        moved(UNIT, [30.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 0.0, 0.0]),
        moved(UNIT, [10.0, 5.0, 0.0]),
        moved(UNIT, [20.0, 0.0, 0.0]),
    ];
    let ids = [0x105, 0x20c, 0x304, 0x400];
    let tri_verts = [[0.0, 2.0, 5.0], [2.0, -2.0, 5.0], [-2.0, -2.0, 5.0]];

    let make_workload = |mode: &str, base_addr: u64| -> (Vec<u32>, Vec<[u32; 4]>) {
        let mut ring = Vec::with_capacity(RING_NODES);
        for k in 0..RING_NODES {
            let shift = (k as f32) * 0.25;
            let bx = [
                moved(lined[0], [shift, 0.0, 0.0]),
                moved(lined[1], [0.0, shift * 0.1, 0.0]),
                moved(lined[2], [0.0, 0.0, shift * 0.1]),
                moved(lined[3], [-shift * 0.2, 0.0, 0.0]),
            ];
            let node = match mode {
                "box32" => nodes::box32(ids, bx),
                "box16" => nodes::box16(ids, bx),
                "tri" => nodes::triangles(
                    &[
                        [tri_verts[0][0] + shift * 0.05, tri_verts[0][1], tri_verts[0][2]],
                        tri_verts[1],
                        tri_verts[2],
                    ],
                    9,
                ),
                "divergent" => match k % 4 {
                    0 => nodes::box32(ids, bx),
                    1 => nodes::box16(ids, bx),
                    _ => nodes::triangles(&tri_verts, 9),
                },
                _ => unreachable!(),
            };
            ring.push(node);
        }
        let mut data = vec![0u32; (NODES / 4) as usize];
        let mut expect = Vec::with_capacity(THREADS as usize);
        for t in 0..THREADS as usize {
            let (kind, active, ray) = match mode {
                "box32" => (5u32, true, Ray::new([0.0; 3], [1.0, (t as f32 - 16.0) * 0.005, 0.0], 1000.0)),
                "box16" => (4u32, true, Ray::new([0.0; 3], [1.0, (t as f32 - 16.0) * 0.005, 0.0], 1000.0)),
                "tri" => (
                    0u32,
                    true,
                    Ray::new([(t as f32 - 16.0) * 0.05, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0),
                ),
                "divergent" => {
                    // Half lanes masked out; among active lanes mix box32, box16, tri, and kind 6/7
                    let active = t < 16;
                    let kind = match t % 4 {
                        0 => 5,
                        1 => 4,
                        2 => 0,
                        _ => 6 + ((t / 4) as u32 & 1),
                    };
                    let ray = if kind == 0 {
                        Ray::new([0.0; 3], [0.0, 0.0, 1.0], 100.0)
                    } else {
                        Ray::new([0.0; 3], [1.0, 0.0, 0.0], 1000.0)
                    };
                    (kind, active, ray)
                }
                _ => unreachable!(),
            };
            // For masked-out lanes or kind 6/7 in Memory mode, point to unmapped page 999 to catch spurious faults!
            let ptr_base = if mode == "divergent" && (!active || kind >= 6) {
                (999u64 * 0x1_0000) >> 3
            } else {
                (base_addr + u64::from(NODES)) >> 3
            };
            let ptr = ptr_base | u64::from(kind);
            let r = &mut data[t * (RECORD / 4) as usize..][..13];
            r[0] = ptr as u32;
            r[1] = (ptr >> 32) as u32;
            r[2] = ray.extent.to_bits();
            for a in 0..3 {
                r[3 + a] = ray.origin[a].to_bits();
                r[6 + a] = ray.direction[a].to_bits();
                r[9 + a] = ray.inverse[a].to_bits();
            }
            r[12] = u32::from(active);
            let mut acc = [0u32; 4];
            let mut cur = BEFORE;
            let mut step = 0u32;
            for _ in 0..ITERS {
                if active {
                    let idx = ((step >> 4) as usize) % RING_NODES;
                    cur = bvh::intersect(&ring[idx], kind, &ray, true);
                }
                for c in 0..4 {
                    acc[c] = acc[c].wrapping_add(cur[c]);
                }
                step = step.wrapping_add(16) & RING_MASK;
            }
            expect.push(acc);
        }
        for node in &ring {
            let mut padded = [0u32; 32];
            padded[..28].copy_from_slice(node);
            data.extend_from_slice(&padded);
        }
        (data, expect)
    };

    let (mem_shader, mem_census) = build_looped(WHOLE, ITERS, RING_MASK);
    let (res_shader, res_census) = build_looped(Descriptor::Table, ITERS, RING_MASK);
    let fmt_census = |c: &Option<super::census::Census>| {
        if let Some(c) = c {
            let idx = |name: &str| super::census::KINDS.iter().position(|&k| k == name).unwrap();
            format!(
                "total={} arith={} math={} sel={} phi={} bload={} bstore={} handle={} blks={}",
                c.counts[idx("total")],
                c.counts[idx("arith")],
                c.counts[idx("math_call")],
                c.counts[idx("select")],
                c.counts[idx("phi")],
                c.counts[idx("buffer_load")],
                c.counts[idx("buffer_store")],
                c.counts[idx("handle")],
                c.counts[idx("blocks")],
            )
        } else {
            "census_off".to_string()
        }
    };
    eprintln!(
        "DXIL Memory:   bytes={} {}",
        mem_shader.dxil.len(),
        fmt_census(&mem_census)
    );
    eprintln!(
        "DXIL Resource: bytes={} {}",
        res_shader.dxil.len(),
        fmt_census(&res_census)
    );

    for (label, mode, desc) in [
        ("box32_memory", "box32", WHOLE),
        ("box16_memory", "box16", WHOLE),
        ("divergent_memory", "divergent", WHOLE),
        ("tri_memory", "tri", WHOLE),
        ("box32_resource", "box32", Descriptor::Table),
        ("tri_resource", "tri", Descriptor::Table),
    ] {
        let (shader, base_addr) = if desc == WHOLE {
            (&mem_shader, TITLE_BASE)
        } else {
            (&res_shader, 0)
        };
        let (data, expect) = make_workload(mode, base_addr);
        let (out, faults, median_us) = dispatch_memory_timed(
            &host,
            shader,
            &data,
            &[(1, 1, 0)], // page 1 -> copy 1 at byte offset 0
            &[],
            4 * THREADS,
            GROUPS,
            5,
            25,
        )
        .expect("GPU benchmark dispatch failed");
        for t in 0..THREADS as usize {
            let got = &out[4 * t..4 * t + 4];
            assert_eq!(got, expect[t], "{label} thread {t} parity mismatch");
        }
        let nonzero_faults = faults.iter().skip(1).filter(|&&f| f != 0).count();
        eprintln!(
            "BENCH {label:<18}: median={median_us:8.2} us  (8.39M tests, parity=OK, spurious_faults={nonzero_faults})"
        );
    }
}

/// Builds a compute program for `image_bvh_intersect_ray` / `image_bvh64_intersect_ray`
/// with custom descriptor setup instructions (`desc_setup` writing `s[24:27]`) and
/// custom `dmask`.
fn custom_bvh_program(wide: bool, desc_setup: &[u32], dmask: u32) -> Vec<u32> {
    let mut w = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
        0xF408_0302,
        0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
        0x3428_0086, // v_lshlrev_b32 v20, 6, v0
        0xE038_1000,
        0x8002_0114, // buffer_load_dwordx4 v[1:4], v20, s[8:11], 0 offen
        0xE038_1010,
        0x8002_0514, // buffer_load_dwordx4 v[5:8], v20, s[8:11], 0 offen offset:16
        0xE038_1020,
        0x8002_0914, // buffer_load_dwordx4 v[9:12], v20, s[8:11], 0 offen offset:32
        0xE030_1030,
        0x8002_1114, // buffer_load_dword v17, v20, s[8:11], 0 offen offset:48
    ];
    for k in 0..4u32 {
        w.push(0x7E00_0200 | (13 + k) << 17 | (0x81 + k)); // v_mov_b32 v13+k, 1+k
    }
    w.extend_from_slice(desc_setup);
    w.extend([
        0xBE94_037E, // s_mov_b32 s20, exec_lo
        0x7DAA_2280, // v_cmpx_ne_u32 exec_lo, 0, v17
    ]);
    let first = 0x0000_0D01 | (24u32 >> 2) << 16;
    let dmask_bits = (dmask & 0xf) << 8;
    if wide {
        w.extend([
            0xF19C_9007 | dmask_bits,
            first,
            0x0504_0302,
            0x0908_0706,
            0x000C_0B0A,
        ]);
    } else {
        w.extend([
            0xF198_9007 | dmask_bits,
            first,
            0x0605_0403,
            0x0A09_0807,
            0x0000_0C0B,
        ]);
    }
    w.extend([
        0xBEFE_0314, // s_mov_b32 exec_lo, s20
        0x342A_0084, // v_lshlrev_b32 v21, 4, v0
        0xE078_1000,
        0x8003_0D15, // buffer_store_dwordx4 v[13:16], v21, s[12:15], 0 offen
        0xBF81_0000, // s_endpgm
    ]);
    w
}

fn build_words(words: &[u32]) -> Result<(HostShader, resource::Resources), ShaderError> {
    let nat = native(words);
    let p = ir::build(&nat.code, nat.wave64, nat.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &nat.layout, nat.stage, Some(&st))?;
    let io = io::raise(&p, &nat).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &nat,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })?;
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    Ok((shader, res))
}

#[test]
fn dynamic_bvh_descriptor_word1_tracks_and_matches_on_the_card() {
    // Root constants s0 = word0 (base >> 8), s1 = word1 (base >> 40 | optional BOX_SORT).
    let base = 0x312_3456_7800u64;
    let structure = base + u64::from(NODES);
    let w0 = (structure >> 8) as u32;
    let w1_unsorted = (structure >> 40) as u32;
    let w1_sorted = w1_unsorted | bvh::BOX_SORT;

    // 1. Dynamic word1 with static `s_or_b32 s25, s1, 0x80000000` -> sort = Some(true).
    let setup_or = [
        0xBE98_0300, // s_mov_b32 s24, s0
        0x8819_FF01,
        bvh::BOX_SORT, // s_or_b32 s25, s1, 0x80000000
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
    ];
    // 2. Dynamic word1 with static `s_and_b32 s25, s1, 0xff` -> sort = Some(false).
    let setup_and = [
        0xBE98_0300, // s_mov_b32 s24, s0
        0x8719_FF01,
        0x0000_00FF, // s_and_b32 s25, s1, 0xff
        0xBE9A_03C1,
        0xBE9B_03FF,
        0x8900_03FF,
    ];
    // 3. Fully runtime word1 (`s_mov_b32 s25, s1`) -> sort = None (checked at runtime!).
    let setup_runtime = [
        0xBE98_0300, // s_mov_b32 s24, s0
        0xBE99_0301, // s_mov_b32 s25, s1
        0xBE9A_03C1,
        0xBE9B_03FF,
        0x8900_03FF,
    ];

    let _memory = Memory::at(base);
    let (sh_or, res_or) = build_words(&custom_bvh_program(false, &setup_or, 0xf)).unwrap();
    assert_eq!(
        res_or.bvh[0].1,
        BvhNodes::Memory {
            sort: Some(true),
            zero_base: false,
            word3: 0x8900_03ff,
        }
    );
    let (sh_and, res_and) = build_words(&custom_bvh_program(false, &setup_and, 0xf)).unwrap();
    assert_eq!(
        res_and.bvh[0].1,
        BvhNodes::Memory {
            sort: Some(false),
            zero_base: false,
            word3: 0x8900_03ff,
        }
    );
    let (sh_rt, res_rt) = build_words(&custom_bvh_program(false, &setup_runtime, 0xf)).unwrap();
    assert_eq!(
        res_rt.bvh[0].1,
        BvhNodes::Memory {
            sort: None,
            zero_base: false,
            word3: 0x8900_03ff,
        }
    );

    let (scene, batches) = scene();
    let batch = &batches[0];
    let data = buffer(&scene, batch, 0);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        for (tag, shader, c0, c1, expect_sort) in [
            ("s_or_b32", &sh_or, w0, w1_unsorted, true),
            ("s_and_b32", &sh_and, w0, w1_sorted, false),
            ("runtime_sorted", &sh_rt, w0, w1_sorted, true),
            ("runtime_unsorted", &sh_rt, w0, w1_unsorted, false),
        ] {
            let (root, pso) = host.compute_pipeline(shader).expect("pso");
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![c0, c1, 0, 0, 0]))],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Raw,
                            },
                            View::RawUav(4 * THREADS),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} {tag}: {e}"));
            for t in 0..THREADS as usize {
                let want = match batch.get(t) {
                    Some(c) => expected(&scene, c, expect_sort),
                    None => BEFORE,
                };
                assert_eq!(&out[0][4 * t..4 * t + 4], want, "{label} {tag} thread {t}");
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn partial_dmask_packs_selected_channels_and_preserves_other_vgprs() {
    let setup_whole = [
        0xBE98_0380, // s_mov_b32 s24, 0
        0xBE99_0B81, // s_brev_b32 s25, 1 (BOX_SORT)
        0xBE9A_03C1, // s_mov_b32 s26, -1
        0xBE9B_03FF,
        0x8900_03FF, // s_mov_b32 s27, 0x890003ff
    ];
    // dmask == 0 must stay refused.
    assert!(build_words(&custom_bvh_program(true, &setup_whole, 0x0)).is_err());

    let _memory = Memory::at(BASE);
    let (scene, batches) = scene();
    let batch = &batches[0];
    let data = buffer(&scene, batch, BASE + u64::from(NODES));
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        for dmask in [0x3u32, 0x5u32] {
            let (shader, _) =
                build_words(&custom_bvh_program(true, &setup_whole, dmask)).expect("lowers");
            let (root, pso) = host.compute_pipeline(&shader).expect("pso");
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(vec![0; 5]))],
                        views: &[
                            View::Buffer {
                                data: &data,
                                stride: Stride::Raw,
                            },
                            View::RawUav(4 * THREADS),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} dmask={dmask:#x}: {e}"));
            let channels: Vec<usize> = (0..4).filter(|ch| (dmask >> ch) & 1 != 0).collect();
            for t in 0..THREADS as usize {
                let full = match batch.get(t) {
                    Some(c) => expected(&scene, c, true),
                    None => BEFORE,
                };
                let mut want = BEFORE;
                if batch.get(t).and_then(|c| *c).is_some() {
                    for (k, &ch) in channels.iter().enumerate() {
                        want[k] = full[ch];
                    }
                }
                assert_eq!(
                    &out[0][4 * t..4 * t + 4],
                    want,
                    "{label} dmask={dmask:#x} thread {t}"
                );
            }
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}

#[test]
fn directory_cross_page_box32_and_fault_isolation_on_the_card() {
    // Runs the real `SLOT_MEMORY` + `SLOT_FAULTS` directory path (no TEST_MEMORY!)
    // on hardware D3D12 (RTX 4070 SUPER) and verifies:
    //  1. A 128-byte `box32` node at `0x1_FFC0` straddling Page 1 (copy 1,
    //     offset 0xFFC0) and Page 2 (copy 2, separate buffer at offset 0x4000)
    //     matches `bvh::intersect` bit for bit with zero faults.
    //  2. Active lane hitting unmapped page 42 records 42 in `faults[43]`,
    //     active lane hitting unmapped page 4095 records 4095 in `faults[1]`,
    //     while page 0, out-of-range page (>= MEMORY_PAGES - 1), masked-out
    //     lanes (page 777), and kinds 6/7 (page 888) NEVER write to `SLOT_FAULTS`
    //     or erase `faults[1]`, and `faults[0]` (the old sink) stays 0.
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; skipped");
        return;
    };
    let (shader, _) = build(true, WHOLE).expect("real directory shader lowers");
    assert!(shader.root_srvs.contains(&super::SLOT_MEMORY));
    assert!(shader.root_uavs.contains(&super::SLOT_FAULTS));

    const UNIT: [f32; 6] = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
    let moved =
        |b: [f32; 6], by: [f32; 3]| -> [f32; 6] { std::array::from_fn(|k| b[k] + by[k % 3]) };
    let ids = [0x105, 0x20c, 0x304, 0x400];
    let boxes = [
        moved(UNIT, [30.0, 0.0, 0.0]), // box 0: bytes 16..40 (page 1)
        moved(UNIT, [10.0, 0.0, 0.0]), // box 1: bytes 40..64 (page 1)
        moved(UNIT, [10.0, 5.0, 0.0]), // box 2: bytes 64..88 (page 2, copy 2!)
        moved(UNIT, [20.0, 0.0, 0.0]), // box 3: bytes 88..112 (page 2, copy 2!)
    ];
    let node = nodes::box32(ids, boxes);
    let ray = Ray::new([0.0, 0.5, 0.0], [1.0, 0.0, 0.0], 1000.0);

    // Copy 1 holds the 64 KiB Page 1: records in bytes 0..2048, first half of
    // the straddling `box32` node in bytes 0xFFC0..0x10000 (`dwords[16368..16384]`).
    let mut page1_buf = vec![0u32; 0x1_0000 / 4];
    page1_buf[(0xFFC0 / 4)..(0x1_0000 / 4)].copy_from_slice(&node[..16]);

    let put_rec = |buf: &mut [u32], t: usize, addr: u64, kind: u32, active: bool| {
        let ptr = (addr >> 3) | u64::from(kind);
        let r = &mut buf[t * (RECORD / 4) as usize..][..13];
        r[0] = ptr as u32;
        r[1] = (ptr >> 32) as u32;
        r[2] = ray.extent.to_bits();
        for a in 0..3 {
            r[3 + a] = ray.origin[a].to_bits();
            r[6 + a] = ray.direction[a].to_bits();
            r[9 + a] = ray.inverse[a].to_bits();
        }
        r[12] = u32::from(active);
    };

    // Thread 0: straddling box32 at 0x1_FFC0 (Page 1 -> Page 2 across two buffers!).
    put_rec(&mut page1_buf, 0, 0x1_FFC0, 5, true);
    // Thread 1: masked-out lane pointing at unmapped page 777 (must NOT fault!).
    put_rec(&mut page1_buf, 1, 777u64 << 16, 5, false);
    // Thread 2: active kind 6 (instance) pointing at unmapped page 888 (must NOT fault!).
    put_rec(&mut page1_buf, 2, 888u64 << 16, 6, true);
    // Thread 3: active kind 7 (procedural) pointing at unmapped page 889 (must NOT fault!).
    put_rec(&mut page1_buf, 3, 889u64 << 16, 7, true);
    // Thread 4: active kind 5 pointing at unmapped page 42 (MUST fault -> faults[43] = 42).
    put_rec(&mut page1_buf, 4, 42u64 << 16, 5, true);
    // Thread 5: active kind 5 pointing at unmapped page 4095 (MUST fault -> faults[1] = 4095).
    put_rec(&mut page1_buf, 5, 4095u64 << 16, 5, true);
    // Thread 6: active kind 5 pointing at unmappable page 0 (must NOT erase faults[1]!).
    put_rec(&mut page1_buf, 6, 0, 5, true);
    // Thread 7: active kind 5 pointing at out-of-range page >= MEMORY_PAGES (must NOT fault!).
    put_rec(&mut page1_buf, 7, u64::from(super::MEMORY_PAGES) << 16, 5, true);

    // Copy 2 (separate D3D12 buffer!) holds Page 2 at both a non-zero offset
    // (`0x4000`) and offset `0` (exercising the `0u32.wrapping_sub(64)` ->
    // `+64/+80/+96` wrapping path end-to-end on the GPU).
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for page2_off in [0x4000u32, 0u32] {
        let mut page2_buf = vec![0x7fc0_deadu32; 0x8000 / 4];
        page2_buf[(page2_off as usize / 4)..(page2_off as usize / 4 + 12)]
            .copy_from_slice(&node[16..28]);

        let (out, faults, _) = dispatch_memory_timed(
            &host,
            &shader,
            &page1_buf,
            &[(1, 1, 0), (2, 2, page2_off)],
            &[&page2_buf],
            4 * THREADS,
            [1, 1, 1],
            0,
            1,
        )
        .expect("directory cross-page dispatch failed");

        // 1. Straddling box32 matches CPU reference bit-exact!
        assert_eq!(
            &out[0..4],
            bvh::intersect(&node, 5, &ray, true),
            "cross-page box32 mismatch (page2_off={page2_off:#x})"
        );
        // 2. Masked-out lane preserves BEFORE; kinds 6/7 return MISS.
        assert_eq!(&out[4..8], BEFORE);
        assert_eq!(&out[8..12], bvh::MISS.map(f32::to_bits));
        assert_eq!(&out[12..16], bvh::MISS.map(f32::to_bits));
        // 3. Unmapped box32 reads zeros from the null UAV, matching `bvh::intersect([0; 28])` (`[NONE; 4]`).
        assert_eq!(&out[16..20], bvh::intersect(&[0u32; 28], 5, &ray, true));
        assert_eq!(&out[16..20], [bvh::NONE; 4]);
        // 4. Fault table isolation: ONLY pages 42 and 4095 recorded; sink [0] stays 0!
        assert_eq!(faults[0], 0, "sink faults[0] must remain untouched");
        assert_eq!(faults[1], 4095, "page 4095 fault must not be erased by page 0");
        assert_eq!(faults[43], 42, "page 42 fault missing");
        let unexpected: Vec<(usize, u32)> = faults
            .iter()
            .enumerate()
            .filter(|&(idx, &val)| val != 0 && idx != 1 && idx != 43)
            .map(|(idx, &val)| (idx, val))
            .collect();
        assert!(
            unexpected.is_empty(),
            "spurious faults recorded: {unexpected:?}"
        );
    }
}
