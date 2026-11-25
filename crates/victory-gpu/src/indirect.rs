//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
//!
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint kind;
    uint stride;
    uint records;
    uint args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing. Bit 2: one dispatch, written as
    // SPLIT records of at most 65,535 groups in x each.
    uint has_count;
    uint count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint eff_count = records;
    bool ahead = (has_count & 2u) != 0;
    bool split = (has_count & 4u) != 0;
    if ((has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(count_offset));
    }
    if (id.x >= eff_count) return;
    uint src_at = args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 asked = src_args.Load3(src_at);
        uint3 g = min(asked, uint3(65535u, 65535u, 65535u));
        // In parts: one dimension, and not the garbage of an argument
        // nothing wrote.
        if (split && g.y <= 1u && g.z <= 1u && asked.x <= SPLIT * 65535u) {
            g.x = asked.x;
        }
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        if (split) {
            for (uint part = 0; part < SPLIT; part++) {
                uint before = part * 65535u;
                uint x = groups.x > before ? min(groups.x - before, 65535u) : 0u;
                dst_args.Store3(dst_at + part * 12u, uint3(x, groups.yz));
            }
            return;
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    let code = compile_blob(SOURCE, s!("clamp"), 0)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// `VICTORY_CLAMP_SCOPED` (on by default, `=0` disables): the wait for earlier
/// shader writes before a clamp pass is a UAV barrier on the argument buffer
/// (and the count buffer) only, not on every resource, and there is no global
/// UAV barrier after the pass (the scratch buffer's transition to
/// `INDIRECT_ARGUMENT` orders the clamp's output, and the budget word uses
/// atomics only). Buffers placed over title memory in place (`own == false`)
/// keep the global wait so aliased buffers stay ordered.
///
/// Ported from ZigZagBoi1/ScarlettPort (`runtime/victory`, GPL-3.0-or-later).
pub(crate) fn clamp_scoped() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_CLAMP_SCOPED").is_none_or(|v| v != "0");
        if !on {
            tracing::info!("indirect clamps wait on every resource (VICTORY_CLAMP_SCOPED=0)");
        }
        on
    })
}

/// The waits around a clamp pass ([`clamp_waits`]): for earlier shader writes
/// before it (global or per-buffer when `scoped`) and after it.
#[derive(Clone, Copy)]
struct Waits {
    before: bool,
    before_src: bool,
    before_count: bool,
    after: bool,
    scoped: bool,
}

impl Waits {
    /// On every resource, before and after (legacy behavior).
    #[cfg(test)]
    const ALL: Waits = Waits {
        before: true,
        before_src: true,
        before_count: true,
        after: true,
        scoped: false,
    };
}

fn clamp_waits(
    list: &HostList,
    src: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
    own: bool,
) -> Waits {
    let scoped = own && clamp_scoped();
    if !crate::host::barrier_merge() {
        return Waits {
            before: true,
            before_src: true,
            before_count: count.is_some(),
            after: !scoped,
            scoped,
        };
    }
    if scoped {
        let before_src = list.take_uav_barrier_for(src);
        let before_count = count
            .filter(|&buf| buf != src)
            .is_some_and(|buf| list.take_uav_barrier_for(buf));
        Waits {
            before: before_src || before_count,
            before_src,
            before_count,
            after: false,
            scoped: true,
        }
    } else {
        let before = list.uav_pending();
        Waits {
            before,
            before_src: before,
            before_count: before && count.is_some(),
            after: false,
            scoped: false,
        }
    }
}

fn clamp_recorded(list: &HostList, Waits { before, scoped, .. }: Waits, recorded: bool) {
    if !crate::host::barrier_merge() || !recorded {
        return;
    }
    use std::sync::atomic::Ordering::Relaxed;
    // A scoped barrier on the argument buffers leaves other UAV writes pending.
    if before && !scoped {
        list.set_uav_pending(false);
    }
    let stats = &crate::host::BARRIER_STATS;
    stats[2].fetch_add(u64::from(before), Relaxed);
    stats[3].fetch_add(1 + u64::from(!before), Relaxed);
}

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, leaving that scratch buffer in `INDIRECT_ARGUMENT` at offset `0`.
/// Replaces the list's compute root signature and pipeline (callers bind
/// theirs after). `None` if the clamp or alignment is not available.
/// `own`: `src` and `count` are dedicated buffers, not placed over title
/// memory that another buffer may alias ([`clamp_scoped`]).
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    own: bool,
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let waits = clamp_waits(list, src.0, count.map(|(b, _)| b), own);
    let recorded = record_clamp_waits(c, &list.list, &dst, budget, src, count, spec, 0, waits);
    clamp_recorded(list, waits, recorded);
    recorded.then_some(dst)
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
    own: bool,
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    let waits = clamp_waits(list, src.0, None, own);
    let recorded = record_clamp_waits(c, &list.list, &dst, budget, src, None, spec, AHEAD, waits);
    clamp_recorded(list, waits, recorded);
    recorded.then_some(dst)
}

/// The parts [`bound_split`] makes of a dispatch, the most groups of each,
/// and the bytes between them.
pub(crate) const SPLIT: u32 = 2;
pub(crate) const PART: u32 = 65_535;
pub(crate) const PART_BYTES: u64 = 12;

/// [`bound`] for one dispatch that a program with a group base runs
/// (`victory_shader::emit::USER_GROUP_BASE`): [`SPLIT`] records follow each
/// other in the scratch buffer, of which the first has the first [`PART`]
/// groups in x and the next the ones after (empty if there are none). The
/// console runs 115,200 groups in one dispatch (the title's final gather of
/// indirect light: 8 for each 16x16 tile of 2560x1440); clamped to what the
/// host runs, the tiles at the bottom of the screen kept what they showed
/// before.
pub(crate) fn bound_split(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    own: bool,
) -> Option<ID3D12Resource> {
    let spec = (2, 12, 1);
    validate_layout(spec, src.1, None)?;
    let (c, budget) = (clamp()?, list.budget()?);
    let dst = list.indirect_scratch(u64::from(SPLIT) * PART_BYTES)?;
    let waits = clamp_waits(list, src.0, None, own);
    let recorded = record_clamp_waits(c, &list.list, &dst, budget, src, None, spec, IN_PARTS, waits);
    clamp_recorded(list, waits, recorded);
    recorded.then_some(dst)
}

/// Bits of the clamp's `has_count`.
const AHEAD: u32 = 2;
const IN_PARTS: u32 = 4;

#[cfg(test)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_waits(c, l, dst, budget, src, count, spec, 0, Waits::ALL)
}

/// [`record_clamp`]; `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    record_clamp_waits(c, l, dst, budget, src, count, spec, if ahead { AHEAD } else { 0 }, Waits::ALL)
}

#[allow(clippy::too_many_arguments)]
fn record_clamp_waits(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    how: u32,
    Waits {
        before,
        before_src,
        before_count,
        after,
        scoped,
    }: Waits,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let constants = [kind, stride, records, args_offset, has_count | how, count_offset];
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `COMMON -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`.
    // Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    // Build `first` on the stack: at most 2 UAV barriers + 1 transition.
    let mut first_buf: [std::mem::MaybeUninit<D3D12_RESOURCE_BARRIER>; 3] =
        [const { std::mem::MaybeUninit::uninit() }; 3];
    let mut first_len = 0usize;
    if waits && before {
        if scoped {
            if before_src {
                first_buf[first_len].write(uav_barrier_on(src));
                first_len += 1;
            }
            if before_count
                && let Some((buf, _)) = count.filter(|(buf, _)| *buf != src)
            {
                first_buf[first_len].write(uav_barrier_on(buf));
                first_len += 1;
            }
        } else {
            first_buf[first_len].write(uav_barrier());
            first_len += 1;
        }
    }
    first_buf[first_len].write(transition(
        dst,
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    ));
    first_len += 1;
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip_second = usize::from(!waits || !after);
    unsafe {
        // SAFETY: `first_buf[..first_len]` were all initialized above, and
        // every borrowed `pResource` inside `ManuallyDrop` stays alive through
        // `ResourceBarrier` without being released.
        let first = std::slice::from_raw_parts(first_buf.as_ptr().cast::<D3D12_RESOURCE_BARRIER>(), first_len);
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(first);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&second[skip_second..]);
    }
    true
}

/// `VICTORY_INDIRECT_GATHER` (on by default, `=0` disables): batches scattered
/// small indirect argument clamps recorded into a hoisted clamp list into a
/// single `[numthreads(64, 1, 1)]` GPU compute gather-clamp pass (or a 2-pass
/// parallel gather + exact ordered budget pass for many distinct source
/// buffers), replacing per-record compute dispatches and Mode 7 UAV barriers.
pub(crate) fn gather_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_INDIRECT_GATHER").is_none_or(|v| v != "0");
        if !on {
            tracing::info!(
                "scattered indirect gather-clamp batching off (VICTORY_INDIRECT_GATHER=0)"
            );
        }
        on
    })
}

/// Maximum record count per indirect call eligible for gather-clamp batching.
/// Calls with more records already fill most of a 64-lane compute wave on
/// their own and stay on the direct dispatch path.
const GATHER_MAX_RECORDS: u32 = 32;
/// Maximum expanded records buffered in a single gather flush.
const GATHER_BATCH_MAX_ITEMS: usize = 2048;
/// Maximum distinct source/count buffers (SRVs) in a single gather flush.
const GATHER_BATCH_MAX_SRVS: usize = 512;
/// Maximum distinct destination chunk buffers (UAVs) in a single gather flush.
const GATHER_BATCH_MAX_UAVS: usize = 64;
/// Descriptor slots per per-list shader-visible gather heap.
const GATHER_HEAP_SLOTS: u32 = 2048;
/// Threshold in contiguous uniform groups below which single-pass `Dispatch(1,1,1)`
/// runs directly with zero internal UAV barriers and no `costs` scratch buffer.
const GATHER_SINGLE_PASS_MAX_GROUPS: usize = 8;

/// Single-pass `[numthreads(64, 1, 1)]` gather-clamp shader for batches whose
/// items form at most `GATHER_SINGLE_PASS_MAX_GROUPS` contiguous uniform
/// `(src_idx, count_idx, dst_idx)` groups. All descriptor table indices are
/// dynamically uniform across the threadgroup (SM 5.1 compliant), and budget
/// charging runs in strict item order via a 64-wide saturating prefix sum.
const GATHER_SINGLE_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), DescriptorTable(SRV(t0, space=1, numDescriptors=unbounded)), DescriptorTable(UAV(u0, space=1, numDescriptors=unbounded))"
cbuffer Batch : register(b0) {
    uint num_groups;
    uint total_items;
};
ByteAddressBuffer items : register(t0);
ByteAddressBuffer groups : register(t1);
RWByteAddressBuffer budget : register(u0);
ByteAddressBuffer srvs[] : register(t0, space1);
RWByteAddressBuffer uavs[] : register(u0, space1);

static const uint BUDGET = 1u << 27;
static const uint CAP = BUDGET + 1u;
static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2u;

groupshared uint s_b0;
groupshared uint s_sum[64];

uint sat_add(uint a, uint b) {
    return (a >= CAP || CAP - a <= b) ? CAP : (a + b);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    if (tid == 0u) {
        uint b0;
        budget.InterlockedOr(0, 0, b0);
        s_b0 = b0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint b0 = s_b0;
    uint rem = (b0 >= BUDGET) ? 0u : (BUDGET - b0);
    uint running = 0u;

    for (uint g = 0u; g < num_groups; g++) {
        uint4 grp = groups.Load4(g * 16u);
        uint first = grp.x;
        uint grp_count = grp.y;
        uint src_idx = grp.z;
        uint count_idx = grp.w & 0xFFFFu;
        uint dst_idx = grp.w >> 16u;

        bool active = tid < grp_count;
        uint kind = 0u;
        uint stride = 0u;
        uint rec_idx = 0u;
        uint flags = 0u;
        uint src_at = 0u;
        uint count_at = 0u;
        uint dst_at = 0u;
        uint cost = 0u;
        uint2 draw_counts = uint2(0u, 0u);
        uint3 dgroups = uint3(0u, 0u, 0u);

        if (active) {
            uint item_off = (first + tid) * 32u;
            uint4 w0 = items.Load4(item_off);
            uint4 w1 = items.Load4(item_off + 16u);
            kind = w0.x;
            stride = w0.y;
            rec_idx = w0.z;
            flags = w0.w;
            src_at = w1.x;
            count_at = w1.y;
            dst_at = w1.z;
            if ((flags & 1u) != 0u) {
                uint cnt = srvs[count_idx].Load(count_at);
                if (rec_idx >= cnt) {
                    active = false;
                }
            }
        }
        if (active) {
            bool ahead = (flags & 2u) != 0u;
            bool split = (flags & 4u) != 0u;
            if (kind <= 1u) {
                uint2 asked = srvs[src_idx].Load2(src_at);
                if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
                    asked = uint2(0u, 0u);
                }
                uint count = min(asked.x, 1u << 24);
                uint instances = min(asked.y, (1u << 26) / max(count, 1u));
                cost = count * instances;
                draw_counts = uint2(count, instances);
            } else if (kind == 2u) {
                uint3 asked = srvs[src_idx].Load3(src_at);
                uint3 dg = min(asked, uint3(65535u, 65535u, 65535u));
                if (split && dg.y <= 1u && dg.z <= 1u && asked.x <= SPLIT * 65535u) {
                    dg.x = asked.x;
                }
                if (dg.z != 0u && dg.x * dg.y > GROUPS / dg.z) {
                    dg = uint3(0u, 0u, 0u);
                }
                cost = dg.x * dg.y * dg.z * 64u;
                dgroups = dg;
            } else {
                uint3 dg = srvs[src_idx].Load3(src_at);
                if (ahead && dg.x > 65535u) {
                    dg = uint3(0u, 0u, 0u);
                }
                uint was;
                budget.InterlockedMax(4, dg.x, was);
                budget.InterlockedAdd(8, min(dg.x, 1u << 24), was);
                dg = uint3(min(dg.x, 65535u), min(dg.y, 1u), min(dg.z, 1u));
                cost = dg.x * dg.y * dg.z * 128u;
                dgroups = dg;
            }
        }

        s_sum[tid] = cost;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint step = 1u; step < 64u; step <<= 1u) {
            uint v = s_sum[tid];
            if (tid >= step) {
                v = sat_add(v, s_sum[tid - step]);
            }
            GroupMemoryBarrierWithGroupSync();
            s_sum[tid] = v;
            GroupMemoryBarrierWithGroupSync();
        }
        uint inc = sat_add(running, s_sum[tid]);
        running = sat_add(running, s_sum[63]);
        GroupMemoryBarrierWithGroupSync();
        bool over = (cost > 0u) && (inc > rem);

        if (active) {
            bool split = (flags & 4u) != 0u;
            if (kind <= 1u) {
                if (over) {
                    draw_counts = uint2(0u, 0u);
                }
                uavs[dst_idx].Store2(dst_at, draw_counts);
                for (uint off = 8u; off < stride; off += 4u) {
                    uavs[dst_idx].Store(dst_at + off, srvs[src_idx].Load(src_at + off));
                }
            } else {
                if (over) {
                    dgroups = uint3(0u, 0u, 0u);
                }
                if (split) {
                    for (uint part = 0u; part < SPLIT; part++) {
                        uint before_g = part * 65535u;
                        uint x = dgroups.x > before_g ? min(dgroups.x - before_g, 65535u) : 0u;
                        uavs[dst_idx].Store3(dst_at + part * 12u, uint3(x, dgroups.yz));
                    }
                } else {
                    uavs[dst_idx].Store3(dst_at, dgroups);
                    for (uint off = 12u; off < stride; off += 4u) {
                        uavs[dst_idx].Store(dst_at + off, srvs[src_idx].Load(src_at + off));
                    }
                }
            }
        }
    }

    if (tid == 0u && b0 < BUDGET && running > 0u) {
        uint next = (running >= rem) ? BUDGET : (b0 + running);
        uint prev;
        budget.InterlockedMax(0, next, prev);
    }
}
"#;

/// Pass 1 of two-pass parallel gather-clamp: runs one 64-lane threadgroup per
/// uniform `(src_idx, count_idx, dst_idx)` group across all SMs, clamping and
/// writing arguments in full parallel and storing each record's `cost` at its
/// original sequence slot `costs[seq]`.
const GATHER_PASS1_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), DescriptorTable(SRV(t0, space=1, numDescriptors=unbounded)), DescriptorTable(UAV(u0, space=1, numDescriptors=unbounded))"
cbuffer Batch : register(b0) {
    uint num_groups;
    uint total_items;
};
ByteAddressBuffer items : register(t0);
ByteAddressBuffer groups : register(t1);
RWByteAddressBuffer budget : register(u0);
RWByteAddressBuffer costs : register(u1);
ByteAddressBuffer srvs[] : register(t0, space1);
RWByteAddressBuffer uavs[] : register(u0, space1);

static const uint GROUPS = 1u << 20;
static const uint SPLIT = 2u;

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint g = gid.x;
    if (g >= num_groups) return;
    uint4 grp = groups.Load4(g * 16u);
    uint first = grp.x;
    uint grp_count = grp.y;
    uint tid = gtid.x;
    if (tid >= grp_count) return;

    uint src_idx = grp.z;
    uint count_idx = grp.w & 0xFFFFu;
    uint dst_idx = grp.w >> 16u;

    uint item_off = (first + tid) * 32u;
    uint4 w0 = items.Load4(item_off);
    uint4 w1 = items.Load4(item_off + 16u);
    uint kind = w0.x;
    uint stride = w0.y;
    uint rec_idx = w0.z;
    uint flags = w0.w;
    uint src_at = w1.x;
    uint count_at = w1.y;
    uint dst_at = w1.z;
    uint seq = w1.w & 0x00FFFFFFu;

    if ((flags & 1u) != 0u) {
        uint cnt = srvs[count_idx].Load(count_at);
        if (rec_idx >= cnt) {
            costs.Store(seq * 4u, 0u);
            return;
        }
    }

    bool ahead = (flags & 2u) != 0u;
    bool split = (flags & 4u) != 0u;
    uint cost = 0u;

    if (kind <= 1u) {
        uint2 asked = srvs[src_idx].Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0u, 0u);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        uavs[dst_idx].Store2(dst_at, uint2(count, instances));
        for (uint off = 8u; off < stride; off += 4u) {
            uavs[dst_idx].Store(dst_at + off, srvs[src_idx].Load(src_at + off));
        }
    } else if (kind == 2u) {
        uint3 asked = srvs[src_idx].Load3(src_at);
        uint3 dg = min(asked, uint3(65535u, 65535u, 65535u));
        if (split && dg.y <= 1u && dg.z <= 1u && asked.x <= SPLIT * 65535u) {
            dg.x = asked.x;
        }
        if (dg.z != 0u && dg.x * dg.y > GROUPS / dg.z) {
            dg = uint3(0u, 0u, 0u);
        }
        cost = dg.x * dg.y * dg.z * 64u;
        if (split) {
            for (uint part = 0u; part < SPLIT; part++) {
                uint before_g = part * 65535u;
                uint x = dg.x > before_g ? min(dg.x - before_g, 65535u) : 0u;
                uavs[dst_idx].Store3(dst_at + part * 12u, uint3(x, dg.yz));
            }
        } else {
            uavs[dst_idx].Store3(dst_at, dg);
            for (uint off = 12u; off < stride; off += 4u) {
                uavs[dst_idx].Store(dst_at + off, srvs[src_idx].Load(src_at + off));
            }
        }
    } else {
        uint3 dg = srvs[src_idx].Load3(src_at);
        if (ahead && dg.x > 65535u) {
            dg = uint3(0u, 0u, 0u);
        }
        uint was;
        budget.InterlockedMax(4, dg.x, was);
        budget.InterlockedAdd(8, min(dg.x, 1u << 24), was);
        dg = uint3(min(dg.x, 65535u), min(dg.y, 1u), min(dg.z, 1u));
        cost = dg.x * dg.y * dg.z * 128u;
        uavs[dst_idx].Store3(dst_at, dg);
        for (uint off = 12u; off < stride; off += 4u) {
            uavs[dst_idx].Store(dst_at + off, srvs[src_idx].Load(src_at + off));
        }
    }
    costs.Store(seq * 4u, cost);
}
"#;

/// Pass 2 of two-pass gather-clamp: computes the exact original-order saturating
/// prefix sum over `costs[0..total_items]` in 64-wide chunks, updates `budget[0]`,
/// and (only if budget overflowed) zeroes the exact over-budget records in `uavs[]`
/// using wave-uniform `dst_idx` from `groups[]`. Shares the root signature with
/// `GATHER_PASS1_CS`.
const GATHER_PASS2_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1), DescriptorTable(SRV(t0, space=1, numDescriptors=unbounded)), DescriptorTable(UAV(u0, space=1, numDescriptors=unbounded))"
cbuffer Batch : register(b0) {
    uint num_groups;
    uint total_items;
};
ByteAddressBuffer items : register(t0);
ByteAddressBuffer groups : register(t1);
RWByteAddressBuffer budget : register(u0);
RWByteAddressBuffer costs : register(u1);
RWByteAddressBuffer uavs[] : register(u0, space1);

static const uint BUDGET = 1u << 27;
static const uint CAP = BUDGET + 1u;
static const uint SPLIT = 2u;

groupshared uint s_b0;
groupshared uint s_sum[64];

uint sat_add(uint a, uint b) {
    return (a >= CAP || CAP - a <= b) ? CAP : (a + b);
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    if (tid == 0u) {
        uint b0;
        budget.InterlockedOr(0, 0, b0);
        s_b0 = b0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint b0 = s_b0;
    uint rem = (b0 >= BUDGET) ? 0u : (BUDGET - b0);
    uint running = 0u;

    for (uint base = 0u; base < total_items; base += 64u) {
        uint seq = base + tid;
        uint c = (seq < total_items) ? costs.Load(seq * 4u) : 0u;
        s_sum[tid] = c;
        GroupMemoryBarrierWithGroupSync();
        [unroll]
        for (uint step = 1u; step < 64u; step <<= 1u) {
            uint v = s_sum[tid];
            if (tid >= step) {
                v = sat_add(v, s_sum[tid - step]);
            }
            GroupMemoryBarrierWithGroupSync();
            s_sum[tid] = v;
            GroupMemoryBarrierWithGroupSync();
        }
        uint inc = sat_add(running, s_sum[tid]);
        running = sat_add(running, s_sum[63]);
        GroupMemoryBarrierWithGroupSync();
        bool over = (seq < total_items) && (c > 0u) && (inc > rem);
        if (seq < total_items) {
            costs.Store(seq * 4u, over ? 1u : 0u);
        }
    }

    if (tid == 0u && b0 < BUDGET && running > 0u) {
        uint next = (running >= rem) ? BUDGET : (b0 + running);
        uint prev;
        budget.InterlockedMax(0, next, prev);
    }

    if (running <= rem) {
        return;
    }

    DeviceMemoryBarrierWithGroupSync();
    for (uint g = 0u; g < num_groups; g++) {
        uint4 grp = groups.Load4(g * 16u);
        uint first = grp.x;
        uint grp_count = grp.y;
        uint dst_idx = grp.w >> 16u;
        if (tid < grp_count) {
            uint item_off = (first + tid) * 32u;
            uint4 w0 = items.Load4(item_off);
            uint4 w1 = items.Load4(item_off + 16u);
            uint seq = w1.w & 0x00FFFFFFu;
            if (costs.Load(seq * 4u) != 0u) {
                uint kind = w0.x;
                uint flags = w0.w;
                uint dst_at = w1.z;
                if (kind <= 1u) {
                    uavs[dst_idx].Store2(dst_at, uint2(0u, 0u));
                } else if ((flags & 4u) != 0u) {
                    for (uint part = 0u; part < SPLIT; part++) {
                        uavs[dst_idx].Store3(dst_at + part * 12u, uint3(0u, 0u, 0u));
                    }
                } else {
                    uavs[dst_idx].Store3(dst_at, uint3(0u, 0u, 0u));
                }
            }
        }
    }
}
"#;

struct GatherClamp {
    single_root: ID3D12RootSignature,
    single_pso: ID3D12PipelineState,
    twopass_root: ID3D12RootSignature,
    pass1_pso: ID3D12PipelineState,
    pass2_pso: ID3D12PipelineState,
}

// SAFETY: D3D12 root signatures and pipeline states are free-threaded.
unsafe impl Send for GatherClamp {}
// SAFETY: as above.
unsafe impl Sync for GatherClamp {}

fn compile_blob(
    source: &str,
    name: windows::core::PCSTR,
    flags: u32,
) -> windows::core::Result<ID3DBlob> {
    // Serialize `D3DCompile` calls across threads: `d3dcompiler_47.dll` can
    // fault when many threads invoke FXC simultaneously.
    static COMPILE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = COMPILE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: source and entry strings live through `D3DCompile`.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            flags,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the error blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            let msg = String::from_utf8_lossy(text);
            tracing::warn!("gather clamp shader compile: {msg}");
            #[cfg(test)]
            eprintln!("gather clamp shader compile error:\n{msg}");
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn make_gather_for(device: &ID3D12Device) -> windows::core::Result<GatherClamp> {
    // Unbounded UAV descriptor tables require Resource Binding Tier 3.
    let mut opts = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    // SAFETY: plain feature query on a live device.
    unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            (&raw mut opts).cast(),
            size_of_val(&opts) as u32,
        )?;
    }
    if opts.ResourceBindingTier.0 < D3D12_RESOURCE_BINDING_TIER_3.0 {
        return Err(windows::core::Error::from_hresult(
            windows::Win32::Foundation::E_NOTIMPL,
        ));
    }
    let unbounded =
        windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES;
    let single_blob = compile_blob(GATHER_SINGLE_CS, s!("gather_single"), unbounded)?;
    let pass1_blob = compile_blob(GATHER_PASS1_CS, s!("gather_pass1"), unbounded)?;
    let pass2_blob = compile_blob(GATHER_PASS2_CS, s!("gather_pass2"), unbounded)?;
    // SAFETY: compiled DXBC blobs with embedded root signatures.
    unsafe {
        let slice = |blob: &ID3DBlob| {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        let single_bytes = slice(&single_blob);
        let pass1_bytes = slice(&pass1_blob);
        let pass2_bytes = slice(&pass2_blob);
        let single_root: ID3D12RootSignature = device.CreateRootSignature(0, single_bytes)?;
        let single_pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&single_root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: single_bytes.as_ptr().cast(),
                BytecodeLength: single_bytes.len(),
            },
            ..Default::default()
        })?;
        let twopass_root: ID3D12RootSignature = device.CreateRootSignature(0, pass1_bytes)?;
        let pass1_pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&twopass_root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: pass1_bytes.as_ptr().cast(),
                BytecodeLength: pass1_bytes.len(),
            },
            ..Default::default()
        })?;
        let pass2_pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&twopass_root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: pass2_bytes.as_ptr().cast(),
                BytecodeLength: pass2_bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(GatherClamp {
            single_root,
            single_pso,
            twopass_root,
            pass1_pso,
            pass2_pso,
        })
    }
}

fn gather_clamp() -> Option<&'static GatherClamp> {
    static GATHER: std::sync::OnceLock<Option<GatherClamp>> = std::sync::OnceLock::new();
    GATHER
        .get_or_init(|| {
            let device = &host()?.device;
            match make_gather_for(device) {
                Ok(g) => Some(g),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "gather-clamp shader unavailable; falling back to per-record clamps"
                    );
                    None
                }
            }
        })
        .as_ref()
}

/// One pending expanded record inside [`GatherState`].
struct PendingItem {
    kind: u32,
    stride: u32,
    rec_idx: u32,
    flags: u32,
    src_offset: u32,
    count_offset: u32,
    dst_offset: u32,
    src_idx: u16,
    count_idx: u16,
    dst_idx: u16,
    seq: u32,
}

/// Stats returned by [`GatherState::flush_on`] for diagnostics and benchmarks.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GatherFlushStats {
    pub(crate) items: u32,
    pub(crate) groups: u32,
    pub(crate) dispatches: u32,
    pub(crate) barriers: u32,
    pub(crate) two_pass: bool,
}

/// Per-`HostList` gather-clamp batching state: accumulates scattered small
/// indirect clamps and flushes them in a single `[numthreads(64, 1, 1)]`
/// compute pass (or 2-pass parallel gather + ordered prefix sum), keeping all
/// referenced descriptors and resources alive until `HostList::reset`.
#[derive(Default)]
pub(crate) struct GatherState {
    pending: Vec<PendingItem>,
    srvs: Vec<(ID3D12Resource, u32)>,
    srv_index: std::collections::HashMap<usize, u16>,
    uavs: Vec<(ID3D12Resource, u32)>,
    uav_index: std::collections::HashMap<usize, u16>,
    /// Shader-visible CBV/SRV/UAV descriptor heap reused across list resets.
    heap: Option<(ID3D12DescriptorHeap, u32)>,
    heap_used: u32,
    /// Earlier heaps replaced mid-list when a single list overflows `GATHER_HEAP_SLOTS`,
    /// kept alive until GPU completion (`reset`).
    retired_heaps: Vec<ID3D12DescriptorHeap>,
    /// Source/count/destination resources referenced by flushed batches in this
    /// list recording, kept alive until `reset` prevents descriptor UAF.
    kept_resources: Vec<ID3D12Resource>,
    /// Device-local `DEFAULT` UAV buffer for two-pass per-record `costs`, reused
    /// across resets.
    costs: Option<(ID3D12Resource, u32)>,
    /// Whether `costs` was already written/read in this recording (needs a UAV
    /// barrier before reuse in a second two-pass flush on the same list).
    costs_used: bool,
}

// SAFETY: D3D12 resources and descriptor heaps are free-threaded; `GatherState`
// is only accessed under its `HostList`'s mutex.
unsafe impl Send for GatherState {}

impl GatherState {
    pub(crate) fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Clears per-recording batch state after the GPU has finished executing
    /// this list, keeping the allocated descriptor heap and `costs` buffer for
    /// reuse on the next recording.
    pub(crate) fn reset(&mut self) {
        self.pending.clear();
        self.srvs.clear();
        self.srv_index.clear();
        self.uavs.clear();
        self.uav_index.clear();
        self.heap_used = 0;
        self.retired_heaps.clear();
        self.kept_resources.clear();
        self.costs_used = false;
    }

    fn raw_num_elements(res: &ID3D12Resource, required_end: u64) -> Option<u32> {
        // `ByteAddressBuffer` addresses bytes via 32-bit offsets (max 2^30 - 1
        // 32-bit raw words so `NumElements * 4` stays strictly within `u32`).
        const MAX_RAW_ELEMS: u64 = (1u64 << 30) - 1;
        const MAX_RAW_BYTES: u64 = MAX_RAW_ELEMS * 4;
        // SAFETY: live D3D12 buffer resource.
        let width = unsafe { res.GetDesc().Width };
        if width < 4 || required_end > width || required_end > MAX_RAW_BYTES {
            return None;
        }
        let elems = (width / 4).min(MAX_RAW_ELEMS);
        Some(elems as u32)
    }

    fn would_overflow(&self, records: u32, src: &ID3D12Resource, count: Option<&ID3D12Resource>, dst: &ID3D12Resource) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        if self.pending.len() + records as usize > GATHER_BATCH_MAX_ITEMS {
            return true;
        }
        use windows::core::Interface;
        let mut extra_srvs = usize::from(!self.srv_index.contains_key(&(src.as_raw() as usize)));
        if let Some(c) = count
            && c.as_raw() != src.as_raw()
            && !self.srv_index.contains_key(&(c.as_raw() as usize))
        {
            extra_srvs += 1;
        }
        if self.srvs.len() + extra_srvs > GATHER_BATCH_MAX_SRVS {
            return true;
        }
        let extra_uavs = usize::from(!self.uav_index.contains_key(&(dst.as_raw() as usize)));
        self.uavs.len() + extra_uavs > GATHER_BATCH_MAX_UAVS
    }

    fn register_srv(&mut self, res: &ID3D12Resource, num_elems: u32) -> u16 {
        use windows::core::Interface;
        let key = res.as_raw() as usize;
        if let Some(&idx) = self.srv_index.get(&key) {
            return idx;
        }
        let idx = self.srvs.len() as u16;
        self.srvs.push((res.clone(), num_elems));
        self.srv_index.insert(key, idx);
        idx
    }

    fn register_uav(&mut self, res: &ID3D12Resource, num_elems: u32) -> u16 {
        use windows::core::Interface;
        let key = res.as_raw() as usize;
        if let Some(&idx) = self.uav_index.get(&key) {
            return idx;
        }
        let idx = self.uavs.len() as u16;
        self.uavs.push((res.clone(), num_elems));
        self.uav_index.insert(key, idx);
        idx
    }

    /// Enqueues a validated indirect clamp call whose output slot `(dst, dst_offset)`
    /// has already been reserved in `UNORDERED_ACCESS`. Returns `false` if the
    /// offsets or resource sizes do not fit raw 32-bit gather addressing.
    #[allow(clippy::too_many_arguments)]
    fn enqueue(
        &mut self,
        (src, src_offset): (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
        (dst, dst_offset): (&ID3D12Resource, u64),
        (kind, stride, records): (u32, u32, u32),
        how: u32,
    ) -> bool {
        let Some(bytes) = validate_layout((kind, stride, records), src_offset, count.map(|(_, o)| o)) else {
            return false;
        };
        if records == 0 || records > GATHER_MAX_RECORDS {
            return false;
        }
        let dst_bytes = if (how & IN_PARTS) != 0 {
            u64::from(SPLIT) * PART_BYTES
        } else {
            bytes
        };
        let Some(src_end) = src_offset.checked_add(bytes).filter(|&e| e <= u64::from(u32::MAX)) else {
            return false;
        };
        let Some(dst_end) = dst_offset.checked_add(dst_bytes).filter(|&e| e <= u64::from(u32::MAX)) else {
            return false;
        };
        let Some(src_elems) = Self::raw_num_elements(src, src_end) else {
            return false;
        };
        let Some(dst_elems) = Self::raw_num_elements(dst, dst_end) else {
            return false;
        };
        let count_info = match count {
            Some((buf, off)) => {
                let Some(end) = off.checked_add(4).filter(|&e| e <= u64::from(u32::MAX)) else {
                    return false;
                };
                let Some(elems) = Self::raw_num_elements(buf, end) else {
                    return false;
                };
                Some((buf, off as u32, elems))
            }
            None => None,
        };

        let src_idx = self.register_srv(src, src_elems);
        let (count_idx, count_offset, has_count) = match count_info {
            Some((buf, off, elems)) => (self.register_srv(buf, elems), off, 1u32),
            None => (src_idx, 0u32, 0u32),
        };
        let dst_idx = self.register_uav(dst, dst_elems);
        let flags = has_count | how;

        for rec_idx in 0..records {
            let seq = self.pending.len() as u32;
            let rec_off = rec_idx * stride;
            self.pending.push(PendingItem {
                kind,
                stride,
                rec_idx,
                flags,
                src_offset: src_offset as u32 + rec_off,
                count_offset,
                dst_offset: dst_offset as u32 + rec_off,
                src_idx,
                count_idx,
                dst_idx,
                seq,
            });
        }
        true
    }

    fn ensure_heap(&mut self, device: &ID3D12Device, needed: u32) -> Option<(ID3D12DescriptorHeap, u32)> {
        let cap = self.heap.as_ref().map_or(0, |(_, c)| *c);
        if self.heap.is_none() || self.heap_used.saturating_add(needed) > cap {
            let new_cap = GATHER_HEAP_SLOTS.max(needed);
            // SAFETY: valid shader-visible CBV/SRV/UAV descriptor heap creation.
            let new_heap: ID3D12DescriptorHeap = unsafe {
                device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: new_cap,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .ok()?
            };
            if let Some((old, _)) = self.heap.replace((new_heap, new_cap)) {
                self.retired_heaps.push(old);
            }
            self.heap_used = 0;
        }
        let base = self.heap_used;
        self.heap_used += needed;
        Some((self.heap.as_ref()?.0.clone(), base))
    }

    fn ensure_costs(&mut self, device: &ID3D12Device, items: u32) -> Option<ID3D12Resource> {
        let needed_bytes = u64::from(items).max(1) * 4;
        if let Some((buf, cap)) = &self.costs
            && *cap >= items
        {
            return Some(buf.clone());
        }
        let alloc_items = items.max(2048).next_power_of_two();
        let size = (u64::from(alloc_items) * 4).max(needed_bytes).next_multiple_of(256);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain device-local UAV buffer creation.
        unsafe {
            device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut out,
                )
                .ok()?;
        }
        let buf = out?;
        if let Some((old, _)) = self.costs.replace((buf.clone(), alloc_items)) {
            self.kept_resources.push(old);
        }
        self.costs_used = false;
        Some(buf)
    }

    /// Scalar fallback for [`Self::flush_on`] when gather scratch/heap/costs
    /// allocation fails: reconstructs each original call from consecutive
    /// `rec_idx = 0..records` runs in `pending` and records scalar `SOURCE`
    /// compute dispatches using root SRVs/UAVs/constants only (zero extra
    /// allocations). Keeps all referenced resources alive until `reset`.
    #[allow(clippy::too_many_arguments)]
    fn record_pending_scalar(
        &mut self,
        c: &Clamp,
        cmd_list: &ID3D12GraphicsCommandList,
        pending: &[PendingItem],
        srvs: Vec<(ID3D12Resource, u32)>,
        uavs: Vec<(ID3D12Resource, u32)>,
        budget_gpu: u64,
        budget_res: &ID3D12Resource,
        budget_barrier_after: bool,
    ) -> GatherFlushStats {
        let mut dispatches = 0u32;
        let mut barriers = 0u32;
        let mut i = 0usize;
        while i < pending.len() {
            let first = &pending[i];
            let mut run = 1usize;
            while i + run < pending.len() {
                let next = &pending[i + run];
                let step = (run as u32).saturating_mul(first.stride);
                if next.rec_idx == run as u32
                    && next.kind == first.kind
                    && next.stride == first.stride
                    && next.flags == first.flags
                    && next.src_idx == first.src_idx
                    && next.count_idx == first.count_idx
                    && next.count_offset == first.count_offset
                    && next.dst_idx == first.dst_idx
                    && next.src_offset == first.src_offset.wrapping_add(step)
                    && next.dst_offset == first.dst_offset.wrapping_add(step)
                {
                    run += 1;
                } else {
                    break;
                }
            }
            let records = run as u32;
            let src_res = &srvs[first.src_idx as usize].0;
            let count_res = &srvs[first.count_idx as usize].0;
            let dst_res = &uavs[first.dst_idx as usize].0;
            // SAFETY: open command list, live buffers/PSOs, valid offsets validated at enqueue.
            unsafe {
                let src_gpu = src_res.GetGPUVirtualAddress();
                let count_gpu = count_res.GetGPUVirtualAddress();
                let dst_gpu = dst_res.GetGPUVirtualAddress() + u64::from(first.dst_offset);
                let constants = [
                    first.kind,
                    first.stride,
                    records,
                    first.src_offset,
                    first.flags,
                    first.count_offset,
                ];
                cmd_list.SetComputeRootSignature(&c.root);
                cmd_list.SetPipelineState(&c.pso);
                cmd_list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                cmd_list.SetComputeRootShaderResourceView(1, src_gpu);
                cmd_list.SetComputeRootShaderResourceView(2, count_gpu);
                cmd_list.SetComputeRootUnorderedAccessView(3, dst_gpu);
                cmd_list.SetComputeRootUnorderedAccessView(4, budget_gpu);
                cmd_list.Dispatch(records.div_ceil(64), 1, 1);
                dispatches += 1;
                if i + run < pending.len() || budget_barrier_after {
                    cmd_list.ResourceBarrier(&[uav_barrier_on(budget_res)]);
                    barriers += 1;
                }
            }
            i += run;
        }
        self.kept_resources.extend(srvs.into_iter().map(|(r, _)| r));
        self.kept_resources.extend(uavs.into_iter().map(|(r, _)| r));
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::IndirectBounded, u64::from(dispatches));
        GatherFlushStats {
            items: pending.len() as u32,
            groups: 0,
            dispatches,
            barriers,
            two_pass: false,
        }
    }

    /// Flushes all pending gathered clamps onto `cmd_list`, allocating metadata
    /// upload memory via `alloc_scratch` (must remain alive until GPU completion).
    /// If upload scratch, descriptor heap, or `costs` allocation fails, falls
    /// back to recording scalar `fallback` clamps so no queued `ExecuteIndirect`
    /// argument slot is ever left uninitialized.
    #[allow(clippy::too_many_arguments)]
    fn flush_on(
        &mut self,
        device: &ID3D12Device,
        g: &GatherClamp,
        fallback: Option<&Clamp>,
        cmd_list: &ID3D12GraphicsCommandList,
        budget_gpu: u64,
        budget_res: &ID3D12Resource,
        budget_barrier_after: bool,
        alloc_scratch: impl FnOnce(u64) -> Option<(u64, *mut u8)>,
    ) -> Option<GatherFlushStats> {
        if self.pending.is_empty() {
            return Some(GatherFlushStats::default());
        }
        let pending = std::mem::take(&mut self.pending);
        let srvs = std::mem::take(&mut self.srvs);
        self.srv_index.clear();
        let uavs = std::mem::take(&mut self.uavs);
        self.uav_index.clear();

        let total_items = pending.len() as u32;
        let num_srvs = srvs.len() as u32;
        let num_uavs = uavs.len() as u32;

        // First check whether contiguous runs in original `seq` order fit within
        // `GATHER_SINGLE_PASS_MAX_GROUPS` uniform groups of up to 64 items.
        let mut seq_groups: Vec<(u32, u32, u16, u16, u16)> = Vec::new();
        for (i, item) in pending.iter().enumerate() {
            let key = (item.src_idx, item.count_idx, item.dst_idx);
            match seq_groups.last_mut() {
                Some(last) if (last.2, last.3, last.4) == key && last.1 < 64 => {
                    last.1 += 1;
                }
                _ => {
                    seq_groups.push((i as u32, 1, key.0, key.1, key.2));
                }
            }
        }

        let single_pass = seq_groups.len() <= GATHER_SINGLE_PASS_MAX_GROUPS;
        let (ordered_items, groups_raw): (Vec<[u32; 8]>, Vec<[u32; 4]>) = if single_pass {
            let items_raw = pending
                .iter()
                .map(|it| {
                    [
                        it.kind,
                        it.stride,
                        it.rec_idx,
                        it.flags,
                        it.src_offset,
                        it.count_offset,
                        it.dst_offset,
                        (it.seq & 0x00FF_FFFF) | (u32::from(it.dst_idx) << 24),
                    ]
                })
                .collect();
            let groups_raw = seq_groups
                .iter()
                .map(|&(first, count, s, c, d)| {
                    [first, count, u32::from(s), u32::from(c) | (u32::from(d) << 16)]
                })
                .collect();
            (items_raw, groups_raw)
        } else {
            // Bucket items by `(src_idx, count_idx, dst_idx)` so each 64-lane
            // threadgroup packs as many items sharing the same uniform resources
            // as possible. Original `seq` is preserved in `w1.w` for Pass 2.
            let mut order: Vec<usize> = (0..pending.len()).collect();
            order.sort_by_key(|&i| {
                let it = &pending[i];
                (it.src_idx, it.count_idx, it.dst_idx, it.seq)
            });
            let mut items_raw = Vec::with_capacity(pending.len());
            let mut groups_raw: Vec<[u32; 4]> = Vec::new();
            let mut last_key: Option<(u16, u16, u16)> = None;
            for &idx in &order {
                let it = &pending[idx];
                let key = (it.src_idx, it.count_idx, it.dst_idx);
                let out_pos = items_raw.len() as u32;
                items_raw.push([
                    it.kind,
                    it.stride,
                    it.rec_idx,
                    it.flags,
                    it.src_offset,
                    it.count_offset,
                    it.dst_offset,
                    (it.seq & 0x00FF_FFFF) | (u32::from(it.dst_idx) << 24),
                ]);
                match groups_raw.last_mut() {
                    Some(last) if last_key == Some(key) && last[1] < 64 => {
                        last[1] += 1;
                    }
                    _ => {
                        groups_raw.push([
                            out_pos,
                            1,
                            u32::from(key.0),
                            u32::from(key.1) | (u32::from(key.2) << 16),
                        ]);
                        last_key = Some(key);
                    }
                }
            }
            (items_raw, groups_raw)
        };

        let num_groups = groups_raw.len() as u32;
        let items_bytes = (ordered_items.len() * 32) as u64;
        let groups_offset = items_bytes.next_multiple_of(16);
        let groups_bytes = (groups_raw.len() * 16) as u64;
        let total_scratch = groups_offset + groups_bytes;
        let Some((scratch_gpu, scratch_cpu)) = alloc_scratch(total_scratch) else {
            return fallback.map(|c| {
                self.record_pending_scalar(
                    c,
                    cmd_list,
                    &pending,
                    srvs,
                    uavs,
                    budget_gpu,
                    budget_res,
                    budget_barrier_after,
                )
            });
        };
        let Some((heap, base)) = self.ensure_heap(device, num_srvs + num_uavs) else {
            return fallback.map(|c| {
                self.record_pending_scalar(
                    c,
                    cmd_list,
                    &pending,
                    srvs,
                    uavs,
                    budget_gpu,
                    budget_res,
                    budget_barrier_after,
                )
            });
        };
        let costs_res = if single_pass {
            None
        } else {
            let Some(buf) = self.ensure_costs(device, total_items) else {
                return fallback.map(|c| {
                    self.record_pending_scalar(
                        c,
                        cmd_list,
                        &pending,
                        srvs,
                        uavs,
                        budget_gpu,
                        budget_res,
                        budget_barrier_after,
                    )
                });
            };
            Some(buf)
        };

        // SAFETY: `scratch_cpu` points to `total_scratch` writable mapped bytes
        // in upload memory that outlive the command list's GPU execution.
        unsafe {
            std::ptr::copy_nonoverlapping(
                ordered_items.as_ptr().cast::<u8>(),
                scratch_cpu,
                items_bytes as usize,
            );
            std::ptr::copy_nonoverlapping(
                groups_raw.as_ptr().cast::<u8>(),
                scratch_cpu.add(groups_offset as usize),
                groups_bytes as usize,
            );
        }

        // SAFETY: live device, shader-visible descriptor heap, valid buffer resources.
        let (srv_gpu_handle, uav_gpu_handle) = unsafe {
            let inc = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let cpu_start = heap.GetCPUDescriptorHandleForHeapStart();
            let gpu_start = heap.GetGPUDescriptorHandleForHeapStart();
            for (i, (res, num_elems)) in srvs.iter().enumerate() {
                let mut h = cpu_start;
                h.ptr += (base as usize + i) * inc;
                let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: *num_elems,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                };
                device.CreateShaderResourceView(res, Some(&desc), h);
            }
            for (i, (res, num_elems)) in uavs.iter().enumerate() {
                let mut h = cpu_start;
                h.ptr += (base as usize + num_srvs as usize + i) * inc;
                let desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                    Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                    Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_UAV {
                            FirstElement: 0,
                            NumElements: *num_elems,
                            StructureByteStride: 0,
                            CounterOffsetInBytes: 0,
                            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
                        },
                    },
                };
                device.CreateUnorderedAccessView(res, None, Some(&desc), h);
            }
            let srv_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + (base as u64) * (inc as u64),
            };
            let uav_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu_start.ptr + (base as u64 + u64::from(num_srvs)) * (inc as u64),
            };
            (srv_gpu, uav_gpu)
        };

        let constants = [num_groups, total_items];
        let mut dispatches = 0u32;
        let mut barriers = 0u32;

        if single_pass {
            // SAFETY: open command list, live root signature/PSO/heap/buffers.
            unsafe {
                cmd_list.SetDescriptorHeaps(&[Some(heap)]);
                cmd_list.SetComputeRootSignature(&g.single_root);
                cmd_list.SetPipelineState(&g.single_pso);
                cmd_list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                cmd_list.SetComputeRootShaderResourceView(1, scratch_gpu);
                cmd_list.SetComputeRootShaderResourceView(2, scratch_gpu + groups_offset);
                cmd_list.SetComputeRootUnorderedAccessView(3, budget_gpu);
                cmd_list.SetComputeRootDescriptorTable(4, srv_gpu_handle);
                cmd_list.SetComputeRootDescriptorTable(5, uav_gpu_handle);
                cmd_list.Dispatch(1, 1, 1);
                dispatches += 1;
                if budget_barrier_after {
                    cmd_list.ResourceBarrier(&[uav_barrier_on(budget_res)]);
                    barriers += 1;
                }
            }
        } else {
            let costs_res = costs_res.expect("two-pass costs buffer ensured above");
            // SAFETY: live device-local UAV buffer.
            let costs_gpu = unsafe { costs_res.GetGPUVirtualAddress() };
            // Order Pass 1 UAV writes to `costs_res`, `budget_res` (mesh
            // diagnostics), and every destination chunk in `uavs` before Pass 2
            // reads `costs`/`budget` and zeroes any over-budget destination slots.
            let mut mid_barriers = Vec::with_capacity(2 + uavs.len());
            mid_barriers.push(uav_barrier_on(&costs_res));
            mid_barriers.push(uav_barrier_on(budget_res));
            for (uav_res, _) in &uavs {
                mid_barriers.push(uav_barrier_on(uav_res));
            }
            // SAFETY: open command list, live root signature/PSOs/heap/buffers;
            // every borrowed COM pointer in `mid_barriers` stays live through
            // `ResourceBarrier`.
            unsafe {
                if self.costs_used {
                    cmd_list.ResourceBarrier(&[uav_barrier_on(&costs_res)]);
                    barriers += 1;
                }
                cmd_list.SetDescriptorHeaps(&[Some(heap)]);
                cmd_list.SetComputeRootSignature(&g.twopass_root);
                cmd_list.SetPipelineState(&g.pass1_pso);
                cmd_list.SetComputeRoot32BitConstants(0, 2, constants.as_ptr().cast(), 0);
                cmd_list.SetComputeRootShaderResourceView(1, scratch_gpu);
                cmd_list.SetComputeRootShaderResourceView(2, scratch_gpu + groups_offset);
                cmd_list.SetComputeRootUnorderedAccessView(3, budget_gpu);
                cmd_list.SetComputeRootUnorderedAccessView(4, costs_gpu);
                cmd_list.SetComputeRootDescriptorTable(5, srv_gpu_handle);
                cmd_list.SetComputeRootDescriptorTable(6, uav_gpu_handle);
                cmd_list.Dispatch(num_groups, 1, 1);
                dispatches += 1;
                cmd_list.ResourceBarrier(&mid_barriers);
                barriers += 1;
                cmd_list.SetPipelineState(&g.pass2_pso);
                cmd_list.Dispatch(1, 1, 1);
                dispatches += 1;
                if budget_barrier_after {
                    cmd_list.ResourceBarrier(&[uav_barrier_on(budget_res)]);
                    barriers += 1;
                }
            }
            self.costs_used = true;
        }

        // Keep all SRV and UAV resources alive until `HostList::reset`.
        self.kept_resources.extend(srvs.into_iter().map(|(r, _)| r));
        self.kept_resources.extend(uavs.into_iter().map(|(r, _)| r));

        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::IndirectBounded, 1);

        Some(GatherFlushStats {
            items: total_items,
            groups: num_groups,
            dispatches,
            barriers,
            two_pass: !single_pass,
        })
    }
}

/// Flushes any pending gathered indirect clamps recorded into `clamps`.
/// Called by `HostList::clamp_end` before transitioning destination chunks to
/// `INDIRECT_ARGUMENT`, and by `bound_before` before a non-gatherable clamp or
/// batch capacity flush.
pub(crate) fn flush_gather(clamps: &HostList, budget_barrier_after: bool) {
    let mut state = clamps.gather_state().lock().unwrap_or_else(|e| e.into_inner());
    if state.is_empty() {
        return;
    }
    let Some((budget_gpu, budget_res)) = clamps.budget_with_resource() else {
        state.pending.clear();
        return;
    };
    if let (Some(h), Some(g)) = (host(), gather_clamp()) {
        let _ = state.flush_on(
            &h.device,
            g,
            clamp(),
            &clamps.list,
            budget_gpu,
            budget_res,
            budget_barrier_after,
            |len| clamps.scratch(len),
        );
    } else if let Some(c) = clamp() {
        let pending = std::mem::take(&mut state.pending);
        let srvs = std::mem::take(&mut state.srvs);
        state.srv_index.clear();
        let uavs = std::mem::take(&mut state.uavs);
        state.uav_index.clear();
        let _ = state.record_pending_scalar(
            c,
            &clamps.list,
            &pending,
            srvs,
            uavs,
            budget_gpu,
            budget_res,
            budget_barrier_after,
        );
    }
}

/// How clamps of indirect draws are recorded: 0 each before its draw; 1 in
/// a list of their own before the list that draws, so that a run of draws
/// is not interleaved with our dispatches; 2 as 1, with a barrier for
/// shader writes before each draw as the clamps had; 3 as 1 for mesh
/// dispatches only; 4 as 1 for draws only; 5 as 1 with a new list for the
/// clamps always; 6 the lists cut as for 1 with each clamp before its draw;
/// 7 as 1 with the clamps run one after another (a barrier between them):
/// the default, avoiding mesh flicker from out-of-order budget charging.
/// `VICTORY_HOIST_CLAMPS=<n>` sets it (0 turns hoisting off).
pub(crate) fn hoist_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        std::env::var("VICTORY_HOIST_CLAMPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(7)
    })
}

/// Whether clamps of indirect draws go to a list before the one that draws.
pub(crate) fn hoisting() -> bool {
    hoist_mode() != 0
}

/// As [`bound`], recorded in `clamps`, a list that runs before the one that
/// executes: the buffer and the offset of the clamped records, readable as
/// indirect arguments once `clamps` ended ([`HostList::clamp_end`]).
pub(crate) fn bound_before(
    clamps: &HostList,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
) -> Option<(ID3D12Resource, u64)> {
    let bytes = validate_layout((kind, stride, records), src_offset, count.map(|(_, off)| off))?;
    if records == 0 {
        return None;
    }
    let (budget, budget_res) = clamps.budget_with_resource()?;
    let (dst, at) = clamps.clamp_room(bytes)?;

    if gather_enabled() && records <= GATHER_MAX_RECORDS && gather_clamp().is_some() {
        let mut state = clamps.gather_state().lock().unwrap_or_else(|e| e.into_inner());
        if state.would_overflow(records, src, count.map(|(b, _)| b), &dst)
            && let (Some(h), Some(g)) = (host(), gather_clamp())
        {
            // Always serialize `budget_res` across consecutive gather flushes
            // (regardless of `hoist_mode()`) so the next flush's `b0` load
            // observes this flush's `InterlockedMax` budget update.
            let _ = state.flush_on(
                &h.device,
                g,
                clamp(),
                &clamps.list,
                budget,
                budget_res,
                true,
                |len| clamps.scratch(len),
            );
        }
        if state.enqueue((src, src_offset), count, (&dst, at), (kind, stride, records), 0) {
            return Some((dst, at));
        }
        // Fallback to scalar path below: always flush queued gathered items
        // with a `budget_res` UAV barrier before this scalar dispatch.
        if !state.is_empty()
            && let (Some(h), Some(g)) = (host(), gather_clamp())
        {
            let _ = state.flush_on(
                &h.device,
                g,
                clamp(),
                &clamps.list,
                budget,
                budget_res,
                true,
                |len| clamps.scratch(len),
            );
        }
    } else {
        // Flush any queued gathered clamps with a `budget_res` UAV barrier
        // before recording a scalar clamp so budget updates never race.
        flush_gather(clamps, true);
    }

    let c = clamp()?;
    // SAFETY: live buffers.
    let (src_base, count_base) = unsafe {
        (
            src.GetGPUVirtualAddress(),
            count.map(|(b, _)| b.GetGPUVirtualAddress()),
        )
    };
    let fits = |offset: u64, len: u64| {
        offset
            .checked_add(len)
            .is_some_and(|end| end <= u64::from(u32::MAX))
    };
    let (src_gpu, args_offset) = if fits(src_offset, bytes) {
        (src_base, src_offset as u32)
    } else {
        (src_base.checked_add(src_offset)?, 0)
    };
    let (count_gpu, has_count, count_offset) = match (count, count_base) {
        (Some((_, off)), Some(base)) if fits(off, 4) => (base, 1u32, off as u32),
        (Some((_, off)), Some(base)) => (base.checked_add(off)?, 1, 0),
        _ => (src_gpu, 0, 0),
    };
    if !src_gpu.is_multiple_of(4) || !count_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return None;
    }
    // SAFETY: a live buffer.
    let dst_gpu = unsafe { dst.GetGPUVirtualAddress() } + at;
    let constants = [kind, stride, records, args_offset, has_count, count_offset];
    let l = &clamps.list;
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, destination (writable: `clamp_room`) and budget buffers.
    unsafe {
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
        l.SetComputeRootShaderResourceView(1, src_gpu);
        l.SetComputeRootShaderResourceView(2, count_gpu);
        l.SetComputeRootUnorderedAccessView(3, dst_gpu);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(records.div_ceil(64), 1, 1);
        // Mode 7 (or whenever gather batching is enabled and a later gather
        // flush on this list may load `budget[0]` once): order budget updates
        // without a full-GPU global UAV drain (destination offsets are disjoint).
        if hoist_mode() == 7 || gather_enabled() {
            l.ResourceBarrier(&[uav_barrier_on(budget_res)]);
        }
    }
    Some((dst, at))
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    done_on(&list.list, scratch);
}

fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
}

/// A wait for earlier unordered writes to `resource` only.
fn uav_barrier_on(resource: &ID3D12Resource) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            // Borrowed, not owned: the barrier is used while `resource` lives
            // and its reference is never released (`ManuallyDrop`).
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                // SAFETY: copies the interface pointer without AddRef inside
                // `ManuallyDrop` so no Release follows.
                pResource: std::mem::ManuallyDrop::new(unsafe { std::mem::transmute_copy(resource) }),
            }),
        },
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        _warp_guard: std::sync::MutexGuard<'static, ()>,
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // Serialize WARP tests: concurrent queues/PSOs on `d3d10warp.dll`
            // hit an intermittent null-deref (`0xc0000005`) in the software
            // rasterizer under heavy multi-thread contention.
            static WARP_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            let _warp_guard = WARP_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            static WARP_DEVICE: std::sync::OnceLock<Option<ID3D12Device>> =
                std::sync::OnceLock::new();
            // Keep one WARP device alive for the test binary so concurrent
            // tests cannot race `d3d10warp.dll` teardown (`Release` -> 0) with
            // another thread's `D3D12CreateDevice(&warp_adapter)`.
            let device = WARP_DEVICE
                .get_or_init(|| {
                    // SAFETY: plain DXGI WARP query and D3D12 device creation.
                    unsafe {
                        let factory: IDXGIFactory4 =
                            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                        let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                        let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                            return None;
                        }
                        let mut device: Option<ID3D12Device> = None;
                        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                        device
                    }
                })
                .clone()?;
            // SAFETY: plain D3D12 test harness setup on the live WARP device.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    _warp_guard,
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let code = compile_blob(source, s!("worker"), 0)?;
        // SAFETY: compiled DXBC blob with embedded root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let (_inline, inline_ptr, inline_gpu) = warp.shared_buffer(64);
        // SAFETY: mapping of 64 bytes.
        unsafe { inline_ptr.cast::<u32>().write_volatile(0) };
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode, 0];
                warp.list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.SetComputeRootUnorderedAccessView(4, inline_gpu);
                warp.list.Dispatch(PIECES, 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// WARP D3D12 execution test for `CARD_COMPARE` inline readback (`mode=4`),
    /// non-16-byte-aligned `skew`, and graceful overflow fallback (`differs=1`).
    #[test]
    fn warp_card_compare_inline_readback_and_overflow_fallback() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        const SKEW: u32 = 7; // non-16-byte aligned: piece 1 widens to 65,552 bytes
        let size = PIECE * PIECES as usize;
        let inline_bytes = crate::host::CARD_INLINE_HEADER + 4 * crate::host::CARD_INLINE_STRIDE;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let (_inline, inline_ptr, inline_gpu) = warp.shared_buffer(inline_bytes as u64);
        // SAFETY: mappings hold their allocated sizes.
        unsafe {
            inline_ptr.cast::<u32>().write_volatile(0);
            for i in 0..size {
                now_ptr.add(i).write_volatile(((i * 37) ^ (i >> 5) ^ 0xA5) as u8);
            }
        }
        let run = |warp: &mut WarpHarness, mode: u32, inline_limit: u32| -> [u32; 3] {
            // SAFETY: live buffers and valid root signature.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(99);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, PIECES, 0, mode, inline_limit];
                warp.list.SetComputeRoot32BitConstants(0, 7, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.SetComputeRootUnorderedAccessView(4, inline_gpu);
                warp.list.Dispatch(PIECES, 1, 1);
            }
            warp.submit_and_wait();
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        // Initial hash pass.
        assert_eq!(run(&mut warp, 0, 0), [99, 99, 99]);
        assert_eq!(run(&mut warp, 4, 2), [0, 0, 0]);
        // Mutate piece 1 (with skew=7 -> shader_lo = (65536 - 7) & !15 = 65520,
        // shader_hi = (131072 - 7 + 15) & !15 = 131072, span = 65552 bytes!).
        let p1_lo = (PIECE - SKEW as usize) & !15;
        let p1_hi = ((2 * PIECE - SKEW as usize) + 15) & !15;
        assert_eq!(p1_hi - p1_lo, 65552);
        // Mutate bytes strictly inside piece 1 (avoiding the 16-byte widened
        // overlap `[131056, 131072)` with piece 2), while still checking that
        // all 65,552 widened bytes `[p1_lo..p1_hi)` are copied inline verbatim.
        // SAFETY: offsets lie inside `now_ptr[..size]`.
        unsafe {
            let cur = now_ptr.add(p1_lo + 123).read_volatile();
            now_ptr.add(p1_lo + 123).write_volatile(cur ^ 0x5A);
            let tail = now_ptr.add(p1_hi - 17).read_volatile();
            now_ptr.add(p1_hi - 17).write_volatile(tail ^ 0xFF);
        }
        let res = run(&mut warp, 4, 2);
        assert_eq!(res[0], 0);
        assert_eq!(res[2], 0);
        assert_eq!(res[1], 2, "first inline ticket must be 0 -> differs value 2");
        // Verify all 65,552 bytes copied inline match `now[p1_lo..p1_hi]` byte for byte.
        // SAFETY: mappings hold `size` and `inline_bytes`.
        unsafe {
            let expected = std::slice::from_raw_parts(now_ptr.add(p1_lo), p1_hi - p1_lo);
            let got = std::slice::from_raw_parts(inline_ptr.add(crate::host::CARD_INLINE_HEADER), p1_hi - p1_lo);
            assert_eq!(got, expected);
        }
        // Mutate all 3 pieces with `inline_limit = 2` (only 1 ticket left: ticket 1).
        // Exactly 1 piece gets ticket 1 (`differs == 3`), the other 2 overflow (`differs == 1`).
        // SAFETY: offsets lie inside `now_ptr[..size]`.
        unsafe {
            for off in [17usize, p1_lo + 333, 2 * PIECE + 44] {
                let cur = now_ptr.add(off).read_volatile();
                now_ptr.add(off).write_volatile(cur ^ 0x77);
            }
        }
        let overflow = run(&mut warp, 4, 2);
        assert_eq!(overflow.iter().filter(|&&d| d == 3).count(), 1, "one piece should take ticket 1: {overflow:?}");
        assert_eq!(overflow.iter().filter(|&&d| d == 1).count(), 2, "two pieces should overflow to fallback 1: {overflow:?}");
        // Re-running with no changes must see 0 across all pieces (hashes were updated).
        assert_eq!(run(&mut warp, 4, 4), [0, 0, 0]);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        // Snapshot `dst_buf` after Call #1 (from `COMMON`, copy 5 * STRIDE bytes,
        // then return `dst_buf` to `COMMON` before Call #2).
        // SAFETY: open command list; `dst_buf` returned to `COMMON` by `done_on`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }
        done_on(&warp.list, &dst_buf);

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                &dst_buf,
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            done_on(&warp.list, &dst_buf);
            // SAFETY: an open list; the scratch buffer is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                )]);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            let words: [u32; 6] = [
                u32::MAX,                    // IndexCount -> clamps to 1 << 24
                u32::MAX,                    // InstanceCount -> clamps to 4 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
        }

        assert!(record_clamp(
            &warp.clamp,
            &warp.list,
            &dst_buf,
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
        ));
        done_on(&warp.list, &dst_buf);

        // SAFETY: open command list; copy clamped scratch buffer for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 24) && rec[1] == 4 {
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `done`, `Host::execute`,
    /// and `HostList::reset` end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64), true).unwrap();
        done(&list, &b1);
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128), true).unwrap();
        done(&list, &b2);
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset.
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000), true).unwrap();
        done(&list, &b3);
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );

        list.close();
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations.
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096), true).unwrap();
        done(&list, &b4);
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        list.close();
        host.execute(&[&list]);
        list.reset();
    }

    /// Verifies `record_clamp_waits` with `scoped: true, before: true, after: false`
    /// orders a preceding compute UAV write into the clamped scratch buffer and
    /// into `ExecuteIndirect` without emitting any global UAV barrier.
    #[test]
    fn warp_scoped_clamp_orders_preceding_uav_write() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const WRITER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer args : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    // Dispatch(5, 1, 1) -> 5 worker hits
    args.Store3(0, uint3(5, 1, 1));
}
"#;
        let writer = compile_cs(&warp.device, WRITER_CS).expect("writer shader must compile");
        let (src, src_ptr, src_gpu) = warp.shared_buffer(64);
        let (dst, _, _) = warp.shared_buffer(64);
        let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(64);
        let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(64);
        // SAFETY: mappings hold 64 bytes each.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0, 64);
            std::ptr::write_bytes(budget_ptr, 0, 64);
            std::ptr::write_bytes(hits_ptr, 0, 64);
        }
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: 12,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: valid 1-argument dispatch command signature.
        unsafe {
            warp.device
                .CreateCommandSignature(&sig_desc, None::<&ID3D12RootSignature>, &mut sig)
                .expect("dispatch signature");
        }
        let sig = sig.unwrap();

        // 1. Writer shader writes (5, 1, 1) via UAV into `src`.
        // SAFETY: live buffers and valid compute pipeline.
        unsafe {
            warp.list.SetComputeRootSignature(&writer.root);
            warp.list.SetPipelineState(&writer.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, src_gpu);
            warp.list.Dispatch(1, 1, 1);
        }
        // 2. Scoped clamp waits ONLY on `src` (no global barrier before or after).
        assert!(record_clamp_waits(
            &warp.clamp,
            &warp.list,
            &dst,
            budget_gpu,
            (&src, 0),
            None,
            (2, 12, 1),
            0,
            Waits {
                before: true,
                before_src: true,
                before_count: false,
                after: false,
                scoped: true,
            },
        ));
        // 3. ExecuteIndirect runs `worker` with the clamped record.
        // SAFETY: `dst` is in `INDIRECT_ARGUMENT` and `worker` writes `hits`.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list.ExecuteIndirect(&sig, 1, &dst, 0, None, 0);
        }
        done_on(&warp.list, &dst);
        warp.submit_and_wait();
        // SAFETY: completed execution; mapping holds 64 bytes.
        let hits = unsafe { hits_ptr.cast::<u32>().read_volatile() };
        assert_eq!(hits, 5, "scoped clamp must order preceding UAV write into ExecuteIndirect");
    }
}

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        #[allow(dead_code)]
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}

#[cfg(test)]
mod gather_hw_tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let code = compile_blob(source, s!("worker"), 0)?;
        // SAFETY: compiled DXBC blob with embedded root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
        }
    }

    fn hw_worker(device: &ID3D12Device) -> &'static Clamp {
        static WORKER: std::sync::OnceLock<Clamp> = std::sync::OnceLock::new();
        WORKER.get_or_init(|| {
            compile_cs(device, WORKER_CS).expect("worker shader must compile on HW GPU")
        })
    }

    /// Hardware D3D12 harness for bit-exact gather-clamp parity tests and
    /// real-GPU benchmarks. Enables the D3D12 debug layer when
    /// `VICTORY_D3D12_DEBUG` is set and asserts zero debug-layer errors after
    /// every submission.
    struct HwGatherHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: &'static Clamp,
        gather: &'static GatherClamp,
        worker: &'static Clamp,
        stamps: ID3D12QueryHeap,
        stamp_readback: ID3D12Resource,
        info: Option<ID3D12InfoQueue>,
        card: String,
    }

    impl HwGatherHarness {
        fn new() -> Option<Self> {
            let require_hw = std::env::var_os("VICTORY_REQUIRE_HW_GPU")
                .is_some_and(|v| v != "0");
            let debug_on = cfg!(feature = "diagnostics")
                && std::env::var_os("VICTORY_D3D12_DEBUG").is_some_and(|v| v != "0");
            // Reuse `crate::host::host()`'s singleton device so `EnableDebugLayer()`
            // is called strictly once before `D3D12CreateDevice` (calling
            // `EnableDebugLayer` on another thread after an existing non-debug
            // device is created removes the live device with DXGI_ERROR_DEVICE_RESET).
            let Some(h) = crate::host::host() else {
                assert!(
                    !require_hw,
                    "VICTORY_REQUIRE_HW_GPU=1 set, but host D3D12 device is unavailable"
                );
                return None;
            };
            let device = h.device.clone();
            // SAFETY: plain DXGI / D3D12 queries and test-rig creation on the live device.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).ok()?;
                let desc = adapter.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    assert!(
                        !require_hw,
                        "VICTORY_REQUIRE_HW_GPU=1 set, but host adapter is software"
                    );
                    return None;
                }
                let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
                let card = String::from_utf16_lossy(&desc.Description[..end]);
                let info = debug_on
                    .then(|| device.cast::<ID3D12InfoQueue>().ok())
                    .flatten();
                if let Some(iq) = &info {
                    iq.ClearStoredMessages();
                }
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = clamp().expect("scalar clamp shader must compile on HW GPU");
                let gather = gather_clamp().expect("gather clamp shaders must compile on HW GPU");
                let worker = hw_worker(&device);
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;
                let stamp_readback = buffer(
                    &device,
                    256,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    gather,
                    worker,
                    stamps,
                    stamp_readback,
                    info,
                    card,
                })
            }
        }

        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut out: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: valid buffer creation and persistent CPU map.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut out,
                    )
                    .unwrap();
                let res = out.unwrap();
                res.Map(0, None, Some(&mut data)).unwrap();
                let gpu = res.GetGPUVirtualAddress();
                (res, data.cast(), gpu)
            }
        }

        fn upload_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let res = buffer(
                &self.device,
                size.max(256).next_multiple_of(256),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let mut data = std::ptr::null_mut();
            // SAFETY: persistent map of upload buffer.
            unsafe {
                res.Map(0, None, Some(&mut data)).unwrap();
                let gpu = res.GetGPUVirtualAddress();
                (res, data.cast(), gpu)
            }
        }

        fn default_uav(&self, size: u64) -> ID3D12Resource {
            buffer(
                &self.device,
                size.max(256),
                D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn assert_debug_clean(&self) {
            let Some(info) = &self.info else {
                return;
            };
            // SAFETY: querying D3D12 InfoQueue messages owned by our test device.
            unsafe {
                let n = info.GetNumStoredMessages();
                let mut errors = Vec::new();
                for i in 0..n {
                    let mut len = 0usize;
                    let _ = info.GetMessage(i, None, &mut len);
                    if len == 0 {
                        continue;
                    }
                    let mut buf = vec![0u8; len];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if info.GetMessage(i, Some(msg), &mut len).is_ok() {
                        let sev = (*msg).Severity;
                        // `D3D12CreateDevice` returns a process-wide singleton per
                        // adapter, so `pipeline::library::tests` running in parallel
                        // can intentionally log an invalid-blob error on the shared
                        // InfoQueue. Ignore only that unrelated ID.
                        if (*msg).ID == D3D12_MESSAGE_ID_CREATEPIPELINELIBRARY_INVALIDLIBRARYBLOB {
                            continue;
                        }
                        if sev == D3D12_MESSAGE_SEVERITY_CORRUPTION
                            || sev == D3D12_MESSAGE_SEVERITY_ERROR
                        {
                            let text = std::slice::from_raw_parts(
                                (*msg).pDescription.cast::<u8>(),
                                (*msg).DescriptionByteLength.saturating_sub(1),
                            );
                            errors.push(String::from_utf8_lossy(text).into_owned());
                        }
                    }
                }
                info.ClearStoredMessages();
                assert!(
                    errors.is_empty(),
                    "D3D12 debug layer reported {} error(s):\n{}",
                    errors.len(),
                    errors.join("\n")
                );
            }
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, submitting, fencing, waiting, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                assert!(
                    self.device.GetDeviceRemovedReason().is_ok(),
                    "GPU device removed during test"
                );
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
            self.assert_debug_clean();
        }

        fn read_timestamp_ms(&self) -> f64 {
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: reading resolved GPU timestamps after fence completion.
            unsafe {
                self.stamp_readback.Map(0, Some(&range), Some(&mut data)).unwrap();
                let ticks = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.stamp_readback.Unmap(0, None);
                let freq = self.queue.GetTimestampFrequency().unwrap_or(1).max(1);
                ticks[1].saturating_sub(ticks[0]) as f64 * 1e3 / freq as f64
            }
        }

        fn read_default_words(&mut self, src: &ID3D12Resource, bytes: u64) -> Vec<u32> {
            let rb = buffer(
                &self.device,
                bytes.max(4),
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            // SAFETY: copying completed DEFAULT buffer to READBACK.
            unsafe {
                self.list.CopyBufferRegion(&rb, 0, src, 0, bytes);
            }
            self.submit_and_wait();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE {
                Begin: 0,
                End: bytes as usize,
            };
            // SAFETY: reading mapped READBACK buffer.
            unsafe {
                rb.Map(0, Some(&range), Some(&mut data)).unwrap();
                let out = std::slice::from_raw_parts(data.cast::<u32>(), (bytes / 4) as usize).to_vec();
                rb.Unmap(0, None);
                out
            }
        }
    }

    /// Record a scalar reference clamp into `dst` at `dst_offset` (already in
    /// `UNORDERED_ACCESS`) followed by a Mode 7 budget UAV barrier.
    #[allow(clippy::too_many_arguments)]
    fn record_scalar_into_uav(
        c: &Clamp,
        l: &ID3D12GraphicsCommandList,
        dst: &ID3D12Resource,
        dst_offset: u64,
        budget_gpu: u64,
        budget_res: &ID3D12Resource,
        (src, src_offset): (&ID3D12Resource, u64),
        count: Option<(&ID3D12Resource, u64)>,
        (kind, stride, records): (u32, u32, u32),
        how: u32,
        mode7_barrier: bool,
    ) {
        // SAFETY: live D3D12 buffers and open command list.
        unsafe {
            let src_gpu = src.GetGPUVirtualAddress();
            let (count_gpu, has_count, count_offset) = match count {
                Some((buf, off)) => (buf.GetGPUVirtualAddress(), 1u32, off as u32),
                None => (src_gpu, 0u32, 0u32),
            };
            let dst_gpu = dst.GetGPUVirtualAddress() + dst_offset;
            let constants = [
                kind,
                stride,
                records,
                src_offset as u32,
                has_count | how,
                count_offset,
            ];
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget_gpu);
            l.Dispatch(records.div_ceil(64), 1, 1);
            if mode7_barrier {
                l.ResourceBarrier(&[uav_barrier_on(budget_res)]);
            }
        }
    }

    /// Proves bit-exact parity of gathered clamps vs scalar reference clamps on
    /// real hardware GPU (with `VICTORY_D3D12_DEBUG=1`) across all four kinds
    /// (`DRAW`, `INDEXED`, `DISPATCH`, `MESH`), overflow/garbage inputs, `AHEAD`,
    /// `SPLIT`, count buffers, trailing words, and BOTH single-pass and two-pass
    /// gather paths.
    #[test]
    fn hw_gather_bit_exact_parity_all_kinds_and_flags() {
        let Some(mut hw) = HwGatherHarness::new() else {
            return;
        };
        struct Case {
            buf_idx: usize,
            src_offset: u64,
            count_idx: Option<(usize, u64)>,
            kind: u32,
            stride: u32,
            records: u32,
            how: u32,
            dst_offset: u64,
            dst_bytes: u64,
        }

        for two_pass_mode in [false, true] {
            let num_bufs = if two_pass_mode { 20 } else { 1 };
            let mut src_bufs = Vec::new();
            for _ in 0..num_bufs {
                src_bufs.push(hw.shared_buffer(4096));
            }
            let (cnt_buf, cnt_ptr, _) = hw.shared_buffer(256);
            // SAFETY: persistent mapping of 256 bytes.
            unsafe {
                let c = cnt_ptr.cast::<u32>();
                c.add(0).write_volatile(3); // partial count (< max 6)
                c.add(1).write_volatile(0); // zero count
                c.add(2).write_volatile(100); // count > max
            }

            let mut cases = Vec::new();
            let mut dst_cursor = 0u64;
            let mut push_case = |buf_idx: usize,
                                 src_offset: u64,
                                 words: &[u32],
                                 count_idx: Option<(usize, u64)>,
                                 kind: u32,
                                 stride: u32,
                                 records: u32,
                                 how: u32| {
                // SAFETY: writing into mapped source buffer within 4096 bytes.
                unsafe {
                    let dst_p = src_bufs[buf_idx].1.add(src_offset as usize).cast::<u32>();
                    for (k, &w) in words.iter().enumerate() {
                        dst_p.add(k).write_volatile(w);
                    }
                }
                let bytes = if (how & IN_PARTS) != 0 {
                    u64::from(SPLIT) * PART_BYTES
                } else {
                    u64::from(stride) * u64::from(records)
                };
                let at = dst_cursor;
                dst_cursor = (dst_cursor + bytes).next_multiple_of(16);
                cases.push(Case {
                    buf_idx,
                    src_offset,
                    count_idx,
                    kind,
                    stride,
                    records,
                    how,
                    dst_offset: at,
                    dst_bytes: bytes,
                });
            };

            // 1. DRAW normal + trailing words (stride 24).
            push_case(0, 0, &[360, 4, 17, 99, 0x1234_5678, 0xdead_beef], None, 0, 24, 1, 0);
            // 2. DRAW overflow vertex + instance counts.
            push_case(1 % num_bufs, 64, &[0xFFFF_FFFF, 0xFFFF_FFFF, 5, 11], None, 0, 16, 1, 0);
            // 3. DRAW with AHEAD flag and garbage (> 1<<24) -> zeroed, 0 budget.
            push_case(2 % num_bufs, 128, &[0xFFFF_FFFF, 2, 7, 8], None, 0, 16, 1, AHEAD);
            // 4. DRAW with zero vertices and non-zero instances (cost == 0, keep instances).
            push_case(3 % num_bufs, 192, &[0, 7, 1, 2], None, 0, 16, 1, 0);
            // 5. INDEXED normal + trailing words (stride 28).
            push_case(4 % num_bufs, 256, &[1200, 3, 10, 0xFFFF_FFFC, 42, 0xaaaa_bbbb, 0x5555_6666], None, 1, 28, 1, 0);
            // 6. INDEXED with count buffer: max=6, count=3 (tail 3 slots hold garbage that must be untouched).
            let mut counted_words = Vec::new();
            for r in 0..6u32 {
                if r < 3 {
                    counted_words.extend_from_slice(&[100 + r * 10, 2, r, 1000 + r, 50 + r]);
                } else {
                    counted_words.extend_from_slice(&[0xFFFF_FFFF, 0xFFFF_FFFF, 0xBAD0_0000 + r, 0, 0]);
                }
            }
            push_case(5 % num_bufs, 320, &counted_words, Some((0, 0)), 1, 20, 6, 0);
            // 7. DRAW with count=0 -> completely skipped, sentinel untouched.
            push_case(6 % num_bufs, 512, &[500, 500, 1, 2], Some((0, 4)), 0, 16, 2, 0);
            // 8. DISPATCH normal + trailing padding (stride 20).
            push_case(7 % num_bufs, 576, &[128, 4, 2, 0x1111_2222, 0x3333_4444], None, 2, 20, 1, 0);
            // 9. DISPATCH product > GROUPS (1<<20) -> zeroed.
            push_case(8 % num_bufs, 640, &[2048, 2048, 1], None, 2, 12, 1, 0);
            // 10. DISPATCH SPLIT (115,200 groups -> 65535 + 49665).
            push_case(9 % num_bufs, 704, &[115_200, 1, 1], None, 2, 12, 1, IN_PARTS);
            // 11. DISPATCH SPLIT overflow (> 2 * 65535 -> clamped to 65535 + 0).
            push_case(10 % num_bufs, 768, &[200_000, 1, 1], None, 2, 12, 1, IN_PARTS);
            // 12. MESH normal + diagnostics update.
            push_case(11 % num_bufs, 832, &[300, 4, 9, 0x7777_8888], None, 10, 16, 1, 0);
            // 13. MESH AHEAD garbage (0xFFFF_FFFF > 65535) -> zeroed, not counted in diagnostics.
            push_case(12 % num_bufs, 896, &[0xFFFF_FFFF, 1, 1], None, 10, 12, 1, AHEAD);
            // 14. MESH non-ahead huge (> 65535) -> clamped to (65535, 1, 1) and diagnostics updated.
            push_case(13 % num_bufs, 960, &[100_000, 2, 3], None, 10, 12, 1, 0);
            // Extra scattered cases: in single-pass mode keep count binding
            // contiguous so `seq_groups <= 8` exercises the 1-dispatch path;
            // in two-pass mode spread across 20 buffers + alternating counts.
            for extra in 0..12usize {
                let b = (14 + extra) % num_bufs;
                let cnt = if two_pass_mode {
                    (extra % 3 == 0).then_some((0, 8))
                } else {
                    None
                };
                push_case(
                    b,
                    1024 + (extra as u64) * 32,
                    &[25 + extra as u32, 1, 1],
                    cnt,
                    if extra % 2 == 0 { 10 } else { 2 },
                    12,
                    1,
                    0,
                );
            }

            let total_bytes = dst_cursor.next_multiple_of(256);
            // Pre-fill both destinations with a sentinel pattern so untouched tail slots
            // (count=0 / rec >= count) are verified to stay untouched too!
            let (sentinel_src, sentinel_ptr, _) = hw.upload_buffer(total_bytes);
            // SAFETY: persistent mapping of `total_bytes`.
            unsafe {
                let s = std::slice::from_raw_parts_mut(sentinel_ptr.cast::<u32>(), (total_bytes / 4) as usize);
                s.fill(0xA5A5_5A5A);
            }

            let dst_scalar = hw.default_uav(total_bytes);
            let dst_gather = hw.default_uav(total_bytes);
            let (budget_scalar, budget_scalar_ptr, budget_scalar_gpu) = hw.shared_buffer(256);
            let (budget_gather, budget_gather_ptr, budget_gather_gpu) = hw.shared_buffer(256);
            // SAFETY: zeroing 12 bytes of each budget buffer.
            unsafe {
                for k in 0..3 {
                    budget_scalar_ptr.cast::<u32>().add(k).write_volatile(0);
                    budget_gather_ptr.cast::<u32>().add(k).write_volatile(0);
                }
                hw.list.CopyBufferRegion(&dst_scalar, 0, &sentinel_src, 0, total_bytes);
                hw.list.CopyBufferRegion(&dst_gather, 0, &sentinel_src, 0, total_bytes);
                hw.list.ResourceBarrier(&[
                    transition(
                        &dst_scalar,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ),
                    transition(
                        &dst_gather,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ),
                ]);
            }

            // Record scalar reference.
            for c in &cases {
                let count_arg = c.count_idx.map(|(_, off)| (&cnt_buf, off));
                record_scalar_into_uav(
                    hw.clamp,
                    &hw.list,
                    &dst_scalar,
                    c.dst_offset,
                    budget_scalar_gpu,
                    &budget_scalar,
                    (&src_bufs[c.buf_idx].0, c.src_offset),
                    count_arg,
                    (c.kind, c.stride, c.records),
                    c.how,
                    true,
                );
            }

            // Record gather-clamp batch.
            let mut gather_state = GatherState::default();
            for c in &cases {
                let count_arg = c.count_idx.map(|(_, off)| (&cnt_buf, off));
                assert!(gather_state.enqueue(
                    (&src_bufs[c.buf_idx].0, c.src_offset),
                    count_arg,
                    (&dst_gather, c.dst_offset),
                    (c.kind, c.stride, c.records),
                    c.how,
                ));
            }
            let (scratch_res, scratch_ptr, scratch_gpu) = hw.upload_buffer(64 << 10);
            let stats = gather_state
                .flush_on(
                    &hw.device,
                    hw.gather,
                    Some(hw.clamp),
                    &hw.list,
                    budget_gather_gpu,
                    &budget_gather,
                    false,
                    |len| {
                        assert!(len <= (64 << 10));
                        Some((scratch_gpu, scratch_ptr))
                    },
                )
                .expect("gather flush must succeed");
            assert_eq!(stats.two_pass, two_pass_mode);
            // SAFETY: transition destinations back to COMMON before submit_and_wait.
            unsafe {
                hw.list.ResourceBarrier(&[
                    transition(
                        &dst_scalar,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition(
                        &dst_gather,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            }
            hw.submit_and_wait();
            drop(scratch_res);

            let scalar_words = hw.read_default_words(&dst_scalar, total_bytes);
            let gather_words = hw.read_default_words(&dst_gather, total_bytes);
            assert_eq!(
                gather_words, scalar_words,
                "gather output diverged from scalar reference (two_pass={two_pass_mode})"
            );
            // SAFETY: reading budget words after fence completion.
            let (bs, bg) = unsafe {
                let r = |p: *mut u8| {
                    let w = p.cast::<u32>();
                    [w.add(0).read_volatile(), w.add(1).read_volatile(), w.add(2).read_volatile()]
                };
                (r(budget_scalar_ptr), r(budget_gather_ptr))
            };
            assert_eq!(
                bg, bs,
                "budget words diverged (two_pass={two_pass_mode}): gather={bg:?} scalar={bs:?}"
            );
            // Also verify count=3 left slots 3..5 as untouched sentinel 0xA5A5_5A5A.
            let c6 = &cases[5];
            let tail_start = ((c6.dst_offset + 3 * u64::from(c6.stride)) / 4) as usize;
            let tail_end = ((c6.dst_offset + c6.dst_bytes) / 4) as usize;
            assert!(
                gather_words[tail_start..tail_end]
                    .iter()
                    .all(|&w| w == 0xA5A5_5A5A),
                "inactive counted tail slots were unexpectedly overwritten"
            );
            gather_state.reset();
        }
    }

    /// Proves bit-exact Mode 7 index-order budget saturation (`BUDGET = 1<<27`)
    /// across 256 scattered records in BOTH single-pass (1 buffer) and two-pass
    /// (32 interleaved buffers) modes, including exact boundary (`rem == cost`)
    /// and non-zero initial budget.
    #[test]
    fn hw_gather_budget_saturation_exact_order() {
        let Some(mut hw) = HwGatherHarness::new() else {
            return;
        };
        const N: usize = 256;
        const STRIDE: u32 = 20;
        const SLOT_ALIGN: u64 = 32;
        let total_bytes = (N as u64 * SLOT_ALIGN).next_multiple_of(256);

        for (num_bufs, initial_budget, expect_two_pass) in [
            (1usize, 0u32, false),
            (1usize, (1u32 << 27) - 5_000_000, false),
            (32usize, 0u32, true),
            (32usize, (1u32 << 27) - 1_000_000, true),
        ] {
            let mut src_bufs = Vec::new();
            for _ in 0..num_bufs {
                src_bufs.push(hw.shared_buffer(16 << 10));
            }
            for i in 0..N {
                let b = i % num_bufs;
                let off = (i / num_bufs) * 32;
                // Each record costs 1_000_000 vertices (10_000 * 100); 134 records fit
                // cleanly into BUDGET=134_217_728, record 134 saturates! Also inject
                // an exact-boundary record at i == 134 for initial_budget == 0.
                let (verts, insts) = if initial_budget == 0 && i == 134 {
                    (217_728u32, 1u32) // exact remaining budget!
                } else if i % 17 == 0 {
                    (0u32, 5u32) // zero-cost record: must stay (0, 5) even after saturation!
                } else {
                    (10_000u32, 100u32)
                };
                // SAFETY: writing 20 bytes inside mapped 16 KiB source buffer.
                unsafe {
                    let p = src_bufs[b].1.add(off).cast::<u32>();
                    p.add(0).write_volatile(verts);
                    p.add(1).write_volatile(insts);
                    p.add(2).write_volatile(i as u32);
                    p.add(3).write_volatile(0x1000 + i as u32);
                    p.add(4).write_volatile(0x2000 + i as u32);
                }
            }

            let dst_scalar = hw.default_uav(total_bytes);
            let dst_gather = hw.default_uav(total_bytes);
            let (budget_s, budget_s_ptr, budget_s_gpu) = hw.shared_buffer(256);
            let (budget_g, budget_g_ptr, budget_g_gpu) = hw.shared_buffer(256);
            // SAFETY: initializing budget words and transitioning destinations.
            unsafe {
                budget_s_ptr.cast::<u32>().write_volatile(initial_budget);
                budget_g_ptr.cast::<u32>().write_volatile(initial_budget);
                for k in 1..3 {
                    budget_s_ptr.cast::<u32>().add(k).write_volatile(0);
                    budget_g_ptr.cast::<u32>().add(k).write_volatile(0);
                }
                hw.list.ResourceBarrier(&[
                    transition(
                        &dst_scalar,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ),
                    transition(
                        &dst_gather,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    ),
                ]);
            }

            let mut gather_state = GatherState::default();
            for i in 0..N {
                let b = i % num_bufs;
                let src_off = ((i / num_bufs) * 32) as u64;
                let dst_off = (i as u64) * SLOT_ALIGN;
                record_scalar_into_uav(
                    hw.clamp,
                    &hw.list,
                    &dst_scalar,
                    dst_off,
                    budget_s_gpu,
                    &budget_s,
                    (&src_bufs[b].0, src_off),
                    None,
                    (1, STRIDE, 1),
                    0,
                    true,
                );
                assert!(gather_state.enqueue(
                    (&src_bufs[b].0, src_off),
                    None,
                    (&dst_gather, dst_off),
                    (1, STRIDE, 1),
                    0,
                ));
            }
            let (scratch_res, scratch_ptr, scratch_gpu) = hw.upload_buffer(64 << 10);
            let stats = gather_state
                .flush_on(
                    &hw.device,
                    hw.gather,
                    Some(hw.clamp),
                    &hw.list,
                    budget_g_gpu,
                    &budget_g,
                    false,
                    |len| (len <= (64 << 10)).then_some((scratch_gpu, scratch_ptr)),
                )
                .expect("gather flush must succeed");
            assert_eq!(stats.two_pass, expect_two_pass);

            // SAFETY: transitioning destinations back to COMMON.
            unsafe {
                hw.list.ResourceBarrier(&[
                    transition(
                        &dst_scalar,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                    transition(
                        &dst_gather,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            }
            hw.submit_and_wait();
            drop(scratch_res);

            let sw = hw.read_default_words(&dst_scalar, total_bytes);
            let gw = hw.read_default_words(&dst_gather, total_bytes);
            assert_eq!(
                gw, sw,
                "saturation output diverged (num_bufs={num_bufs}, init={initial_budget})"
            );
            // SAFETY: reading final budgets after GPU fence.
            let (bs0, bg0) = unsafe {
                (
                    budget_s_ptr.cast::<u32>().read_volatile(),
                    budget_g_ptr.cast::<u32>().read_volatile(),
                )
            };
            assert_eq!(bg0, 1u32 << 27, "budget should have saturated at 1<<27");
            assert_eq!(bg0, bs0);
            gather_state.reset();
        }
    }

    /// Tests mid-list hazard flush (gather batch 1 -> Mode 7 budget barrier ->
    /// wide scalar clamp -> gather batch 2 -> transition to `INDIRECT_ARGUMENT`
    /// -> real GPU `ExecuteIndirect` running `WORKER_CS` on clamped dispatches).
    #[test]
    fn hw_gather_mid_list_hazard_flush_and_execute_indirect() {
        let Some(mut hw) = HwGatherHarness::new() else {
            return;
        };
        let (src, src_ptr, _) = hw.shared_buffer(4096);
        let (_hits, hits_ptr, hits_gpu) = hw.shared_buffer(64);
        let (budget, budget_ptr, budget_gpu) = hw.shared_buffer(256);
        let dst = hw.default_uav(4096);
        // SAFETY: writing test dispatch arguments and zeroing hits/budget.
        unsafe {
            hits_ptr.cast::<u32>().write_volatile(0);
            for k in 0..3 {
                budget_ptr.cast::<u32>().add(k).write_volatile(0);
            }
            let w = src_ptr.cast::<u32>();
            // Batch 1 (4 scattered single dispatches): 3 + 5 + 7 + 0 (overflow > GROUPS) = 15 groups.
            for (i, g) in [
                [3u32, 1, 1],
                [5, 1, 1],
                [7, 1, 1],
                [2048, 2048, 1], // > GROUPS => clamped to (0,0,0)
            ]
            .iter()
            .enumerate()
            {
                let base = i * 4;
                w.add(base).write_volatile(g[0]);
                w.add(base + 1).write_volatile(g[1]);
                w.add(base + 2).write_volatile(g[2]);
            }
            // Mid-list scalar clamp at offset 64: 1 dispatch of 11 groups.
            w.add(16).write_volatile(11);
            w.add(17).write_volatile(1);
            w.add(18).write_volatile(1);
            // Batch 2 (2 scattered dispatches at offsets 128, 144): 4 + 6 = 10 groups.
            w.add(32).write_volatile(4);
            w.add(33).write_volatile(1);
            w.add(34).write_volatile(1);
            w.add(36).write_volatile(6);
            w.add(37).write_volatile(1);
            w.add(38).write_volatile(1);
        }

        let (scratch1, scratch1_ptr, scratch1_gpu) = hw.upload_buffer(16 << 10);
        let (scratch2, scratch2_ptr, scratch2_gpu) = hw.upload_buffer(16 << 10);
        let mut gs = GatherState::default();
        // SAFETY: recording gather -> mid-list flush -> scalar -> gather -> ExecuteIndirect.
        unsafe {
            hw.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
        for i in 0..4u64 {
            assert!(gs.enqueue((&src, i * 16), None, (&dst, i * 16), (2, 12, 1), 0));
        }
        // Mid-list flush BEFORE a scalar clamp (budget_barrier_after = true).
        let s1 = gs
            .flush_on(
                &hw.device,
                hw.gather,
                Some(hw.clamp),
                &hw.list,
                budget_gpu,
                &budget,
                true,
                |_| Some((scratch1_gpu, scratch1_ptr)),
            )
            .unwrap();
        assert_eq!(s1.items, 4);
        assert_eq!(s1.barriers, 1);

        // Intervening scalar clamp at dst offset 64.
        record_scalar_into_uav(
            hw.clamp,
            &hw.list,
            &dst,
            64,
            budget_gpu,
            &budget,
            (&src, 64),
            None,
            (2, 12, 1),
            0,
            true,
        );

        // Batch 2 after scalar clamp.
        assert!(gs.enqueue((&src, 128), None, (&dst, 128), (2, 12, 1), 0));
        assert!(gs.enqueue((&src, 144), None, (&dst, 144), (2, 12, 1), 0));
        let s2 = gs
            .flush_on(
                &hw.device,
                hw.gather,
                Some(hw.clamp),
                &hw.list,
                budget_gpu,
                &budget,
                false,
                |_| Some((scratch2_gpu, scratch2_ptr)),
            )
            .unwrap();
        assert_eq!(s2.items, 2);

        let sig = hw.signature(2, 12);
        // SAFETY: transition clamped scratch to INDIRECT_ARGUMENT and execute all 7 dispatches.
        unsafe {
            hw.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
            hw.list.SetComputeRootSignature(&hw.worker.root);
            hw.list.SetPipelineState(&hw.worker.pso);
            hw.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            for off in [0u64, 16, 32, 48, 64, 128, 144] {
                hw.list.ExecuteIndirect(&sig, 1, &dst, off, None, 0);
            }
            hw.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        hw.submit_and_wait();
        drop((scratch1, scratch2));

        // Expected total worker hits: (3 + 5 + 7 + 0) + 11 + (4 + 6) = 36.
        // Expected budget: 36 * 64 = 2304.
        // SAFETY: reading completed buffers after fence.
        let (got_hits, got_budget) = unsafe {
            (
                hits_ptr.cast::<u32>().read_volatile(),
                budget_ptr.cast::<u32>().read_volatile(),
            )
        };
        assert_eq!(got_hits, 36, "real GPU ExecuteIndirect hit count mismatch");
        assert_eq!(got_budget, 36 * 64, "combined gather+scalar budget mismatch");
        gs.reset();
    }

    /// End-to-end production `HostList::bound_before` -> `clamp_end` ->
    /// `Host::execute` test on the real hardware GPU, verifying both
    /// `VICTORY_INDIRECT_GATHER=1` and `VICTORY_INDIRECT_GATHER=0` produce
    /// identical clamped output and reset cleanly across multiple frames.
    #[test]
    fn hw_gather_hostlist_bound_before_end_to_end() {
        let require_hw = std::env::var_os("VICTORY_REQUIRE_HW_GPU").is_some_and(|v| v != "0");
        let Some(host) = crate::host::host() else {
            assert!(!require_hw, "VICTORY_REQUIRE_HW_GPU=1 set, but no host GPU");
            return;
        };
        let (src, src_ptr) = host.shared_buffer(4096).expect("shared buffer");
        // SAFETY: writing 4 scattered indirect draw records + 1 wide record (64 records).
        unsafe {
            let w = src_ptr.cast::<u32>();
            for i in 0..4usize {
                let base = i * 8;
                w.add(base).write_volatile(120 + i as u32 * 10);
                w.add(base + 1).write_volatile(2);
                w.add(base + 2).write_volatile(i as u32);
                w.add(base + 3).write_volatile(77 + i as u32);
            }
            // Record 4 (overflow): vertices > 1<<24 -> clamped to 1<<24, instances clamped to 4.
            w.add(32).write_volatile(0xFFFF_FFFF);
            w.add(33).write_volatile(0xFFFF_FFFF);
            w.add(34).write_volatile(9);
            w.add(35).write_volatile(13);
        }

        let clamps = host.new_list().expect("host list");
        let mut outs = Vec::new();
        for i in 0..5u64 {
            let out = bound_before(&clamps, (&src, i * 32), None, (0, 16, 1))
                .expect("bound_before must succeed");
            outs.push(out);
        }
        clamps.clamp_end();
        clamps.close();
        host.execute(&[&clamps]);

        // Read back the clamped chunk buffer through a second copy list.
        let copy_list = host.new_list().expect("copy list");
        let rb = buffer(
            &host.device,
            256,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: `outs[0].0` is in INDIRECT_ARGUMENT after `clamp_end`; copy to readback.
        unsafe {
            copy_list.list.ResourceBarrier(&[transition(
                &outs[0].0,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            copy_list.list.CopyBufferRegion(&rb, 0, &outs[0].0, 0, 256);
            copy_list.list.ResourceBarrier(&[transition(
                &outs[0].0,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
        copy_list.close();
        host.execute(&[&copy_list]);

        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: 256 };
        // SAFETY: reading completed readback buffer.
        let words = unsafe {
            rb.Map(0, Some(&range), Some(&mut data)).unwrap();
            let v = std::slice::from_raw_parts(data.cast::<u32>(), 64).to_vec();
            rb.Unmap(0, None);
            v
        };
        for (i, (_, at)) in outs.iter().enumerate() {
            let base = (*at / 4) as usize;
            if i < 4 {
                assert_eq!(
                    &words[base..base + 4],
                    &[120 + i as u32 * 10, 2, i as u32, 77 + i as u32]
                );
            } else {
                assert_eq!(&words[base..base + 4], &[1u32 << 24, 4, 9, 13]);
            }
        }
        // Reset and reuse `clamps` for a second frame to verify clean reset.
        clamps.reset();
        let out2 = bound_before(&clamps, (&src, 0), None, (0, 16, 1)).expect("second frame bound_before");
        assert_eq!(out2.1, 0);
        clamps.clamp_end();
        clamps.close();
        host.execute(&[&clamps]);
    }

    /// Regression test for F5 (allocation-failure scalar fallback) and F2
    /// (consecutive mid-list gather flushes accumulating exact `budget[0]`):
    /// forces `alloc_scratch` to return `None` on Batch 1 (including single
    /// records and a multi-record counted draw with active count < max), then
    /// runs Batch 2 via normal gather-clamp with `budget_barrier_after = true`,
    /// and verifies bit-exact output and total budget on the real GPU.
    #[test]
    fn hw_gather_alloc_failure_falls_back_to_scalar_bit_exact() {
        let Some(mut hw) = HwGatherHarness::new() else {
            return;
        };
        let (src, src_ptr, _) = hw.shared_buffer(4096);
        let (cnt, cnt_ptr, _) = hw.shared_buffer(64);
        let (budget, budget_ptr, budget_gpu) = hw.shared_buffer(256);
        let dst = hw.default_uav(1024);
        let (sentinel, sentinel_ptr, _) = hw.upload_buffer(1024);
        // SAFETY: initializing test source, count, sentinel, and budget buffers.
        unsafe {
            std::slice::from_raw_parts_mut(sentinel_ptr.cast::<u32>(), 256).fill(0x5A5A_A5A5);
            cnt_ptr.cast::<u32>().write_volatile(2); // count=2 out of max=4
            for k in 0..3 {
                budget_ptr.cast::<u32>().add(k).write_volatile(0);
            }
            let w = src_ptr.cast::<u32>();
            // Call 0 (offset 0, DRAW stride 16, records=1): 100 verts * 3 insts = cost 300.
            w.add(0).write_volatile(100);
            w.add(1).write_volatile(3);
            w.add(2).write_volatile(7);
            w.add(3).write_volatile(9);
            // Call 1 (offset 32, INDEXED stride 20, max=4, count=2):
            //   rec 0: 50*2 = 100, rec 1: 60*2 = 120; rec 2 & 3 hold garbage (untouched).
            for r in 0..4usize {
                let base = 8 + r * 5;
                if r < 2 {
                    w.add(base).write_volatile(50 + r as u32 * 10);
                    w.add(base + 1).write_volatile(2);
                    w.add(base + 2).write_volatile(r as u32);
                    w.add(base + 3).write_volatile(10 + r as u32);
                    w.add(base + 4).write_volatile(20 + r as u32);
                } else {
                    w.add(base).write_volatile(0xFFFF_FFFF);
                    w.add(base + 1).write_volatile(0xFFFF_FFFF);
                    w.add(base + 2).write_volatile(0xDEAD_0000);
                    w.add(base + 3).write_volatile(0);
                    w.add(base + 4).write_volatile(0);
                }
            }
            // Call 2 (offset 256, DRAW stride 16, records=1): 200 verts * 4 insts = cost 800.
            w.add(64).write_volatile(200);
            w.add(65).write_volatile(4);
            w.add(66).write_volatile(11);
            w.add(67).write_volatile(13);

            hw.list.CopyBufferRegion(&dst, 0, &sentinel, 0, 1024);
            hw.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }

        let mut gs = GatherState::default();
        assert!(gs.enqueue((&src, 0), None, (&dst, 0), (0, 16, 1), 0));
        assert!(gs.enqueue((&src, 32), Some((&cnt, 0)), (&dst, 32), (1, 20, 4), 0));
        // Force scratch allocation failure -> must fall back to scalar clamps, NOT drop!
        let fb_stats = gs
            .flush_on(
                &hw.device,
                hw.gather,
                Some(hw.clamp),
                &hw.list,
                budget_gpu,
                &budget,
                true,
                |_| None,
            )
            .expect("scalar fallback on alloc failure must succeed");
        assert_eq!(fb_stats.items, 5);
        assert_eq!(fb_stats.groups, 0);
        assert_eq!(fb_stats.dispatches, 2);

        // Second batch flushed normally via gather-clamp after the scalar fallback.
        let (scratch, scratch_ptr, scratch_gpu) = hw.upload_buffer(16 << 10);
        assert!(gs.enqueue((&src, 256), None, (&dst, 256), (0, 16, 1), 0));
        let g_stats = gs
            .flush_on(
                &hw.device,
                hw.gather,
                Some(hw.clamp),
                &hw.list,
                budget_gpu,
                &budget,
                false,
                |_| Some((scratch_gpu, scratch_ptr)),
            )
            .expect("second gather flush must succeed");
        assert_eq!(g_stats.dispatches, 1);

        // SAFETY: transition dst back to COMMON before readback.
        unsafe {
            hw.list.ResourceBarrier(&[transition(
                &dst,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
        hw.submit_and_wait();
        drop(scratch);

        let words = hw.read_default_words(&dst, 1024);
        assert_eq!(&words[0..4], &[100, 3, 7, 9]);
        assert_eq!(&words[8..13], &[50, 2, 0, 10, 20]);
        assert_eq!(&words[13..18], &[60, 2, 1, 11, 21]);
        // Inactive tail slots (rec 2, 3 >= count 2) must stay untouched sentinel!
        assert!(
            words[18..28].iter().all(|&w| w == 0x5A5A_A5A5),
            "scalar fallback overwrote inactive counted tail slots"
        );
        assert_eq!(&words[64..68], &[200, 4, 11, 13]);
        // Total budget = 300 + 100 + 120 + 800 = 1320.
        // SAFETY: reading completed budget word after fence.
        let total_budget = unsafe { budget_ptr.cast::<u32>().read_volatile() };
        assert_eq!(total_budget, 1320, "budget lost update across fallback + gather flush");
        gs.reset();
    }

    /// Real-GPU benchmark comparing BEFORE (scalar Mode 7: 256 per-record
    /// dispatches + 256 budget UAV barriers) vs AFTER (gather-clamp batching:
    /// single-pass 1-buffer, two-pass 16-buffer, two-pass 256-distinct-buffer)
    /// on 256 scattered indirect records.
    #[test]
    #[ignore = "hardware benchmark: run with --ignored --nocapture"]
    fn hw_bench_scattered_indirect_gather_before_vs_after() {
        let Some(mut hw) = HwGatherHarness::new() else {
            eprintln!("no hardware adapter; skipped");
            return;
        };
        const RECORDS: usize = 256;
        const STRIDE: u32 = 20;
        const SLOT_ALIGN: u64 = 32;
        const ITERS: usize = 15;
        let total_bytes = (RECORDS as u64 * SLOT_ALIGN).next_multiple_of(256);

        let dst = hw.default_uav(total_bytes);
        let (budget, budget_ptr, budget_gpu) = hw.shared_buffer(256);
        let (scratch_res, scratch_ptr, scratch_gpu) = hw.upload_buffer(128 << 10);

        eprintln!(
            "Scattered indirect gather-clamp benchmark on {} ({RECORDS} scattered records, {ITERS} iterations, VICTORY_INDIRECT_GATHER={})",
            hw.card,
            u8::from(gather_enabled())
        );

        for (label, num_bufs) in [
            ("1 shared arg buffer (256 scattered offsets)", 1usize),
            ("16 arg buffers (interleaved)", 16usize),
            ("256 distinct arg buffers", 256usize),
        ] {
            let mut src_bufs = Vec::with_capacity(num_bufs);
            for _ in 0..num_bufs {
                src_bufs.push(hw.shared_buffer(16 << 10));
            }
            for i in 0..RECORDS {
                let b = i % num_bufs;
                let off = (i / num_bufs) * 32;
                // SAFETY: writing 20 bytes inside mapped 16 KiB buffer.
                unsafe {
                    let p = src_bufs[b].1.add(off).cast::<u32>();
                    p.add(0).write_volatile(300 + (i as u32 % 64));
                    p.add(1).write_volatile(1);
                    p.add(2).write_volatile(0);
                    p.add(3).write_volatile(0);
                    p.add(4).write_volatile(i as u32);
                }
            }

            let mut scalar_cpu_us = Vec::with_capacity(ITERS);
            let mut scalar_gpu_ms = Vec::with_capacity(ITERS);
            for iter in 0..=ITERS {
                // SAFETY: zeroing budget and timing scalar Mode 7 path.
                unsafe {
                    for k in 0..3 {
                        budget_ptr.cast::<u32>().add(k).write_volatile(0);
                    }
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    hw.list.EndQuery(&hw.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                }
                let t0 = std::time::Instant::now();
                for i in 0..RECORDS {
                    let b = i % num_bufs;
                    let src_off = ((i / num_bufs) * 32) as u64;
                    record_scalar_into_uav(
                        hw.clamp,
                        &hw.list,
                        &dst,
                        (i as u64) * SLOT_ALIGN,
                        budget_gpu,
                        &budget,
                        (&src_bufs[b].0, src_off),
                        None,
                        (1, STRIDE, 1),
                        0,
                        true,
                    );
                }
                let rec_us = t0.elapsed().as_secs_f64() * 1e6;
                // SAFETY: resolve GPU timestamps and return dst to COMMON.
                unsafe {
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    )]);
                    hw.list.EndQuery(&hw.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    hw.list.ResolveQueryData(
                        &hw.stamps,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        2,
                        &hw.stamp_readback,
                        0,
                    );
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                hw.submit_and_wait();
                if iter > 0 {
                    scalar_cpu_us.push(rec_us);
                    scalar_gpu_ms.push(hw.read_timestamp_ms());
                }
            }

            let mut gather_state = GatherState::default();
            let mut gather_cpu_us = Vec::with_capacity(ITERS);
            let mut gather_gpu_ms = Vec::with_capacity(ITERS);
            let mut last_stats = GatherFlushStats::default();
            for iter in 0..=ITERS {
                // SAFETY: zeroing budget and timing gather-clamp path.
                unsafe {
                    for k in 0..3 {
                        budget_ptr.cast::<u32>().add(k).write_volatile(0);
                    }
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_COMMON,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )]);
                    hw.list.EndQuery(&hw.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                }
                let t0 = std::time::Instant::now();
                for i in 0..RECORDS {
                    let b = i % num_bufs;
                    let src_off = ((i / num_bufs) * 32) as u64;
                    assert!(gather_state.enqueue(
                        (&src_bufs[b].0, src_off),
                        None,
                        (&dst, (i as u64) * SLOT_ALIGN),
                        (1, STRIDE, 1),
                        0,
                    ));
                }
                last_stats = gather_state
                    .flush_on(
                        &hw.device,
                        hw.gather,
                        Some(hw.clamp),
                        &hw.list,
                        budget_gpu,
                        &budget,
                        false,
                        |len| (len <= (128 << 10)).then_some((scratch_gpu, scratch_ptr)),
                    )
                    .unwrap();
                let rec_us = t0.elapsed().as_secs_f64() * 1e6;
                // SAFETY: transition to INDIRECT_ARGUMENT, resolve GPU timestamps, return to COMMON.
                unsafe {
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    )]);
                    hw.list.EndQuery(&hw.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    hw.list.ResolveQueryData(
                        &hw.stamps,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        2,
                        &hw.stamp_readback,
                        0,
                    );
                    hw.list.ResourceBarrier(&[transition(
                        &dst,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                        D3D12_RESOURCE_STATE_COMMON,
                    )]);
                }
                hw.submit_and_wait();
                gather_state.reset();
                if iter > 0 {
                    gather_cpu_us.push(rec_us);
                    gather_gpu_ms.push(hw.read_timestamp_ms());
                }
            }

            let median = |v: &mut Vec<f64>| -> f64 {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let s_cpu = median(&mut scalar_cpu_us);
            let s_gpu = median(&mut scalar_gpu_ms);
            let g_cpu = median(&mut gather_cpu_us);
            let g_gpu = median(&mut gather_gpu_ms);
            eprintln!(
                "  [{label}]\n    BEFORE (scalar Mode 7): dispatches={RECORDS}, uav_barriers={RECORDS}, cpu_rec={s_cpu:.1} us, gpu={s_gpu:.4} ms\n    AFTER  (gather-clamp) : dispatches={}, uav_barriers={}, groups={}, two_pass={}, cpu_rec={g_cpu:.1} us ({:.2}x), gpu={g_gpu:.4} ms ({:.2}x)",
                last_stats.dispatches,
                last_stats.barriers,
                last_stats.groups,
                last_stats.two_pass,
                s_cpu / g_cpu.max(1e-6),
                s_gpu / g_gpu.max(1e-6),
            );
            assert!(g_gpu.is_finite() && s_gpu.is_finite());
        }
        drop(scratch_res);
    }
}
