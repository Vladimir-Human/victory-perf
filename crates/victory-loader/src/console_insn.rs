//! Instructions the console's CPU (Zen 2) has and an Intel PC's lacks, emulated: the title's
//! code raises STATUS_ILLEGAL_INSTRUCTION on them (the title zeroes memory with `clzero` loops
//! and spins on `monitorx`/`mwaitx`), and the vectored handler steps over or patches them here.
//!
//! Ported from XCL (github.com/ivall/xcl, GPL-2.0-only): `insn.rs` (decoder, as
//! `console_insn_decode.rs`) and `eh.rs` `emulate_console_only`.

#[path = "console_insn_decode.rs"]
pub(crate) mod decode;

use decode::{ClzeroLoop, ConsoleOnly, LoopCmpOrder};
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT, FlushInstructionCache};
use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const CONTINUE_EXECUTION: i32 = -1;

/// Mask of status bits in `EFlags` updated by `cmp`/`sub`/`add`: CF, PF, AF, ZF, SF, OF.
pub(crate) const CMP_FLAGS: u32 = 0x8D5;

/// Whether what Windows said about memory is remembered per thread (the regions it found
/// writable, the code bytes it found readable) and not asked again at every emulated
/// instruction. On by default; `VICTORY_CLZERO_ASK=1` asks every time.
fn remembers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLZERO_ASK").is_none_or(|v| v != "1"))
}

/// Whether `monitorx`/`mwaitx` (`0F 01 FA`/`FB`) are atomically patched in place to
/// multi-byte NOPs (`0F 1F FA`/`FB`) on first fault. On by default; `VICTORY_PATCH_WAIT=0`
/// disables patching and steps over each wait instruction per exception instead.
fn patch_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PATCH_WAIT").is_none_or(|v| v != "0"))
}

/// Serializes `VirtualProtect` + in-place code patching across threads so two threads faulting
/// on instructions in the same page never race `VirtualProtect` protection save/restore.
static PATCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Answers a remembered region gives before Windows is asked about it again: one freed or
/// protected meanwhile is forgotten then.
const ANSWERS: u32 = 4096;

thread_local! {
    /// The regions Windows last said are committed and writable (start, end), newest first,
    /// and the answers they gave since they were emptied.
    static WRITABLE: Cell<([(usize, usize); 4], u32)> = const { Cell::new(([(0, 0); 4], 0)) };
    /// Faulting addresses and the code bytes found readable there, newest first.
    static CODE: Cell<[(usize, usize); 8]> = const { Cell::new([(0, 0); 8]) };
    /// Per-thread count of emulated instructions (for race-free assertions in parallel tests).
    #[cfg(test)]
    static THREAD_EMULATED: Cell<u64> = const { Cell::new(0) };
}

/// Instructions emulated, the times Windows was asked about memory for them, and the bytes
/// `clzero` loops zeroed, since the start.
static EMULATED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);

#[cfg(test)]
pub(crate) fn emulated_count() -> u64 {
    THREAD_EMULATED.with(Cell::get)
}

/// The general-purpose register numbered `n` in ModRM/REX encoding (`0..=15`).
pub(crate) fn gpr(ctx: &mut CONTEXT, n: u8) -> &mut u64 {
    match n & 15 {
        0 => &mut ctx.Rax,
        1 => &mut ctx.Rcx,
        2 => &mut ctx.Rdx,
        3 => &mut ctx.Rbx,
        4 => &mut ctx.Rsp,
        5 => &mut ctx.Rbp,
        6 => &mut ctx.Rsi,
        7 => &mut ctx.Rdi,
        8 => &mut ctx.R8,
        9 => &mut ctx.R9,
        10 => &mut ctx.R10,
        11 => &mut ctx.R11,
        12 => &mut ctx.R12,
        13 => &mut ctx.R13,
        14 => &mut ctx.R14,
        _ => &mut ctx.R15,
    }
}

fn read_gpr(ctx: &CONTEXT, n: u8) -> u64 {
    match n & 15 {
        0 => ctx.Rax,
        1 => ctx.Rcx,
        2 => ctx.Rdx,
        3 => ctx.Rbx,
        4 => ctx.Rsp,
        5 => ctx.Rbp,
        6 => ctx.Rsi,
        7 => ctx.Rdi,
        8 => ctx.R8,
        9 => ctx.R9,
        10 => ctx.R10,
        11 => ctx.R11,
        12 => ctx.R12,
        13 => ctx.R13,
        14 => ctx.R14,
        _ => ctx.R15,
    }
}

/// Status flags (`CF`, `PF`, `AF`, `ZF`, `SF`, `OF`) left by `cmp`/`sub a, b`.
pub(crate) fn cmp_flags(a: u64, b: u64) -> u32 {
    sub_flags(a, b, true)
}

fn sub_flags(a: u64, b: u64, is_64bit: bool) -> u32 {
    let (a, b, sign_shift) = if is_64bit {
        (a, b, 63)
    } else {
        (a as u32 as u64, b as u32 as u64, 31)
    };
    let mask = if is_64bit { u64::MAX } else { u32::MAX as u64 };
    let r = a.wrapping_sub(b) & mask;
    u32::from(a < b)
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | (((a ^ b ^ r) >> 4) as u32 & 1) << 4
        | u32::from(r == 0) << 6
        | (((r >> sign_shift) as u32) & 1) << 7
        | ((((a ^ b) & (a ^ r)) >> sign_shift) as u32 & 1) << 11
}

fn add_flags(a: u64, b: u64, is_64bit: bool) -> u32 {
    let (a, b, sign_shift) = if is_64bit {
        (a, b, 63)
    } else {
        (a as u32 as u64, b as u32 as u64, 31)
    };
    let mask = if is_64bit { u64::MAX } else { u32::MAX as u64 };
    let r = a.wrapping_add(b) & mask;
    u32::from(r < a)
        | u32::from((r as u8).count_ones().is_multiple_of(2)) << 2
        | (((a ^ b ^ r) >> 4) as u32 & 1) << 4
        | u32::from(r == 0) << 6
        | (((r >> sign_shift) as u32) & 1) << 7
        | ((((a ^ r) & (b ^ r)) >> sign_shift) as u32 & 1) << 11
}

fn readable_len(va: usize, max: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery};
    let mut got = 0usize;
    while got < max {
        let cur = va.wrapping_add(got);
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let found = unsafe { VirtualQuery(cur as *const _, &mut m, size_of_val(&m)) } != 0;
        ASKED.fetch_add(1, Ordering::Relaxed);
        if !found || m.State != MEM_COMMIT || m.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 {
            break;
        }
        let avail = (m.BaseAddress as usize).saturating_add(m.RegionSize).saturating_sub(cur);
        if avail == 0 {
            break;
        }
        got = (got + avail).min(max);
    }
    got
}

fn is_committed_readonly(va: usize) -> bool {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_NOACCESS, PAGE_READONLY, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Ordering::Relaxed);
    found && m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0 && (m.Protect & 0xFF) == PAGE_READONLY
}

fn writable(va: usize, bytes: usize) -> bool {
    let Some(end) = va.checked_add(bytes) else {
        return false;
    };
    writable_to(va, end) == Some(end)
}

/// Where the committed, writable memory that `va` lies in ends, at most `end`; None if `va`
/// is not in such memory. A region found is remembered ([`remembers`]).
fn writable_to(va: usize, end: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, PAGE_READWRITE,
        PAGE_WRITECOPY, VirtualQuery,
    };
    let remember = remembers();
    if remember {
        let (regions, answers) = WRITABLE.with(Cell::get);
        if let Some(&(_, to)) = regions.iter().find(|&&(from, to)| from <= va && va < to) {
            let next = if answers >= ANSWERS { ([(0, 0); 4], 0) } else { (regions, answers + 1) };
            WRITABLE.with(|w| w.set(next));
            return Some(to.min(end));
        }
    }
    // SAFETY: plain data filled in by the query.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Ordering::Relaxed);
    let prot_ok = matches!(m.Protect & 0xFF, PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY);
    if !found || m.State != MEM_COMMIT || m.Protect & (PAGE_NOACCESS | PAGE_GUARD) != 0 || !prot_ok {
        return None;
    }
    let (from, to) = (m.BaseAddress as usize, m.BaseAddress as usize + m.RegionSize);
    if remember {
        WRITABLE.with(|w| {
            let (mut regions, answers) = w.get();
            regions.rotate_right(1);
            regions[0] = (from, to);
            w.set((regions, answers));
        });
    }
    Some(to.min(end))
}

/// Atomically rewrites the 1-byte opcode at `rip + op_off` from `expected` (`0x01` or `0xAE`)
/// to `0x1F` (`0F 1F ...` multi-byte NOP), serialized by `PATCH_LOCK` so concurrent faults on
/// the same page never race `VirtualProtect`.
///
/// # Safety
/// `rip..rip + len` is readable code and `op_off < len`.
unsafe fn patch_opcode_byte(rip: usize, len: usize, op_off: usize, expected: u8) -> bool {
    let op_addr = rip + op_off;
    let _guard = PATCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: `op_addr` lies within `[rip, rip + len)` checked readable by the caller.
    let byte = unsafe { &*(op_addr as *const AtomicU8) };
    match byte.load(Ordering::Acquire) {
        0x1F => return true,
        b if b == expected => {}
        _ => return false,
    }
    let mut old = 0u32;
    // SAFETY: temporarily makes the 1-byte opcode writable under `PATCH_LOCK`, restores `old` below.
    if unsafe { VirtualProtect(op_addr as *const _, 1, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
        return false;
    }
    let _ = byte.compare_exchange(expected, 0x1F, Ordering::SeqCst, Ordering::Acquire);
    // SAFETY: `rip..rip + len` is the instruction just patched; `old` is its prior protection.
    unsafe {
        FlushInstructionCache(GetCurrentProcess(), rip as *const _, len);
        VirtualProtect(op_addr as *const _, 1, old, &mut old);
    }
    true
}

/// Computes `(iterations, final_state)` for an accelerated `clzero` loop, or `None` when the
/// runtime entry state should fall back to stepping a single `clzero` instruction.
struct LoopOutcome {
    iters: u64,
    final_rax: u64,
    final_aux: Option<(u8, u64)>,
    final_eflags: u32,
}

fn plan_clzero_loop(ctx: &CONTEXT, lp: ClzeroLoop) -> Option<LoopOutcome> {
    const MAX_BYTES: u64 = 2 << 30;
    let start = ctx.Rax;
    match lp {
        ClzeroLoop::PtrCmp { ptr_reg, end_reg, ptr_lhs, order } => {
            if ptr_reg != 0 && read_gpr(ctx, ptr_reg) != start {
                return None;
            }
            let end = read_gpr(ctx, end_reg);
            let iters = match order {
                LoopCmpOrder::Below if start < end && end - start < MAX_BYTES => (end - start).div_ceil(64),
                LoopCmpOrder::BelowOrEqual if start <= end && end - start < MAX_BYTES => (end - start) / 64 + 1,
                LoopCmpOrder::NotEqual if start < end && (end - start).is_multiple_of(64) && end - start < MAX_BYTES => (end - start) / 64,
                _ => return None,
            };
            let final_ptr = start.checked_add(iters.checked_mul(64)?)?;
            let flags = if ptr_lhs { cmp_flags(final_ptr, end) } else { cmp_flags(end, final_ptr) };
            Some(LoopOutcome {
                iters,
                final_rax: final_ptr,
                final_aux: (ptr_reg != 0).then_some((ptr_reg, final_ptr)),
                final_eflags: (ctx.EFlags & !CMP_FLAGS) | flags,
            })
        }
        ClzeroLoop::CountDown { cnt_reg, step, is_64bit, is_dec, dec_cf, above, is_add_neg } => {
            let raw_cnt = read_gpr(ctx, cnt_reg);
            let cnt = if is_64bit { raw_cnt } else { raw_cnt as u32 as u64 };
            let step_u64 = u64::from(step);
            let iters = match (step, above) {
                (1, _) if (1..MAX_BYTES / 64).contains(&cnt) => cnt,
                (64, false) if cnt >= 64 && cnt.is_multiple_of(64) && cnt < MAX_BYTES => cnt / 64,
                (64, true) if (1..MAX_BYTES).contains(&cnt) => cnt.div_ceil(64),
                _ => return None,
            };
            let last_before = cnt - (iters - 1) * step_u64;
            let raw_after = last_before.wrapping_sub(step_u64);
            let final_cnt = if is_64bit { raw_after } else { raw_after as u32 as u64 };
            let final_rax = start.checked_add(iters.checked_mul(64)?)?;
            let flags = if is_dec {
                let f = sub_flags(last_before, step_u64, is_64bit);
                match dec_cf {
                    Some(cf) => (ctx.EFlags & !CMP_FLAGS) | (f & !1) | u32::from(cf),
                    None => (ctx.EFlags & !(CMP_FLAGS & !1)) | (f & !1),
                }
            } else if is_add_neg {
                let f = add_flags(last_before, (-(step as i64)) as u64, is_64bit);
                (ctx.EFlags & !CMP_FLAGS) | f
            } else {
                let f = sub_flags(last_before, step_u64, is_64bit);
                (ctx.EFlags & !CMP_FLAGS) | f
            };
            Some(LoopOutcome { iters, final_rax, final_aux: Some((cnt_reg, final_cnt)), final_eflags: flags })
        }
        ClzeroLoop::CountUp { cnt_reg, end_reg, step, is_64bit, cnt_lhs, order } => {
            let mask = if is_64bit { u64::MAX } else { u32::MAX as u64 };
            let cnt = read_gpr(ctx, cnt_reg) & mask;
            let end = read_gpr(ctx, end_reg) & mask;
            let step_u64 = u64::from(step);
            let max_iters = MAX_BYTES / 64;
            let iters = match order {
                LoopCmpOrder::Below if cnt < end => (end - cnt).div_ceil(step_u64),
                LoopCmpOrder::BelowOrEqual if cnt <= end => (end - cnt) / step_u64 + 1,
                LoopCmpOrder::NotEqual if cnt < end && (end - cnt).is_multiple_of(step_u64) => (end - cnt) / step_u64,
                _ => return None,
            };
            if iters == 0 || iters >= max_iters {
                return None;
            }
            let delta = iters.checked_mul(step_u64)?;
            let final_cnt = cnt.checked_add(delta)?;
            if !is_64bit && final_cnt > u32::MAX as u64 {
                return None;
            }
            let final_rax = start.checked_add(iters.checked_mul(64)?)?;
            let flags = if cnt_lhs {
                sub_flags(final_cnt, end, is_64bit)
            } else {
                sub_flags(end, final_cnt, is_64bit)
            };
            Some(LoopOutcome { iters, final_rax, final_aux: Some((cnt_reg, final_cnt)), final_eflags: (ctx.EFlags & !CMP_FLAGS) | flags })
        }
    }
}

fn advance_partial_clzero_loop(ctx: &mut CONTEXT, lp: ClzeroLoop, done_iters: u64) {
    let delta_bytes = done_iters * 64;
    ctx.Rax = ctx.Rax.wrapping_add(delta_bytes);
    match lp {
        ClzeroLoop::PtrCmp { ptr_reg, .. } if ptr_reg != 0 => {
            *gpr(ctx, ptr_reg) = ctx.Rax;
        }
        ClzeroLoop::CountDown { cnt_reg, step, is_64bit, .. } => {
            let r = gpr(ctx, cnt_reg);
            let next = r.wrapping_sub(done_iters * u64::from(step));
            *r = if is_64bit { next } else { next as u32 as u64 };
        }
        ClzeroLoop::CountUp { cnt_reg, step, is_64bit, .. } => {
            let r = gpr(ctx, cnt_reg);
            let next = r.wrapping_add(done_iters * u64::from(step));
            *r = if is_64bit { next } else { next as u32 as u64 };
        }
        _ => {}
    }
}

/// Executes a single `clzero` cache-line zero at `RAX` (or `EAX` if `0x67` prefix is present).
/// In live VEH mode (`allow_readonly_fault == true`), committed `PAGE_READONLY` memory is written
/// directly so an upstream write-watch VEH (`victory-gpu::watch`) can open the guard page, or a
/// genuine read-only fault can surface as `STATUS_ACCESS_VIOLATION`.
fn exec_single_clzero(ctx: &CONTEXT, code: &[u8], clzero_end: usize, allow_readonly_fault: bool) -> bool {
    let a32 = code[..clzero_end - 3].contains(&0x67);
    let line = (if a32 { ctx.Rax & 0xFFFF_FFFF } else { ctx.Rax } as usize) & !63;
    if !writable(line, 64) && !(allow_readonly_fault && is_committed_readonly(line)) {
        return false; // unmapped, guard, or non-faulting unit test on RO memory
    }
    // SAFETY: `line` is writable or committed `PAGE_READONLY` (watch-guarded or raising write AV).
    unsafe { std::ptr::write_bytes(line as *mut u8, 0, 64) };
    true
}

/// Emulates a decoded instruction without modifying executable code in place (used by unit
/// tests that pass stack byte slices to `crash::emulate_amd_only`).
///
/// # Safety
/// `ctx` is a valid thread context and `code` holds readable instruction bytes at `ctx.Rip`.
#[cfg(test)]
pub(crate) unsafe fn emulate_slice(ctx: &mut CONTEXT, code: &[u8]) -> Option<usize> {
    let (what, len) = decode::decode(code)?;
    // SAFETY: forwarded from caller.
    let res = unsafe { apply_emulation(ctx, code, what, len, false) };
    if res == CONTINUE_EXECUTION { Some(len) } else { None }
}

/// An illegal instruction at `ctx.Rip`: a console-only one is emulated and stepped over (or
/// patched in place, for `clwb`/`monitorx`/`mwaitx`) returning `CONTINUE_EXECUTION`; anything else
/// returns `0` so crash reporting and title handlers see it.
///
/// # Safety
/// `ctx` is the faulting thread's context, from the exception being dispatched.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    // The same few instructions fault again and again: their readable length is asked once.
    let known = if remembers() { CODE.with(Cell::get) } else { [(0, 0); 8] };
    let cached = known.iter().find(|&&(at, n)| at == rip && n > 0).map(|&(_, n)| n);
    let n = match cached {
        Some(n) => n,
        None => readable_len(rip, 32),
    };
    if n == 0 {
        return 0;
    }
    // SAFETY: `n` bytes at `rip` are readable.
    let code = unsafe { std::slice::from_raw_parts(rip as *const u8, n) };
    let Some((what, len)) = decode::decode(code) else { return 0 };
    if cached.is_none() && remembers() {
        let mut known = known;
        known.rotate_right(1);
        known[0] = (rip, n);
        CODE.with(|c| c.set(known));
    }
    if what == ConsoleOnly::PatchedInsn {
        // Another thread patched `0F 01 FA/FB` or `66 0F AE /6` -> `0F 1F ...` right after we
        // faulted: retry at the same `Rip` so the CPU executes the multi-byte NOP.
        return CONTINUE_EXECUTION;
    }
    static SEEN: std::sync::Mutex<Vec<std::mem::Discriminant<ConsoleOnly>>> = std::sync::Mutex::new(Vec::new());
    if what != ConsoleOnly::Ud2
        && let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&std::mem::discriminant(&what))
    {
        seen.push(std::mem::discriminant(&what));
        tracing::warn!(instruction = ?what, at = format_args!("{rip:#x}"), "a console CPU instruction this CPU lacks: emulated (logged once)");
    }
    // SAFETY: `code` is `[rip..rip + n]` and `len <= n`.
    let r = unsafe { apply_emulation(ctx, code, what, len, true) };
    if r == CONTINUE_EXECUTION {
        #[cfg(test)]
        THREAD_EMULATED.with(|c| c.set(c.get() + 1));
        let emulated = EMULATED.fetch_add(1, Ordering::Relaxed) + 1;
        if emulated >= 1 << 16 && emulated.is_power_of_two() {
            tracing::info!(
                emulated,
                asked = ASKED.load(Ordering::Relaxed),
                zeroed_mb = ZEROED.load(Ordering::Relaxed) >> 20,
                "console CPU instructions emulated so far, the times Windows was asked about memory for them, and what clzero loops zeroed"
            );
        }
    }
    r
}

/// # Safety
/// `code` holds at least `len` readable bytes at `ctx.Rip`.
unsafe fn apply_emulation(ctx: &mut CONTEXT, code: &[u8], what: ConsoleOnly, len: usize, allow_patch: bool) -> i32 {
    let rip = ctx.Rip as usize;
    let mut step_len = len;
    match what {
        ConsoleOnly::Ud2 => return 0,
        ConsoleOnly::PatchedInsn => return CONTINUE_EXECUTION,
        ConsoleOnly::Clwb => {
            if allow_patch && let Some(op_off) = decode::opcode_byte_offset(&code[..len]) {
                // SAFETY: `rip..rip + len` is readable code for `clwb` (`0F AE`).
                let _ = unsafe { patch_opcode_byte(rip, len, op_off, 0xAE) };
            }
        }
        ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {
            if allow_patch && patch_wait() {
                // SAFETY: `rip..rip + len` is readable code ending in `0F 01 FA/FB`.
                let _ = unsafe { patch_opcode_byte(rip, len, len - 2, 0x01) };
            } else if what == ConsoleOnly::Mwaitx {
                std::hint::spin_loop();
            }
        }
        ConsoleOnly::Clzero => {
            if !exec_single_clzero(ctx, code, len, allow_patch) {
                return 0;
            }
        }
        ConsoleOnly::ClzeroLoop(lp) => {
            if let Some(outcome) = plan_clzero_loop(ctx, lp) {
                let zero_start = (ctx.Rax as usize) & !63;
                let Some(last_line) = (ctx.Rax as usize).checked_add((outcome.iters as usize - 1) * 64) else {
                    return 0;
                };
                let Some(zero_end) = (last_line & !63).checked_add(64) else {
                    return 0;
                };
                let mut at = zero_start;
                while at < zero_end {
                    let Some(to) = writable_to(at, zero_end) else { break };
                    // SAFETY: committed and writable from `at` to `to`.
                    unsafe { std::ptr::write_bytes(at as *mut u8, 0, to - at) };
                    at = to;
                }
                if at == zero_start {
                    if allow_patch && is_committed_readonly(zero_start) {
                        // Trigger write on `PAGE_READONLY` so an upstream write-watch VEH can
                        // open the guard page (or a real read-only fault surfaces as AV).
                        // SAFETY: committed `PAGE_READONLY` 64-byte cache line.
                        unsafe { std::ptr::write_bytes(zero_start as *mut u8, 0, 64) };
                        ZEROED.fetch_add(64, Ordering::Relaxed);
                        if outcome.iters > 1 {
                            advance_partial_clzero_loop(ctx, lp, 1);
                            return CONTINUE_EXECUTION;
                        }
                    } else {
                        return 0;
                    }
                } else {
                    ZEROED.fetch_add((at - zero_start) as u64, Ordering::Relaxed);
                    if at < zero_end {
                        let done_iters = ((at - zero_start) / 64) as u64;
                        if done_iters == 0 {
                            return 0;
                        }
                        advance_partial_clzero_loop(ctx, lp, done_iters);
                        return CONTINUE_EXECUTION;
                    }
                }
                ctx.Rax = outcome.final_rax;
                if let Some((reg, val)) = outcome.final_aux {
                    *gpr(ctx, reg) = val;
                }
                ctx.EFlags = outcome.final_eflags;
            } else {
                // Entry state doesn't take >0 full accelerated passes: emulate just `clzero` (3 bytes).
                let clzero_len = code[..len].windows(3).position(|w| w == [0x0F, 0x01, 0xFC]).map_or(3, |p| p + 3);
                if !exec_single_clzero(ctx, code, clzero_len, allow_patch) {
                    return 0;
                }
                step_len = clzero_len;
            }
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            // SAFETY: a valid out pointer.
            unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            // SAFETY: no preconditions.
            let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u64;
            *gpr(ctx, r) = cpu;
        }
        ConsoleOnly::Extrq { dst, src, len, idx } => {
            // SAFETY: the context's XMM registers (CONTEXT_FLOATING_POINT is captured for exceptions).
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let (len, idx) = match src {
                Some(s) => {
                    let c = x[s as usize & 15].Low;
                    (c as u8, (c >> 8) as u8)
                }
                None => (len, idx),
            };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::extrq(d.Low, len, idx);
            d.High = 0;
        }
        ConsoleOnly::Insertq { dst, src, reg, len, idx } => {
            // SAFETY: as above.
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let s = x[src as usize & 15];
            let (len, idx) = if reg { (s.High as u8, (s.High >> 8) as u8) } else { (len, idx) };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::insertq(d.Low, s.Low, len, idx);
            d.High = 0;
        }
    }
    ctx.Rip += step_len as u64;
    CONTINUE_EXECUTION
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_READONLY, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        VirtualQuery,
    };

    struct ExecBuf {
        ptr: *mut u8,
        size: usize,
    }

    // SAFETY: `ExecBuf` owns an immutable-to-readers executable allocation released only on `Drop`.
    unsafe impl Send for ExecBuf {}
    // SAFETY: see `Send`.
    unsafe impl Sync for ExecBuf {}

    impl ExecBuf {
        fn new(bytes: &[u8]) -> Self {
            // SAFETY: reserves and commits a fresh RW page, copies code, then marks PAGE_EXECUTE_READ.
            unsafe {
                let ptr = VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
                assert!(!ptr.is_null());
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
                let mut old = 0u32;
                assert_ne!(VirtualProtect(ptr.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
                FlushInstructionCache(GetCurrentProcess(), ptr.cast(), bytes.len());
                Self { ptr, size: bytes.len() }
            }
        }

        fn protect(&self) -> u32 {
            // SAFETY: queries our allocated page.
            unsafe {
                let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(self.ptr.cast(), &mut m, size_of_val(&m)), 0);
                m.Protect
            }
        }

        fn bytes(&self) -> &[u8] {
            // SAFETY: valid for `self.size` bytes while `ExecBuf` lives.
            unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
        }
    }

    impl Drop for ExecBuf {
        fn drop(&mut self) {
            // SAFETY: `self.ptr` was allocated with `VirtualAlloc`.
            unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
        }
    }

    fn cpu_has_monitorx() -> bool {
        let max_ext = std::arch::x86_64::__cpuid(0x8000_0000).eax;
        max_ext >= 0x8000_0001 && (std::arch::x86_64::__cpuid(0x8000_0001).ecx & (1 << 29)) != 0
    }

    fn cpu_has_clzero() -> bool {
        let max_ext = std::arch::x86_64::__cpuid(0x8000_0000).eax;
        max_ext >= 0x8000_0008 && (std::arch::x86_64::__cpuid(0x8000_0008).ebx & 1) != 0
    }

    #[test]
    fn wait_loop_patches_on_first_fault_and_takes_zero_exceptions_after() {
        crate::report_crashes();
        // loop(rcx = iters):
        //   monitorx          (0F 01 FA)
        //   mwaitx            (0F 01 FB)
        //   dec rcx           (48 FF C9)
        //   jne loop          (75 F5)
        //   ret               (C3)
        let code = [0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x48, 0xFF, 0xC9, 0x75, 0xF5, 0xC3];
        let buf = ExecBuf::new(&code);
        assert_eq!(buf.protect(), PAGE_EXECUTE_READ);
        // SAFETY: `buf.ptr` holds a valid `extern "win64" fn(u64)` in PAGE_EXECUTE_READ memory.
        let func: unsafe extern "win64" fn(u64) = unsafe { std::mem::transmute(buf.ptr) };

        let before = emulated_count();
        if !cpu_has_monitorx() {
            // Run 10,000 iterations: 1st iteration faults once on monitorx + once on mwaitx (2 faults),
            // patches both to `0F 1F FA` / `0F 1F FB`, and iterations 2..10_000 take 0 faults.
            unsafe { func(10_000) };
            assert_eq!(emulated_count() - before, 2, "expected exactly 2 faults (monitorx + mwaitx) across 10,000 iterations");
        } else {
            // CPU supports MONITORX natively: exercise `emulate` directly to patch, then execute NOPs natively.
            // SAFETY: valid CONTEXTs pointing at the two wait instructions in `buf`.
            unsafe {
                let mut ctx: CONTEXT = std::mem::zeroed();
                ctx.Rip = buf.ptr as u64;
                assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                ctx.Rip = buf.ptr.add(3) as u64;
                assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            }
        }
        let after_first_run = emulated_count();
        assert_eq!(&buf.bytes()[0..6], &[0x0F, 0x1F, 0xFA, 0x0F, 0x1F, 0xFB]);
        assert_eq!(buf.protect(), PAGE_EXECUTE_READ);

        // Second 10,000 iterations on the patched code: strictly zero faults!
        unsafe { func(10_000) };
        assert_eq!(emulated_count(), after_first_run, "patched wait loop must take zero exceptions");
    }

    #[test]
    fn prefixed_monitorx_and_mwaitx_patch_and_execute_as_nops() {
        crate::report_crashes();
        // Prefix combinations before 0F 01 FA / 0F 01 FB, preserving rax/rbx/rcx/rdx/flags:
        //   67 0F 01 FA ; 48 0F 01 FB ; 66 0F 01 FA ; F2 0F 01 FB ; F3 0F 01 FA ; ret
        let code = [
            0x67, 0x0F, 0x01, 0xFA, //
            0x48, 0x0F, 0x01, 0xFB, //
            0x66, 0x0F, 0x01, 0xFA, //
            0xF2, 0x0F, 0x01, 0xFB, //
            0xF3, 0x0F, 0x01, 0xFA, //
            0xC3,
        ];
        let buf = ExecBuf::new(&code);
        // SAFETY: valid `extern "win64" fn()` returning cleanly after NOPs.
        let func: unsafe extern "win64" fn() = unsafe { std::mem::transmute(buf.ptr) };
        let before = emulated_count();
        if !cpu_has_monitorx() {
            unsafe { func() };
            assert_eq!(emulated_count() - before, 5);
        } else {
            // SAFETY: steps through all 5 prefixed instructions via `emulate` directly.
            unsafe {
                let mut ctx: CONTEXT = std::mem::zeroed();
                for off in [0usize, 4, 8, 12, 16] {
                    ctx.Rip = buf.ptr.add(off) as u64;
                    assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                }
            }
        }
        let after_first = emulated_count();
        assert_eq!(
            buf.bytes(),
            &[
                0x67, 0x0F, 0x1F, 0xFA, //
                0x48, 0x0F, 0x1F, 0xFB, //
                0x66, 0x0F, 0x1F, 0xFA, //
                0xF2, 0x0F, 0x1F, 0xFB, //
                0xF3, 0x0F, 0x1F, 0xFA, //
                0xC3,
            ]
        );
        for _ in 0..1000 {
            unsafe { func() };
        }
        assert_eq!(emulated_count(), after_first);
    }

    #[test]
    fn concurrent_threads_patching_same_wait_site_do_not_race() {
        crate::report_crashes();
        let code = [0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x48, 0xFF, 0xC9, 0x75, 0xF5, 0xC3];
        let buf = Arc::new(ExecBuf::new(&code));
        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let native = !cpu_has_monitorx();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let buf = Arc::clone(&buf);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    if native {
                        // SAFETY: valid `extern "win64" fn(u64)` in PAGE_EXECUTE_READ memory.
                        let func: unsafe extern "win64" fn(u64) = unsafe { std::mem::transmute(buf.ptr) };
                        unsafe { func(10_000) };
                    } else {
                        // SAFETY: valid CONTEXTs racing `PATCH_LOCK` on `buf`.
                        unsafe {
                            let mut ctx: CONTEXT = std::mem::zeroed();
                            ctx.Rip = buf.ptr as u64;
                            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                            ctx.Rip = buf.ptr.add(3) as u64;
                            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                        }
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(&buf.bytes()[0..6], &[0x0F, 0x1F, 0xFA, 0x0F, 0x1F, 0xFB]);
        assert_eq!(buf.protect(), PAGE_EXECUTE_READ);
    }

    #[test]
    fn already_patched_wait_retries_at_same_rip() {
        let code = [0x0F, 0x1F, 0xFA, 0xC3];
        let buf = ExecBuf::new(&code);
        // SAFETY: zeroed CONTEXT with Rip pointing at `0F 1F FA`.
        unsafe {
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = buf.ptr as u64;
            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            assert_eq!(ctx.Rip, buf.ptr as u64, "must retry at same Rip");
        }
    }

    #[test]
    fn clzero_loop_variants_execute_and_set_exact_flags() {
        crate::report_crashes();
        // Test via real PAGE_EXECUTE_READ machine code + VEH on this CPU:
        // 1) Non-rax induction (15-byte): clzero; add r8, 0x40; mov rax, r8; cmp r8, r9; jb -15; ret
        //    rcx = start (moved to rax & r8), rdx = end (moved to r9), returns final rax.
        //    48 89 C8 (mov rax, rcx); 49 89 C8 (mov r8, rcx); 49 89 D1 (mov r9, rdx);
        //    0F 01 FC; 49 83 C0 40; 4C 89 C0; 4D 39 C8; 72 F1; C3
        let ptr_cmp_code = [
            0x48, 0x89, 0xC8, // mov rax, rcx
            0x49, 0x89, 0xC8, // mov r8, rcx
            0x49, 0x89, 0xD1, // mov r9, rdx
            0x0F, 0x01, 0xFC, // clzero
            0x49, 0x83, 0xC0, 0x40, // add r8, 0x40
            0x4C, 0x89, 0xC0, // mov rax, r8
            0x4D, 0x39, 0xC8, // cmp r8, r9
            0x72, 0xF1, // jb -15
            0xC3, // ret
        ];
        let buf1 = ExecBuf::new(&ptr_cmp_code);
        // SAFETY: `buf1.ptr` is `extern "win64" fn(*mut u8, *mut u8) -> usize`.
        let f1: unsafe extern "win64" fn(*mut u8, *mut u8) -> usize = unsafe { std::mem::transmute(buf1.ptr) };

        #[repr(align(64))]
        struct Arena([u8; 64 * 10]);
        let mut arena = Arena([0xA5; 64 * 10]);
        let start = unsafe { arena.0.as_mut_ptr().add(64) };
        let end = unsafe { arena.0.as_mut_ptr().add(64 * 8 + 16) }; // unaligned end -> 8 lines zeroed (1..9)
        let before = emulated_count();
        let ret = unsafe { f1(start, end) };
        // Entire 8-line loop ran in 1 exception on CPUs without CLZERO (0 on AMD)!
        assert_eq!(emulated_count() - before, u64::from(!cpu_has_clzero()));
        assert_eq!(ret, unsafe { arena.0.as_ptr().add(64 * 9) } as usize);
        assert!(arena.0[..64].iter().all(|&b| b == 0xA5));
        assert!(arena.0[64..64 * 9].iter().all(|&b| b == 0));
        assert!(arena.0[64 * 9..].iter().all(|&b| b == 0xA5));

        // 2) 32-bit countdown with `lea rax, [rax+0x40]` after `dec ecx`:
        //    mov rax, rcx; mov ecx, edx; clzero; dec ecx; lea rax, [rax+0x40]; jne -11; ret
        let cnt_down_code = [
            0x48, 0x89, 0xC8, // mov rax, rcx
            0x89, 0xD1, // mov ecx, edx
            0x0F, 0x01, 0xFC, // clzero
            0xFF, 0xC9, // dec ecx
            0x48, 0x8D, 0x40, 0x40, // lea rax, [rax+0x40]
            0x75, 0xF5, // jne -11
            0xC3, // ret
        ];
        let buf2 = ExecBuf::new(&cnt_down_code);
        let f2: unsafe extern "win64" fn(*mut u8, u32) -> usize = unsafe { std::mem::transmute(buf2.ptr) };
        arena.0.fill(0x5A);
        let before2 = emulated_count();
        let ret2 = unsafe { f2(arena.0.as_mut_ptr().add(128), 5) };
        assert_eq!(emulated_count() - before2, u64::from(!cpu_has_clzero()));
        assert_eq!(ret2, unsafe { arena.0.as_ptr().add(128 + 5 * 64) } as usize);
        assert!(arena.0[..128].iter().all(|&b| b == 0x5A));
        assert!(arena.0[128..128 + 5 * 64].iter().all(|&b| b == 0));
        assert!(arena.0[128 + 5 * 64..].iter().all(|&b| b == 0x5A));

        // 3) Check exact EFlags on `cmp end, rax; ja`:
        let code_ja = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC2, 0x77, 0xF4];
        let mut buf3 = [0xCCu8; 64 * 4];
        let mut ctx: CONTEXT = unsafe { std::mem::zeroed() };
        ctx.Rax = buf3.as_mut_ptr() as u64;
        ctx.Rdx = (buf3.as_mut_ptr() as u64) + 128;
        let stepped = unsafe { emulate_slice(&mut ctx, &code_ja) };
        assert_eq!(stepped, Some(12));
        // At loop exit, Rax == Rdx -> cmp rdx, rax leaves ZF=1, CF=0
        assert_eq!(ctx.Rax, ctx.Rdx);
        assert_eq!(ctx.EFlags & 0x41, 0x40);
    }

    #[test]
    fn clzero_loop_partially_writable_range_stops_at_unwritable_page() {
        // Two adjacent pages: page 0 is PAGE_READWRITE, page 1 is PAGE_READONLY.
        // SAFETY: allocates 2 pages with distinct protections and frees them before returning.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            std::ptr::write_bytes(base, 0xCC, 0x2000);
            let mut old = 0u32;
            assert_ne!(VirtualProtect(base.add(0x1000).cast(), 0x1000, PAGE_READONLY, &mut old), 0);

            let code = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4];
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = code.as_ptr() as u64;
            ctx.Rax = base as u64;
            ctx.Rdx = base.add(0x2000) as u64;

            // First call zeroes page 0, stops at page 1, keeps Rip at clzero.
            assert_eq!(emulate_slice(&mut ctx, &code), Some(12));
            assert_eq!(ctx.Rip, code.as_ptr() as u64, "partial loop keeps Rip at clzero");
            assert_eq!(ctx.Rax, base.add(0x1000) as u64);
            assert!(std::slice::from_raw_parts(base, 0x1000).iter().all(|&b| b == 0));
            assert!(std::slice::from_raw_parts(base.add(0x1000), 0x1000).iter().all(|&b| b == 0xCC));

            // Second call at unwritable page 1 must refuse (`None` / `0`), not write or advance!
            ctx.Rip = code.as_ptr() as u64;
            assert_eq!(emulate_slice(&mut ctx, &code), None);
            assert_eq!(ctx.Rax, base.add(0x1000) as u64);

            VirtualFree(base.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn opt_out_patch_wait_zero_leaves_code_unpatched() {
        const ITERS: u64 = 100;
        if std::env::var_os("VICTORY_OPTOUT_CHILD").is_some() {
            crate::report_crashes();
            let code = [0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x48, 0xFF, 0xC9, 0x75, 0xF5, 0xC3];
            let buf = ExecBuf::new(&code);
            let before = emulated_count();
            if !cpu_has_monitorx() {
                // SAFETY: valid `extern "win64" fn(u64)` in PAGE_EXECUTE_READ memory.
                let func: unsafe extern "win64" fn(u64) = unsafe { std::mem::transmute(buf.ptr) };
                unsafe { func(ITERS) };
            } else {
                // SAFETY: exercises `emulate` directly on CPUs that natively execute MONITORX.
                unsafe {
                    let mut ctx: CONTEXT = std::mem::zeroed();
                    for _ in 0..ITERS {
                        ctx.Rip = buf.ptr as u64;
                        assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                        ctx.Rip = buf.ptr.add(3) as u64;
                        assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                    }
                }
            }
            let faults = emulated_count() - before;
            assert_eq!(faults, ITERS * 2, "unpatched mode must emulate every iteration");
            assert_eq!(&buf.bytes()[0..6], &[0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB]);
            return;
        }
        let exe = std::env::current_exe().unwrap();
        let out = std::process::Command::new(exe)
            .args(["--exact", "console_insn::tests::opt_out_patch_wait_zero_leaves_code_unpatched", "--nocapture"])
            .env("VICTORY_OPTOUT_CHILD", "1")
            .env("VICTORY_PATCH_WAIT", "0")
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "opt-out child failed: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn countdown_dec_cf_semantics() {
        let mut buf = [0xAAu8; 64 * 3];
        // 1) add rax, 0x40; dec rcx; jne -> pre-loop CF=1 must become CF=0, ZF=1
        let add_dec = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0xFF, 0xC9, 0x75, 0xF4];
        let mut ctx: CONTEXT = unsafe { std::mem::zeroed() };
        ctx.Rax = buf.as_mut_ptr() as u64;
        ctx.Rcx = 2;
        ctx.EFlags = 1; // CF=1
        assert_eq!(unsafe { emulate_slice(&mut ctx, &add_dec) }, Some(12));
        assert_eq!(ctx.EFlags & 0x41, 0x40, "add +0x40; dec must clear CF and set ZF");

        // 2) sub rax, -0x40; dec rcx; jne -> pre-loop CF=0 must become CF=1, ZF=1
        let sub_dec = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xE8, 0xC0, 0x48, 0xFF, 0xC9, 0x75, 0xF4];
        ctx = unsafe { std::mem::zeroed() };
        ctx.Rax = buf.as_mut_ptr() as u64;
        ctx.Rcx = 2;
        ctx.EFlags = 0; // CF=0
        assert_eq!(unsafe { emulate_slice(&mut ctx, &sub_dec) }, Some(12));
        assert_eq!(ctx.EFlags & 0x41, 0x41, "sub -0x40; dec must set CF=1 and ZF=1");

        // 3) lea rax, [rax+0x40]; dec rcx; jne AND dec rcx; lea rax, [rax+0x40]; jne preserve CF
        let lea_dec = [0x0F, 0x01, 0xFC, 0x48, 0x8D, 0x40, 0x40, 0x48, 0xFF, 0xC9, 0x75, 0xF4];
        let dec_lea = [0x0F, 0x01, 0xFC, 0x48, 0xFF, 0xC9, 0x48, 0x8D, 0x40, 0x40, 0x75, 0xF4];
        for seq in [lea_dec, dec_lea] {
            for init_cf in [0u32, 1u32] {
                ctx = unsafe { std::mem::zeroed() };
                ctx.Rax = buf.as_mut_ptr() as u64;
                ctx.Rcx = 2;
                ctx.EFlags = init_cf;
                assert_eq!(unsafe { emulate_slice(&mut ctx, &seq) }, Some(12));
                assert_eq!(ctx.EFlags & 0x41, 0x40 | init_cf, "lea + dec must preserve CF={init_cf}");
            }
        }
    }

    #[test]
    fn straddling_wait_and_clzero_loop_across_distinct_page_protections() {
        crate::report_crashes();
        // Two adjacent executable pages with distinct protections: PAGE_EXECUTE_READ + PAGE_EXECUTE_READWRITE.
        // SAFETY: allocates 2 pages with VirtualAlloc and frees them at the end.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            let p0 = base;
            let p1 = base.add(0x1000);
            let mut old = 0u32;

            let prot_of = |addr: *const u8| {
                let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
                assert_ne!(VirtualQuery(addr.cast(), &mut m, size_of_val(&m)), 0);
                m.Protect
            };

            // Case A: `0F` at p0_end - 1, `01 FA C3` on p1
            assert_ne!(VirtualProtect(p0.cast(), 0x2000, PAGE_READWRITE, &mut old), 0);
            let site_a = p1.sub(1);
            std::ptr::copy_nonoverlapping([0x0F, 0x01, 0xFA, 0xC3].as_ptr(), site_a, 4);
            assert_ne!(VirtualProtect(p0.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
            assert_ne!(VirtualProtect(p1.cast(), 0x1000, PAGE_EXECUTE_READWRITE, &mut old), 0);

            let fa: unsafe extern "win64" fn() = std::mem::transmute(site_a);
            let e0 = emulated_count();
            if !cpu_has_monitorx() {
                fa();
                assert_eq!(emulated_count() - e0, 1);
            } else {
                let mut ctx: CONTEXT = std::mem::zeroed();
                ctx.Rip = site_a as u64;
                assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            }
            assert_eq!(std::slice::from_raw_parts(site_a, 4), &[0x0F, 0x1F, 0xFA, 0xC3]);
            assert_eq!(prot_of(p0), PAGE_EXECUTE_READ);
            assert_eq!(prot_of(p1), PAGE_EXECUTE_READWRITE);
            let e0_after = emulated_count();
            fa();
            assert_eq!(emulated_count(), e0_after);

            // Case B: `0F 01` at p0_end - 2, `FA C3` on p1 (patched byte `01` sits on p0!)
            assert_ne!(VirtualProtect(p0.cast(), 0x1000, PAGE_READWRITE, &mut old), 0);
            let site_b = p1.sub(2);
            std::ptr::copy_nonoverlapping([0x0F, 0x01, 0xFA, 0xC3].as_ptr(), site_b, 4);
            assert_ne!(VirtualProtect(p0.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
            FlushInstructionCache(GetCurrentProcess(), site_b.cast(), 4);

            let fb: unsafe extern "win64" fn() = std::mem::transmute(site_b);
            let e1 = emulated_count();
            if !cpu_has_monitorx() {
                fb();
                assert_eq!(emulated_count() - e1, 1);
            } else {
                let mut ctx: CONTEXT = std::mem::zeroed();
                ctx.Rip = site_b as u64;
                assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            }
            assert_eq!(std::slice::from_raw_parts(site_b, 4), &[0x0F, 0x1F, 0xFA, 0xC3]);
            assert_eq!(prot_of(p0), PAGE_EXECUTE_READ);
            assert_eq!(prot_of(p1), PAGE_EXECUTE_READWRITE);
            let e1_after = emulated_count();
            fb();
            assert_eq!(emulated_count(), e1_after);

            // Case C: 12-byte clzero loop + ret straddling p0 (RX) -> p1 (RWX)
            assert_ne!(VirtualProtect(p0.cast(), 0x1000, PAGE_READWRITE, &mut old), 0);
            // mov rax, rcx; clzero; add rax, 0x40; cmp rax, rdx; jb -12; ret
            let loop_bytes = [
                0x48, 0x89, 0xC8, // mov rax, rcx (3)
                0x0F, 0x01, 0xFC, // clzero (starts at p1 - 4!)
                0x48, 0x83, 0xC0, 0x40, // add rax, 0x40
                0x48, 0x39, 0xD0, // cmp rax, rdx
                0x72, 0xF4, // jb -12
                0xC3, // ret
            ];
            let entry_c = p1.sub(7); // clzero starts at p1 - 4, 4 bytes on p0, 8+1 on p1
            std::ptr::copy_nonoverlapping(loop_bytes.as_ptr(), entry_c, loop_bytes.len());
            assert_ne!(VirtualProtect(p0.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
            FlushInstructionCache(GetCurrentProcess(), entry_c.cast(), loop_bytes.len());

            #[repr(align(64))]
            struct Buf([u8; 64 * 4]);
            let mut target = Buf([0x77; 64 * 4]);
            let fc: unsafe extern "win64" fn(*mut u8, *mut u8) = std::mem::transmute(entry_c);
            let e2 = emulated_count();
            fc(target.0.as_mut_ptr(), target.0.as_mut_ptr().add(64 * 4));
            assert_eq!(emulated_count() - e2, u64::from(!cpu_has_clzero()), "straddling clzero loop must execute in 1 fault on non-CLZERO CPUs");
            assert!(target.0.iter().all(|&b| b == 0));

            // Also verify direct `emulate` decodes the straddling clzero loop across RX/RWX on any CPU.
            target.0.fill(0x55);
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = p1.sub(4) as u64;
            ctx.Rax = target.0.as_mut_ptr() as u64;
            ctx.Rdx = target.0.as_mut_ptr().add(64 * 4) as u64;
            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            assert!(target.0.iter().all(|&b| b == 0));

            VirtualFree(base.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn clzero_and_clzero_loop_open_guarded_page_readonly_via_upstream_veh() {
        use std::sync::atomic::AtomicUsize;
        use windows_sys::Win32::Foundation::EXCEPTION_ACCESS_VIOLATION;
        use windows_sys::Win32::System::Diagnostics::Debug::{
            AddVectoredExceptionHandler, EXCEPTION_POINTERS, RemoveVectoredExceptionHandler,
        };

        static WATCH_BASE: AtomicUsize = AtomicUsize::new(0);
        static WATCH_LEN: AtomicUsize = AtomicUsize::new(0);
        static WATCH_OPENED: AtomicUsize = AtomicUsize::new(0);

        unsafe extern "system" fn open_ro_veh(info: *mut EXCEPTION_POINTERS) -> i32 {
            // SAFETY: system passes valid exception pointers.
            let r = unsafe { &*(*info).ExceptionRecord };
            if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2 && r.ExceptionInformation[0] == 1 {
                let addr = r.ExceptionInformation[1];
                let base = WATCH_BASE.load(Ordering::Acquire);
                let len = WATCH_LEN.load(Ordering::Acquire);
                if base != 0 && addr >= base && addr < base + len {
                    let page = addr & !0xFFF;
                    let mut old = 0u32;
                    // SAFETY: test's own guarded page opened to PAGE_READWRITE.
                    if unsafe { VirtualProtect(page as *const _, 0x1000, PAGE_READWRITE, &mut old) } != 0 {
                        WATCH_OPENED.fetch_add(1, Ordering::Relaxed);
                        return CONTINUE_EXECUTION;
                    }
                }
            }
            0
        }

        crate::report_crashes();
        // SAFETY: allocates 2 pages for this test, registers scoped first-chance VEH, and cleans up.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            std::ptr::write_bytes(base, 0xEE, 0x2000);
            WATCH_BASE.store(base as usize, Ordering::Release);
            WATCH_LEN.store(0x2000, Ordering::Release);
            WATCH_OPENED.store(0, Ordering::Release);

            let veh = AddVectoredExceptionHandler(1, Some(open_ro_veh));
            assert!(!veh.is_null());

            let mut old = 0u32;
            // 1) Single `clzero` directly on guarded PAGE_READONLY page 0
            assert_ne!(VirtualProtect(base.cast(), 0x1000, PAGE_READONLY, &mut old), 0);
            std::arch::asm!(".byte 0x0f, 0x01, 0xfc", in("rax") base);
            assert_eq!(WATCH_OPENED.load(Ordering::Relaxed), 1);
            assert!(std::slice::from_raw_parts(base, 64).iter().all(|&b| b == 0));
            assert!(std::slice::from_raw_parts(base.add(64), 64).iter().all(|&b| b == 0xEE));

            // 2) ClzeroLoop starting on PAGE_READWRITE page 0 and crossing into guarded PAGE_READONLY page 1
            std::ptr::write_bytes(base, 0xDD, 0x2000);
            assert_ne!(VirtualProtect(base.add(0x1000).cast(), 0x1000, PAGE_READONLY, &mut old), 0);
            let loop_buf = ExecBuf::new(&[
                0x48, 0x89, 0xC8, // mov rax, rcx
                0x0F, 0x01, 0xFC, // clzero
                0x48, 0x83, 0xC0, 0x40, // add rax, 0x40
                0x48, 0x39, 0xD0, // cmp rax, rdx
                0x72, 0xF4, // jb -12
                0xC3, // ret
            ]);
            let f: unsafe extern "win64" fn(*mut u8, *mut u8) = std::mem::transmute(loop_buf.ptr);
            f(base, base.add(0x2000));
            assert_eq!(WATCH_OPENED.load(Ordering::Relaxed), 2);
            assert!(std::slice::from_raw_parts(base, 0x2000).iter().all(|&b| b == 0));

            // 3) ClzeroLoop starting directly on guarded PAGE_READONLY page 0
            std::ptr::write_bytes(base, 0xCC, 0x1000);
            assert_ne!(VirtualProtect(base.cast(), 0x1000, PAGE_READONLY, &mut old), 0);
            f(base, base.add(0x1000));
            assert_eq!(WATCH_OPENED.load(Ordering::Relaxed), 3);
            assert!(std::slice::from_raw_parts(base, 0x1000).iter().all(|&b| b == 0));

            RemoveVectoredExceptionHandler(veh);
            WATCH_BASE.store(0, Ordering::Release);
            VirtualFree(base.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn clwb_patches_atomically_with_null_pointer_nop_and_retries() {
        // 1) Executing `66 0F 1F 30` (patched `clwb [rax]`) with rax = 0 (NULL) must NOT fault!
        let patched_clwb = [0x66, 0x0F, 0x1F, 0x30, 0xC3];
        let buf = ExecBuf::new(&patched_clwb);
        // SAFETY: `buf.ptr` executes a 4-byte memory-operand NOP with rax = 0 and returns.
        unsafe {
            let f: unsafe extern "win64" fn() = std::mem::transmute(buf.ptr);
            std::arch::asm!("call {f}", f = in(reg) f, in("rax") 0usize, clobber_abi("win64"));
        }

        // 2) `emulate` on unpatched `66 0F AE 30` (clwb [rax]) atomically flips 0xAE -> 0x1F only.
        let clwb = [0x66, 0x0F, 0xAE, 0x30, 0xC3];
        let clwb_buf = ExecBuf::new(&clwb);
        // SAFETY: valid CONTEXT pointing at clwb_buf.
        unsafe {
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = clwb_buf.ptr as u64;
            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            assert_eq!(ctx.Rip, clwb_buf.ptr as u64 + 4);
            assert_eq!(clwb_buf.bytes(), &[0x66, 0x0F, 0x1F, 0x30, 0xC3]);
            assert_eq!(clwb_buf.protect(), PAGE_EXECUTE_READ);

            // 3) Already-patched `66 0F 1F 30` retries at same Rip.
            ctx.Rip = clwb_buf.ptr as u64;
            assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
            assert_eq!(ctx.Rip, clwb_buf.ptr as u64);
        }
    }

    #[test]
    fn clzero_handles_writecopy_guard_and_overflow() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_COPY, MapViewOfFile, PAGE_GUARD, UnmapViewOfFile};

        // 1) `PAGE_WRITECOPY` mapped section view must allow `clzero` and `ClzeroLoop`.
        // SAFETY: creates an anonymous 1-page section, maps a copy-on-write view, and unmaps/closes it.
        unsafe {
            let map = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, 0x1000, std::ptr::null());
            assert!(!map.is_null());
            let view = MapViewOfFile(map, FILE_MAP_COPY, 0, 0, 0x1000).Value.cast::<u8>();
            assert!(!view.is_null());
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            assert_ne!(VirtualQuery(view.cast(), &mut m, size_of_val(&m)), 0);
            assert_eq!(m.Protect & 0xFF, windows_sys::Win32::System::Memory::PAGE_WRITECOPY);

            let clzero = [0x0F, 0x01, 0xFC];
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = clzero.as_ptr() as u64;
            ctx.Rax = view as u64;
            assert_eq!(emulate_slice(&mut ctx, &clzero), Some(3));
            assert!(std::slice::from_raw_parts(view, 64).iter().all(|&b| b == 0));

            let clzero_loop = [0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4];
            ctx = std::mem::zeroed();
            ctx.Rip = clzero_loop.as_ptr() as u64;
            ctx.Rax = view.add(64) as u64;
            ctx.Rdx = view.add(256) as u64;
            assert_eq!(emulate_slice(&mut ctx, &clzero_loop), Some(12));
            assert!(std::slice::from_raw_parts(view, 256).iter().all(|&b| b == 0));

            UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view.cast() });
            CloseHandle(map);
        }

        // 2) `PAGE_READWRITE | PAGE_GUARD` must be refused and never cached in `WRITABLE`.
        // SAFETY: allocates 1 page with `PAGE_READWRITE | PAGE_GUARD` and frees it.
        unsafe {
            let page = VirtualAlloc(std::ptr::null(), 0x1000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE | PAGE_GUARD).cast::<u8>();
            assert!(!page.is_null());
            let clzero = [0x0F, 0x01, 0xFC];
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rip = clzero.as_ptr() as u64;
            ctx.Rax = page as u64;
            assert_eq!(emulate_slice(&mut ctx, &clzero), None);
            assert_eq!(emulate_slice(&mut ctx, &clzero), None, "guard page must not be cached in WRITABLE");
            VirtualFree(page.cast(), 0, MEM_RELEASE);
        }

        // 3) Overflow addresses near `u64::MAX` must fail cleanly without panicking.
        // SAFETY: zeroed CONTEXT; `writable` rejects overflow before reading/writing memory.
        unsafe {
            let clzero = [0x0F, 0x01, 0xFC];
            for bad_rax in [u64::MAX, u64::MAX - 31, u64::MAX - 63] {
                let mut ctx: CONTEXT = std::mem::zeroed();
                ctx.Rip = clzero.as_ptr() as u64;
                ctx.Rax = bad_rax;
                assert_eq!(emulate_slice(&mut ctx, &clzero), None);
            }
        }
    }

    #[test]
    #[ignore = "real CPU microbenchmark: run with `cargo test --release -p victory-loader -- --ignored --nocapture bench_wait_patch`"]
    fn bench_wait_patch_before_vs_after() {
        const ITERS: u64 = 50_000;
        if cpu_has_monitorx() {
            println!("BENCH_SKIPPED cpu_has_monitorx=true");
            return;
        }
        if let Ok(mode) = std::env::var("VICTORY_BENCH_CHILD") {
            crate::report_crashes();
            let code = [0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x48, 0xFF, 0xC9, 0x75, 0xF5, 0xC3];
            let buf = ExecBuf::new(&code);
            // SAFETY: valid `extern "win64" fn(u64)` in PAGE_EXECUTE_READ memory.
            let func: unsafe extern "win64" fn(u64) = unsafe { std::mem::transmute(buf.ptr) };
            let e0 = emulated_count();
            let t0 = std::time::Instant::now();
            unsafe { func(ITERS) };
            let elapsed = t0.elapsed();
            let faults = emulated_count() - e0;
            println!("BENCH_RESULT mode={mode} iters={ITERS} ns={} faults={faults}", elapsed.as_nanos());
            return;
        }

        let exe = std::env::current_exe().unwrap();
        let run_child = |mode: &str, patch_wait: &str| {
            let out = std::process::Command::new(&exe)
                .args(["--exact", "console_insn::tests::bench_wait_patch_before_vs_after", "--ignored", "--nocapture"])
                .env("VICTORY_BENCH_CHILD", mode)
                .env("VICTORY_PATCH_WAIT", patch_wait)
                .output()
                .unwrap();
            assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stderr));
            String::from_utf8_lossy(&out.stdout).into_owned()
        };

        let before = run_child("BEFORE_UNPATCHED", "0");
        let after = run_child("AFTER_PATCHED", "1");
        print!("{before}{after}");
    }
}
