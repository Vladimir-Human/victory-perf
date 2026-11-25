//! Instructions the console's CPU (Zen 2) has and a PC's may lack – a title raises
//! STATUS_ILLEGAL_INSTRUCTION on them on an Intel CPU before Ice Lake / Alder Lake (CLWB,
//! SHA) or on any Intel CPU (CLZERO, RDPRU, MONITORX / MWAITX and SSE4a's EXTRQ / INSERTQ,
//! which are AMD's). The exception handler (`crash`) emulates the ones decoded here and steps
//! over them (or patches wait hints in place).

/// Comparison sense in a `clzero` clearing loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCmpOrder {
    /// Loop continues while `lhs < rhs` (`jb` / `ja`).
    Below,
    /// Loop continues while `lhs <= rhs` (`jbe` / `jae`).
    BelowOrEqual,
    /// Loop continues while `lhs != rhs` (`jne`).
    NotEqual,
}

/// Recognized `clzero` clearing loop shape starting at `clzero`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClzeroLoop {
    /// Pointer register `ptr_reg` advances by +64 (`0x40`) each pass and is compared
    /// against `end_reg` (64-bit). When `ptr_reg != 0`, `rax` is updated from `ptr_reg`.
    PtrCmp {
        ptr_reg: u8,
        end_reg: u8,
        /// True when `cmp` has `ptr_reg` on the left (`cmp ptr, end`); false for `cmp end, ptr`.
        ptr_lhs: bool,
        order: LoopCmpOrder,
    },
    /// `rax` advances by +64 (`0x40`); counter `cnt_reg` (`!= 0`) decrements by `step`
    /// (`1` line or `64` bytes) and branches on `jne` (`above == false`) or `ja` (`above == true`).
    CountDown {
        cnt_reg: u8,
        step: u8,
        is_64bit: bool,
        /// `dec` leaves CF unchanged; `sub`/`add` updates CF.
        is_dec: bool,
        above: bool,
        /// True for `add cnt, -step`; false for `sub cnt, step`.
        is_add_neg: bool,
    },
    /// `rax` advances by +64 (`0x40`); index/offset counter `cnt_reg` (`!= 0`) increments by
    /// `step` (`1` or `64`) and is compared against `end_reg` (32-bit or 64-bit).
    CountUp {
        cnt_reg: u8,
        end_reg: u8,
        step: u8,
        is_64bit: bool,
        cnt_lhs: bool,
        order: LoopCmpOrder,
    },
}

/// A console-only instruction at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// `clzero` clearing loop: zeroes `[rax..end)` in one exception
    ClzeroLoop(ClzeroLoop),
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips or patches to `0F 1F FA/FB`
    Monitorx,
    Mwaitx,
    /// Already-patched `monitorx`/`mwaitx` (`[prefixes..., 0F, 1F, FA|FB]`) seen if another
    /// thread faulted right before the atomic byte patch landed: retry at the same `Rip`.
    PatchedWait,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

/// Skips zero or more x86 NOP instructions (`90`, `66 90`, `[66|seg|REX]* 0F 1F /r`) at the
/// start of `b`, returning the total byte count skipped.
fn skip_nops(b: &[u8]) -> usize {
    let mut off = 0;
    while off < b.len() {
        let rest = &b[off..];
        if rest[0] == 0x90 {
            off += 1;
            continue;
        }
        if rest.starts_with(&[0x66, 0x90]) {
            off += 2;
            continue;
        }
        let mut p = 0;
        while p < rest.len() && p < 4 && matches!(rest[p], 0x66 | 0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67) {
            p += 1;
        }
        if p < rest.len() && (0x40..=0x4F).contains(&rest[p]) {
            p += 1;
        }
        if rest.get(p..p + 2) == Some(&[0x0F, 0x1F])
            && let Some(ml) = modrm_len(&rest[p + 2..])
        {
            off += p + 2 + ml;
            continue;
        }
        break;
    }
    off
}

/// Decodes a 64-bit `+0x40` pointer advance (`add reg, 0x40`, `sub reg, -0x40`, or
/// `lea reg, [reg + 0x40]`). Returns `(reg, modifies_flags, len)`.
fn decode_ptr_step64(b: &[u8]) -> Option<(u8, bool, usize)> {
    let rex = *b.first()?;
    if rex == 0x48 && b.get(1..6) == Some(&[0x05, 0x40, 0x00, 0x00, 0x00]) {
        return Some((0, true, 6));
    }
    if rex == 0x48 && b.get(1..6) == Some(&[0x2D, 0xC0, 0xFF, 0xFF, 0xFF]) {
        return Some((0, true, 6));
    }
    if !(0x48..=0x4F).contains(&rex) {
        return None;
    }
    let op = *b.get(1)?;
    let modrm = *b.get(2)?;
    match op {
        0x83 if modrm >> 6 == 3 => {
            let reg = (modrm & 7) | ((rex & 1) << 3);
            let sub_op = (modrm >> 3) & 7;
            match (sub_op, *b.get(3)?) {
                (0, 0x40) | (5, 0xC0) => Some((reg, true, 4)),
                _ => None,
            }
        }
        0x81 if modrm >> 6 == 3 => {
            let reg = (modrm & 7) | ((rex & 1) << 3);
            let sub_op = (modrm >> 3) & 7;
            let imm = b.get(3..7)?;
            match sub_op {
                0 if imm == [0x40, 0x00, 0x00, 0x00] => Some((reg, true, 7)),
                5 if imm == [0xC0, 0xFF, 0xFF, 0xFF] => Some((reg, true, 7)),
                _ => None,
            }
        }
        0x8D if rex & 0x02 == 0 => {
            let md = modrm >> 6;
            let reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
            let rm = (modrm & 7) | ((rex & 1) << 3);
            if reg != rm || (md != 1 && md != 2) {
                return None;
            }
            let disp_at = if rm & 7 == 4 {
                if *b.get(3)? != 0x24 {
                    return None;
                }
                4
            } else {
                3
            };
            if md == 1 {
                (*b.get(disp_at)? == 0x40).then_some((reg, false, disp_at + 1))
            } else {
                (b.get(disp_at..disp_at + 4)? == [0x40, 0x00, 0x00, 0x00]).then_some((reg, false, disp_at + 4))
            }
        }
        _ => None,
    }
}

/// Decodes `mov rax, src` (64-bit, `src != 0`). Returns `(src_reg, len)`.
fn decode_mov_rax(b: &[u8]) -> Option<(u8, usize)> {
    let &[rex @ 0x48..=0x4F, op @ (0x89 | 0x8B), modrm] = b.get(..3)? else {
        return None;
    };
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    let rm = (modrm & 7) | ((rex & 1) << 3);
    match op {
        0x89 if rm == 0 && reg != 0 => Some((reg, 3)),
        0x8B if reg == 0 && rm != 0 => Some((rm, 3)),
        _ => None,
    }
}

/// Decodes register-register `cmp lhs, rhs`. Returns `(lhs, rhs, is_64bit, len)`.
fn decode_cmp_regs(b: &[u8]) -> Option<(u8, u8, bool, usize)> {
    let (rex, at) = match *b.first()? {
        r @ 0x40..=0x4F => (r, 1),
        _ => (0, 0),
    };
    let op = *b.get(at)?;
    if op != 0x39 && op != 0x3B {
        return None;
    }
    let modrm = *b.get(at + 1)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = ((modrm >> 3) & 7) | ((rex & 4) << 1);
    let rm = (modrm & 7) | ((rex & 1) << 3);
    let (lhs, rhs) = if op == 0x39 { (rm, reg) } else { (reg, rm) };
    Some((lhs, rhs, (rex & 0x08) != 0, at + 2))
}

struct CntStep {
    cnt_reg: u8,
    delta: i8,
    is_64bit: bool,
    is_dec: bool,
    is_add_neg: bool,
    len: usize,
}

fn decode_cnt_step(b: &[u8]) -> Option<CntStep> {
    let (rex, at) = match *b.first()? {
        r @ 0x40..=0x4F => (r, 1),
        _ => (0, 0),
    };
    let is_64bit = (rex & 0x08) != 0;
    let op = *b.get(at)?;
    let modrm = *b.get(at + 1)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let cnt_reg = (modrm & 7) | ((rex & 1) << 3);
    let sub_op = (modrm >> 3) & 7;
    match op {
        0xFF if sub_op == 0 || sub_op == 1 => Some(CntStep {
            cnt_reg,
            delta: if sub_op == 0 { 1 } else { -1 },
            is_64bit,
            is_dec: true,
            is_add_neg: false,
            len: at + 2,
        }),
        0x83 => {
            let imm = *b.get(at + 2)?;
            let (delta, is_add_neg) = match (sub_op, imm) {
                (0, 1) => (1, false),
                (0, 0xFF) => (-1, true),
                (0, 0x40) => (64, false),
                (0, 0xC0) => (-64, true),
                (5, 1) => (-1, false),
                (5, 0x40) => (-64, false),
                _ => return None,
            };
            Some(CntStep {
                cnt_reg,
                delta,
                is_64bit,
                is_dec: false,
                is_add_neg,
                len: at + 3,
            })
        }
        _ => None,
    }
}

/// Checks a short conditional jump `jcc rel8` jumping back to offset `0`.
fn decode_jcc_back(b: &[u8], pos: usize) -> Option<(u8, usize)> {
    let &[op @ 0x72..=0x77, rel] = b.get(..2)? else {
        return None;
    };
    let total = pos + 2;
    (i8::from_le_bytes([rel]) as isize == -(total as isize)).then_some((op, total))
}

fn cmp_order(lhs_is_var: bool, jcc: u8) -> Option<LoopCmpOrder> {
    match (lhs_is_var, jcc) {
        (true, 0x72) | (false, 0x77) => Some(LoopCmpOrder::Below),
        (true, 0x76) | (false, 0x73) => Some(LoopCmpOrder::BelowOrEqual),
        (_, 0x75) => Some(LoopCmpOrder::NotEqual),
        _ => None,
    }
}

fn match_cmp_loop_regs(lhs: u8, rhs: u8, var_reg: u8) -> Option<(u8, bool)> {
    if lhs == var_reg && rhs != var_reg {
        Some((rhs, true))
    } else if rhs == var_reg && lhs != var_reg {
        Some((lhs, false))
    } else {
        None
    }
}

fn decode_clzero_loop(code: &[u8], i: usize) -> Option<(ClzeroLoop, usize)> {
    if code[..i].iter().any(|b| matches!(b, 0x66 | 0x67 | 0xF2 | 0xF3)) {
        return None;
    }
    let mut pos = i + 3;
    pos += skip_nops(&code[pos..]);

    if let Some((ptr_reg, _, step_len)) = decode_ptr_step64(&code[pos..]) {
        pos += step_len;
        pos += skip_nops(&code[pos..]);

        if ptr_reg != 0 {
            let mut saw_mov = false;
            if let Some((src, ml)) = decode_mov_rax(&code[pos..])
                && src == ptr_reg
            {
                saw_mov = true;
                pos += ml;
                pos += skip_nops(&code[pos..]);
            }
            let (lhs, rhs, is_64bit, clen) = decode_cmp_regs(&code[pos..])?;
            if !is_64bit {
                return None;
            }
            let (end_reg, ptr_lhs) = match_cmp_loop_regs(lhs, rhs, ptr_reg)?;
            pos += clen;
            pos += skip_nops(&code[pos..]);
            if !saw_mov {
                let (src, ml) = decode_mov_rax(&code[pos..])?;
                if src != ptr_reg {
                    return None;
                }
                pos += ml;
                pos += skip_nops(&code[pos..]);
            }
            let (jcc, total) = decode_jcc_back(&code[pos..], pos)?;
            let order = cmp_order(ptr_lhs, jcc)?;
            return Some((ClzeroLoop::PtrCmp { ptr_reg, end_reg, ptr_lhs, order }, total));
        }

        // `ptr_reg == 0` (`rax` updated directly)
        if let Some((lhs, rhs, is_64bit, clen)) = decode_cmp_regs(&code[pos..]) {
            if !is_64bit {
                return None;
            }
            let (end_reg, ptr_lhs) = match_cmp_loop_regs(lhs, rhs, 0)?;
            pos += clen;
            pos += skip_nops(&code[pos..]);
            let (jcc, total) = decode_jcc_back(&code[pos..], pos)?;
            let order = cmp_order(ptr_lhs, jcc)?;
            return Some((ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg, ptr_lhs, order }, total));
        }

        let cnt = decode_cnt_step(&code[pos..])?;
        if cnt.cnt_reg == 0 {
            return None;
        }
        pos += cnt.len;
        pos += skip_nops(&code[pos..]);
        return finish_cnt_loop(code, pos, cnt, true);
    }

    // Counter step before `lea rax, [rax + 0x40]` (or before `add rax, 0x40` + `cmp`)
    let cnt = decode_cnt_step(&code[pos..])?;
    if cnt.cnt_reg == 0 {
        return None;
    }
    pos += cnt.len;
    pos += skip_nops(&code[pos..]);
    let (ptr_reg, ptr_modifies_flags, step_len) = decode_ptr_step64(&code[pos..])?;
    if ptr_reg != 0 {
        return None;
    }
    pos += step_len;
    pos += skip_nops(&code[pos..]);
    finish_cnt_loop(code, pos, cnt, !ptr_modifies_flags)
}

fn finish_cnt_loop(code: &[u8], mut pos: usize, cnt: CntStep, flags_live: bool) -> Option<(ClzeroLoop, usize)> {
    if cnt.delta < 0 {
        if !flags_live {
            return None;
        }
        let (jcc, total) = decode_jcc_back(&code[pos..], pos)?;
        let above = match jcc {
            0x75 => false,
            0x77 if !cnt.is_dec => true,
            _ => return None,
        };
        return Some((
            ClzeroLoop::CountDown {
                cnt_reg: cnt.cnt_reg,
                step: (-cnt.delta) as u8,
                is_64bit: cnt.is_64bit,
                is_dec: cnt.is_dec,
                above,
                is_add_neg: cnt.is_add_neg,
            },
            total,
        ));
    }

    let (lhs, rhs, cmp_64, clen) = decode_cmp_regs(&code[pos..])?;
    if cmp_64 != cnt.is_64bit {
        return None;
    }
    let (end_reg, cnt_lhs) = match_cmp_loop_regs(lhs, rhs, cnt.cnt_reg)?;
    if end_reg == 0 {
        return None;
    }
    pos += clen;
    pos += skip_nops(&code[pos..]);
    let (jcc, total) = decode_jcc_back(&code[pos..], pos)?;
    let order = cmp_order(cnt_lhs, jcc)?;
    Some((
        ClzeroLoop::CountUp {
            cnt_reg: cnt.cnt_reg,
            end_reg,
            step: cnt.delta as u8,
            is_64bit: cnt.is_64bit,
            cnt_lhs,
            order,
        },
        total,
    ))
}

/// The console-only instruction at the start of `code` and its length.
pub fn decode(code: &[u8]) -> Option<(ConsoleOnly, usize)> {
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            if let Some((lp, len)) = decode_clzero_loop(code, i) {
                Some((ConsoleOnly::ClzeroLoop(lp), len))
            } else {
                Some((ConsoleOnly::Clzero, i + 3))
            }
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        // Already-patched `monitorx`/`mwaitx` (`0F 01 FA/FB` -> `0F 1F FA/FB`).
        (0x1F, Some(0xFA | 0xFB)) => Some((ConsoleOnly::PatchedWait, i + 3)),
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // 66 0F AE /6 with a memory operand (without 66 it is xsaveopt)
        (0xAE, Some(m)) if p66 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // clzero loop against r10 (4C 39 D0 = cmp rax, r10)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 10, ptr_lhs: true, order: LoopCmpOrder::Below }),
                12,
            ))
        );
        // clzero loop against rdx (48 39 D0 = cmp rax, rdx)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: true, order: LoopCmpOrder::Below }),
                12,
            ))
        );
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x1F, 0xFA]), Some((ConsoleOnly::PatchedWait, 3)));
        assert_eq!(decode(&[0x0F, 0x1F, 0xFB]), Some((ConsoleOnly::PatchedWait, 3)));
        assert_eq!(decode(&[0x67, 0x48, 0x0F, 0x1F, 0xFA]), Some((ConsoleOnly::PatchedWait, 5)));
        assert_eq!(decode(&[0x0F, 0x1F, 0x00]), None);
        assert_eq!(decode(&[0xF3, 0x0F, 0x1E, 0xFA]), None);
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // SSE4a, as a retail title has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them: the last of F2 / F3 decides, 66 next to them does not;
        // 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }

    #[test]
    fn decodes_extended_clzero_loops_and_rejects_bad_shapes() {
        // cmp end, rax; ja (reversed operands, 0x39 and 0x3B)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC2, 0x77, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: false, order: LoopCmpOrder::Below }),
                12,
            ))
        );
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x3B, 0xD0, 0x77, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 10, ptr_lhs: false, order: LoopCmpOrder::Below }),
                12,
            ))
        );
        // 0x3B cmp rax, r10; jbe
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x49, 0x3B, 0xC2, 0x76, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 10, ptr_lhs: true, order: LoopCmpOrder::BelowOrEqual }),
                12,
            ))
        );
        // add rax, imm32(0x40) (48 05 40 00 00 00) and (48 81 C0 40 00 00 00)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x05, 0x40, 0, 0, 0, 0x48, 0x39, 0xD0, 0x72, 0xF2]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: true, order: LoopCmpOrder::Below }),
                14,
            ))
        );
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x81, 0xC0, 0x40, 0, 0, 0, 0x4C, 0x39, 0xD0, 0x72, 0xF1]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 10, ptr_lhs: true, order: LoopCmpOrder::Below }),
                15,
            ))
        );
        // lea rax, [rax + 0x40] (disp8 and disp32)
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x8D, 0x40, 0x40, 0x48, 0x39, 0xD0, 0x75, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: true, order: LoopCmpOrder::NotEqual }),
                12,
            ))
        );
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x8D, 0x80, 0x40, 0, 0, 0, 0x48, 0x39, 0xD0, 0x72, 0xF1]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: true, order: LoopCmpOrder::Below }),
                15,
            ))
        );
        // non-rax induction (15-byte): clzero; add r8, 0x40; mov rax, r8; cmp r8, r9; jb -15
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x49, 0x83, 0xC0, 0x40, 0x4C, 0x89, 0xC0, 0x4D, 0x39, 0xC8, 0x72, 0xF1]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 8, end_reg: 9, ptr_lhs: true, order: LoopCmpOrder::Below }),
                15,
            ))
        );
        // non-rax induction with mov after cmp: clzero; add rbx, 0x40; cmp rbx, rdi; mov rax, rbx; jb -15
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC3, 0x40, 0x48, 0x39, 0xFB, 0x48, 0x89, 0xD8, 0x72, 0xF1]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 3, end_reg: 7, ptr_lhs: true, order: LoopCmpOrder::Below }),
                15,
            ))
        );
        // 32-bit countdown: clzero; add rax, 0x40; sub ecx, 1; jne -12
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x83, 0xE9, 0x01, 0x75, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::CountDown { cnt_reg: 1, step: 1, is_64bit: false, is_dec: false, above: false, is_add_neg: false }),
                12,
            ))
        );
        // dec rcx after lea rax, [rax+0x40] (or before lea!): clzero; dec rcx; lea rax, [rax+0x40]; jne -12
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0xFF, 0xC9, 0x48, 0x8D, 0x40, 0x40, 0x75, 0xF4]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::CountDown { cnt_reg: 1, step: 1, is_64bit: true, is_dec: true, above: false, is_add_neg: false }),
                12,
            ))
        );
        // 64-byte countdown: clzero; add rax, 0x40; sub r8, 0x40; ja -13
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x49, 0x83, 0xE8, 0x40, 0x77, 0xF3]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::CountDown { cnt_reg: 8, step: 64, is_64bit: true, is_dec: false, above: true, is_add_neg: false }),
                13,
            ))
        );
        // 32-bit count-up index: clzero; add rax, 0x40; add r8d, 1; cmp r8d, r9d; jb -16
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x41, 0x83, 0xC0, 0x01, 0x45, 0x39, 0xC8, 0x72, 0xF0]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::CountUp { cnt_reg: 8, end_reg: 9, step: 1, is_64bit: false, cnt_lhs: true, order: LoopCmpOrder::Below }),
                16,
            ))
        );
        // 66 0F 1F 44 00 00 padding before jcc
        assert_eq!(
            decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00, 0x72, 0xEE]),
            Some((
                ConsoleOnly::ClzeroLoop(ClzeroLoop::PtrCmp { ptr_reg: 0, end_reg: 2, ptr_lhs: true, order: LoopCmpOrder::Below }),
                18,
            ))
        );

        // Negative controls: must fall back to single Clzero (len 3, or 4 with 0x67)
        // 1) Mismatched jb offset (-10 instead of -12)
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF6]), Some((ConsoleOnly::Clzero, 3)));
        // 2) Wrong immediate (0x20 instead of 0x40)
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x20, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        // 3) 0x67 prefix on clzero
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF3]), Some((ConsoleOnly::Clzero, 4)));
        // 4) Memory operand in cmp (mod != 3)
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0x11, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        // 5) cmp rax, rax (same register)
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        // 6) add rax, 0x40 after dec rcx clobbers flags before jne
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0xFF, 0xC9, 0x48, 0x83, 0xC0, 0x40, 0x75, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
        // 7) cmp rdx, rax; jb (opposite comparison direction)
        assert_eq!(decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC2, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 3)));
    }
}
