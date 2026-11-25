//! `VICTORY_REBAR` (opt-in, off by default): host copies of title memory in
//! the card's own memory that the CPU maps (Resizable BAR, D3D12's GPU
//! upload heap, `D3D12_HEAP_TYPE_GPU_UPLOAD`).
//!
//! The console's memory is one: what the CPU writes, the GPU reads where it
//! lies. Here a host copy of title memory is either in the PC's memory
//! (`Host::shared_buffer`: the CPU writes it at full speed, every shader
//! load crosses the bus) or, for big title buffers, in the card's memory
//! with a second side in the PC's memory and a copy between the two before
//! each submission (`crate::shadow::cards`). With a large CPU-visible
//! window on the card's memory the CPU can write the card's side itself:
//! no PC-side twin, no copy before the submission, shader loads at the
//! card's speed.
//!
//! What is placed there is what the CPU writes and shaders only read:
//! - `VICTORY_REBAR=1`: windows of title memory outside buffers (the
//!   title's own allocator: interface vertices, mesh streams, constants;
//!   run 91: shaders write none of them; run 104: 3.5 GB of the 5.4 GB
//!   the card shares with the PC).
//! - `VICTORY_REBAR=2`: also title buffers below the card-buffer size
//!   (`VICTORY_CARD_BUFFERS_MIN_MB`), which shaders may write. What they
//!   write is still found and copied back on the CPU, reading the card's
//!   memory over the bus (uncached: slow), so this level is for A/B tests.
//!
//! Never placed there, at any level: the buffers the title reads on the CPU
//! (its read-back ring, `crate::readback`). The CPU reads those, and a read
//! of CPU-visible card memory is uncached; they stay in cached PC memory
//! (write-back custom heap, as before), which is what a read-back heap is.
//!
//! Without the heap (Resizable BAR off: a 256 MB BAR; an older driver or
//! system; the software adapter) nothing changes: every copy is made as
//! before, and the start line says why.
//!
//! `VICTORY_REBAR_MB` sets the most bytes placed (default: a quarter of
//! the card's budget, at most 4 GiB); as for card buffers, a copy goes
//! there only while a fifth of the card's budget stays free.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// What a copy is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// A window of title memory outside buffers.
    Window,
    /// A title buffer's copy (one too small for the card-buffer path).
    Buffer,
}

/// Where a copy goes, and why not to the card's mapped memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Decision {
    Rebar,
    Off,
    Unsupported,
    /// The level does not place this kind.
    NotThisKind,
    /// The title reads it on the CPU.
    CpuRead,
    OverBudget,
    /// The card's memory is nearly full.
    CardFull,
}

/// What a decision is made from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Ask {
    pub level: u8,
    pub supported: bool,
    pub kind: Kind,
    pub size: u64,
    pub cpu_read: bool,
    /// Bytes already placed, and the most that may be.
    pub placed: u64,
    pub most: u64,
    /// The card's budget and use, if known.
    pub card: Option<(u64, u64)>,
}

/// Computes `budget * percent / 100` without intermediate truncation or u64 overflow.
fn card_limit_bytes(budget: u64, percent: u64) -> u64 {
    ((u128::from(budget) * u128::from(percent)) / 100) as u64
}

/// Maximum share of VRAM budget (%) up to which ReBAR copies may be placed
/// (`VICTORY_REBAR_VRAM_PERCENT`, default 80 — leaving a fifth of the card
/// free; clamped 50..=90).
fn vram_percent() -> u64 {
    static PERCENT: OnceLock<u64> = OnceLock::new();
    *PERCENT.get_or_init(|| {
        std::env::var("VICTORY_REBAR_VRAM_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(80, |p| p.clamp(50, 90))
    })
}

/// The placement rule (pure: tested below).
#[cfg(test)]
pub(crate) fn decide(ask: &Ask) -> Decision {
    decide_with_percent(ask, 80)
}

fn decide_with_percent(ask: &Ask, vram_pct: u64) -> Decision {
    if ask.level == 0 {
        return Decision::Off;
    }
    if !ask.supported {
        return Decision::Unsupported;
    }
    if ask.cpu_read {
        return Decision::CpuRead;
    }
    match ask.kind {
        Kind::Window => {}
        Kind::Buffer if ask.level >= 2 => {}
        Kind::Buffer => return Decision::NotThisKind,
    }
    if ask.placed.saturating_add(ask.size) > ask.most {
        return Decision::OverBudget;
    }
    if let Some((budget, used)) = ask.card
        && used.saturating_add(ask.size) > card_limit_bytes(budget, vram_pct)
    {
        return Decision::CardFull;
    }
    Decision::Rebar
}

/// `VICTORY_REBAR`: 0 (unset, or anything else), 1 or 2.
pub(crate) fn level_of(value: Option<&str>) -> u8 {
    match value.map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    }
}

/// The default most bytes: a quarter of the card's budget, at most 4 GiB
/// (256 MiB when the budget is unknown).
pub(crate) fn default_most(card_budget: Option<u64>) -> u64 {
    match card_budget {
        Some(b) => (b / 4).min(4 << 30),
        None => 256 << 20,
    }
}

pub(crate) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| level_of(std::env::var("VICTORY_REBAR").ok().as_deref()))
}

/// Whether the host's device has GPU upload heaps: set once at the host's
/// start ([`probe`]).
static SUPPORTED: OnceLock<bool> = OnceLock::new();
/// Local video memory budget captured at startup by [`probe`].
static PROBED_BUDGET: OnceLock<Option<u64>> = OnceLock::new();

pub(crate) fn supported() -> bool {
    SUPPORTED.get().copied().unwrap_or(false)
}

fn most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let env = std::env::var("VICTORY_REBAR_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok());
        match env {
            Some(mb) => mb.saturating_mul(1 << 20),
            None => {
                let budget = PROBED_BUDGET
                    .get()
                    .copied()
                    .flatten()
                    .or_else(|| crate::host::host().and_then(|h| h.video_memory()).map(|[b, ..]| b));
                default_most(budget)
            }
        }
    })
}

/// Bytes placed alive, copies placed alive, and copies that wanted to be
/// placed but were not (over the budget, card full, creation failed).
static PLACED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
/// Bytes reserved by [`reserve`] whose D3D12 resource creation has not yet
/// completed (and therefore may not yet appear in DXGI `CurrentUsage`).
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
/// The next multiple of 64 MiB of placed bytes at which a line is written.
static NEXT_LINE: AtomicU64 = AtomicU64::new(64 << 20);

/// Reserves bytes against the current count, so concurrent callers cannot
/// both spend the same remaining budget. Returns the new total.
fn reserve_bytes(placed: &AtomicU64, size: u64, most: u64) -> Option<u64> {
    if size == 0 {
        return None;
    }
    placed
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
            used.checked_add(size).filter(|&next| next <= most)
        })
        .ok()
        .map(|before| before + size)
}

/// Atomically reserves `size` bytes against `placed`, bounded simultaneously by
/// `most` and the card's remaining VRAM headroom (accounting for `in_flight`
/// bytes whose D3D12 allocations have not yet updated DXGI `CurrentUsage`).
fn reserve_with_card(
    placed: &AtomicU64,
    in_flight: &AtomicU64,
    size: u64,
    most: u64,
    card: Option<(u64, u64)>,
    vram_pct: u64,
) -> Option<u64> {
    if size == 0 {
        return None;
    }
    // Step 1: when card VRAM headroom is known, atomically claim it in
    // `in_flight` BEFORE claiming `placed`, so concurrent callers racing
    // against a tight VRAM wall cannot observe the same stale headroom or slip
    // between `placed` and `in_flight`.
    let Some((budget, used)) = card else {
        let next = reserve_bytes(placed, size, most)?;
        in_flight.fetch_add(size, Ordering::Release);
        return Some(next);
    };
    let limit = card_limit_bytes(budget, vram_pct);
    in_flight
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |cur| {
            let effective_used = used.checked_add(cur)?;
            let next_used = effective_used.checked_add(size)?;
            (next_used <= limit).then(|| cur + size)
        })
        .ok()?;
    // Step 2: atomically claim ReBAR byte budget in `placed`, rolling back
    // `in_flight` on failure.
    match reserve_bytes(placed, size, most) {
        Some(next) => Some(next),
        None => {
            in_flight.fetch_sub(size, Ordering::Release);
            None
        }
    }
}

/// RAII token returned by [`reserve`]. Calling [`Reservation::commit`] hands
/// ownership of the byte count to the live `Shadow` (which returns it via
/// [`released`] on drop); dropping without committing rolls back both the
/// placed counters and the in-flight headroom counter.
pub(crate) struct Reservation {
    size: u64,
    committed: bool,
}

impl Reservation {
    /// Marks the D3D12 resource creation complete and returns the reserved byte
    /// count to store on the `Shadow`.
    pub(crate) fn commit(mut self) -> u64 {
        self.committed = true;
        IN_FLIGHT.fetch_sub(self.size, Ordering::Release);
        self.size
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        if !self.committed {
            IN_FLIGHT.fetch_sub(self.size, Ordering::Release);
            released(self.size);
        }
    }
}

/// At the host's start: whether the device has GPU upload heaps, and the
/// line that says so.
pub(crate) fn probe(device: &windows::Win32::Graphics::Direct3D12::ID3D12Device, card_budget: Option<u64>) {
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS16, D3D12_FEATURE_DATA_D3D12_OPTIONS16};
    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: a plain query that fills our structure.
    let asked = unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32)
    };
    let supported = asked.is_ok() && o16.GPUUploadHeapSupported.as_bool();
    let _ = SUPPORTED.set(supported);
    let _ = PROBED_BUDGET.set(card_budget);
    let level = level();
    let most_mb = match std::env::var("VICTORY_REBAR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
        Some(mb) => mb,
        None => default_most(card_budget) >> 20,
    };
    tracing::info!(
        gpu_upload_heaps = supported,
        options16_known = asked.is_ok(),
        card_budget_mb = card_budget.map(|b| b >> 20),
        level,
        most_mb,
        vram_percent = vram_percent(),
        "Resizable BAR (D3D12 GPU upload heaps): {}",
        match (supported, level) {
            (true, 0) => "available, not used (VICTORY_REBAR=1 or 2 places host copies there)",
            (true, 1) => "used for windows of title memory",
            (true, _) => "used for windows of title memory and small title buffers",
            (false, 0) => "not available",
            (false, _) => "NOT available: VICTORY_REBAR is ignored, copies are made as before \
                           (Resizable BAR off in the firmware/driver, or no OS/driver support)",
        }
    );
}

fn note_refused(kind: Kind, size: u64) {
    if PLACED[2].fetch_add(1, Ordering::Relaxed) == 0 {
        tracing::warn!(
            ?kind,
            size,
            "Resizable BAR: a copy stays in the PC's memory (budget or card full)"
        );
    }
}

/// Whether a copy of `size` bytes of `kind` goes to the card's mapped
/// memory; if it does, returns a [`Reservation`] counting its bytes as placed
/// (call [`Reservation::commit`] once the buffer is created, or drop it to
/// roll back automatically).
pub(crate) fn reserve(kind: Kind, size: u64, cpu_read: bool) -> Option<Reservation> {
    let level = level();
    if size == 0 || level == 0 || !supported() || cpu_read {
        return None;
    }
    if matches!(kind, Kind::Buffer) && level < 2 {
        return None;
    }
    let most = most();
    let placed = PLACED[0].load(Ordering::Relaxed);
    // Avoid DXGI `QueryVideoMemoryInfo` syscalls once the ReBAR byte cap is full.
    if placed.saturating_add(size) > most {
        note_refused(kind, size);
        return None;
    }
    let host = crate::host::host();
    let card = host.and_then(|h| h.video_memory()).map(|[b, u, ..]| (b, u));
    let vram_pct = vram_percent();
    let ask = Ask {
        level,
        supported: true,
        kind,
        size,
        cpu_read,
        placed,
        most,
        card: card.map(|(b, u)| (b, u.saturating_add(IN_FLIGHT.load(Ordering::Acquire)))),
    };
    let reserved = match decide_with_percent(&ask, vram_pct) {
        Decision::Rebar => reserve_with_card(&PLACED[0], &IN_FLIGHT, size, most, card, vram_pct),
        Decision::OverBudget | Decision::CardFull => None,
        _ => return None,
    };
    let Some(now) = reserved else {
        note_refused(kind, size);
        return None;
    };
    PLACED[1].fetch_add(1, Ordering::Relaxed);
    let line = NEXT_LINE.load(Ordering::Relaxed);
    if now >= line
        && NEXT_LINE
            .compare_exchange(
                line,
                now / (64 << 20) * (64 << 20) + (64 << 20),
                Ordering::Relaxed,
                Ordering::Relaxed,
            )
            .is_ok()
    {
        tracing::info!(
            placed_mb = now >> 20,
            copies = PLACED[1].load(Ordering::Relaxed),
            "Resizable BAR: host copies placed in the card's mapped memory"
        );
    }
    Some(Reservation {
        size,
        committed: false,
    })
}

/// Gives back what [`reserve`] counted.
pub(crate) fn released(size: u64) {
    PLACED[0].fetch_sub(size, Ordering::Relaxed);
    PLACED[1].fetch_sub(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;

    fn ask(level: u8, kind: Kind) -> Ask {
        Ask {
            level,
            supported: true,
            kind,
            size: 4 * MIB,
            cpu_read: false,
            placed: 0,
            most: GIB,
            card: Some((11 * GIB, 2 * GIB)),
        }
    }

    #[test]
    fn the_switch_reads_only_one_and_two() {
        assert_eq!(level_of(None), 0);
        assert_eq!(level_of(Some("0")), 0);
        assert_eq!(level_of(Some(" 1 ")), 1);
        assert_eq!(level_of(Some("2")), 2);
        assert_eq!(level_of(Some("yes")), 0);
        assert_eq!(level_of(Some("3")), 0);
    }

    #[test]
    fn off_or_unsupported_places_nothing() {
        assert_eq!(decide(&ask(0, Kind::Window)), Decision::Off);
        let mut a = ask(2, Kind::Window);
        a.supported = false;
        assert_eq!(decide(&a), Decision::Unsupported);
    }

    #[test]
    fn level_one_places_windows_only_level_two_buffers_too() {
        assert_eq!(decide(&ask(1, Kind::Window)), Decision::Rebar);
        assert_eq!(decide(&ask(1, Kind::Buffer)), Decision::NotThisKind);
        assert_eq!(decide(&ask(2, Kind::Buffer)), Decision::Rebar);
    }

    #[test]
    fn cpu_read_buffers_never_go_there() {
        for level in [1, 2] {
            for kind in [Kind::Window, Kind::Buffer] {
                let mut a = ask(level, kind);
                a.cpu_read = true;
                assert_eq!(decide(&a), Decision::CpuRead);
            }
        }
    }

    #[test]
    fn the_budget_holds_to_the_byte() {
        let mut a = ask(1, Kind::Window);
        a.placed = GIB - 4 * MIB;
        assert_eq!(decide(&a), Decision::Rebar);
        a.placed += 1;
        assert_eq!(decide(&a), Decision::OverBudget);
        a.placed = u64::MAX;
        assert_eq!(decide(&a), Decision::OverBudget);
    }

    #[test]
    fn reserved_bytes_stop_at_the_cap_and_can_be_returned() {
        let placed = AtomicU64::new(4 * MIB);
        assert_eq!(reserve_bytes(&placed, 4 * MIB, 8 * MIB), Some(8 * MIB));
        assert_eq!(reserve_bytes(&placed, 1, 8 * MIB), None);
        assert_eq!(placed.load(Ordering::Relaxed), 8 * MIB);
        // Allocation failure and teardown return the successful reservation.
        placed.fetch_sub(4 * MIB, Ordering::Relaxed);
        assert_eq!(reserve_bytes(&placed, 4 * MIB, 8 * MIB), Some(8 * MIB));
    }

    #[test]
    fn reserved_bytes_do_not_wrap() {
        let placed = AtomicU64::new(u64::MAX - 1);
        assert_eq!(reserve_bytes(&placed, 2, u64::MAX), None);
        assert_eq!(placed.load(Ordering::Relaxed), u64::MAX - 1);
        assert_eq!(reserve_bytes(&placed, 1, u64::MAX), Some(u64::MAX));
    }

    #[test]
    fn concurrent_reservations_share_one_budget() {
        let placed = AtomicU64::new(0);
        let gate = std::sync::Barrier::new(8);
        let accepted = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        reserve_bytes(&placed, 4 * MIB, 4 * MIB)
                    })
                })
                .collect();
            workers
                .into_iter()
                .filter_map(|worker| worker.join().unwrap())
                .count()
        });
        assert_eq!(accepted, 1);
        assert_eq!(placed.load(Ordering::Relaxed), 4 * MIB);
    }

    #[test]
    fn a_fifth_of_the_card_stays_free() {
        let mut a = ask(1, Kind::Window);
        a.card = Some((10 * GIB, 8 * GIB - 4 * MIB));
        assert_eq!(decide(&a), Decision::Rebar);
        a.card = Some((10 * GIB, 8 * GIB));
        assert_eq!(decide(&a), Decision::CardFull);
        a.card = None;
        assert_eq!(decide(&a), Decision::Rebar);
    }

    #[test]
    fn default_budget_is_a_quarter_of_the_card_at_most_4_gib() {
        assert_eq!(default_most(None), 256 * MIB);
        assert_eq!(default_most(Some(11 * GIB)), 11 * GIB / 4);
        assert_eq!(default_most(Some(30 * GIB)), 4 * GIB);
    }

    #[test]
    fn zero_size_reservation_is_refused() {
        let placed = AtomicU64::new(0);
        let in_flight = AtomicU64::new(0);
        assert_eq!(reserve_bytes(&placed, 0, GIB), None);
        assert_eq!(
            reserve_with_card(&placed, &in_flight, 0, GIB, Some((10 * GIB, 0)), 80),
            None
        );
        assert_eq!(placed.load(Ordering::Relaxed), 0);
        assert_eq!(in_flight.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn concurrent_reservations_respect_card_headroom_wall() {
        // ReBAR budget has plenty of room (2 GiB), but VRAM only has 4 MiB left
        // before the 80% wall on a 12 GiB card. Exactly one of 8 concurrent
        // 4 MiB callers may succeed.
        let placed = AtomicU64::new(512 * MIB);
        let in_flight = AtomicU64::new(0);
        let budget = 12 * GIB;
        let used = card_limit_bytes(budget, 80) - 4 * MIB;
        let gate = std::sync::Barrier::new(8);
        let accepted = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        gate.wait();
                        reserve_with_card(
                            &placed,
                            &in_flight,
                            4 * MIB,
                            2 * GIB,
                            Some((budget, used)),
                            80,
                        )
                    })
                })
                .collect();
            workers
                .into_iter()
                .filter_map(|w| w.join().unwrap())
                .count()
        });
        assert_eq!(accepted, 1);
        assert_eq!(placed.load(Ordering::Relaxed), 516 * MIB);
        assert_eq!(in_flight.load(Ordering::Relaxed), 4 * MIB);
    }

    #[test]
    fn in_flight_reservation_blocks_overshoot_before_dxgi_updates() {
        let placed = AtomicU64::new(0);
        let in_flight = AtomicU64::new(0);
        let budget = 11 * GIB;
        let limit = card_limit_bytes(budget, 80);
        let used = limit - 4 * MIB;

        // First 4 MiB fits and increments in_flight while CreateCommittedResource runs.
        assert_eq!(
            reserve_with_card(&placed, &in_flight, 4 * MIB, 2 * GIB, Some((budget, used)), 80),
            Some(4 * MIB)
        );
        // Second caller sees stale DXGI `used` before first resource completes: blocked by in_flight!
        assert_eq!(
            reserve_with_card(&placed, &in_flight, 4 * MIB, 2 * GIB, Some((budget, used)), 80),
            None
        );
        // Once the first allocation commits (or fails and rolls back), in_flight drops.
        in_flight.fetch_sub(4 * MIB, Ordering::Release);
        assert_eq!(in_flight.load(Ordering::Relaxed), 0);
        // With fresh DXGI usage reflecting `limit`, still blocked.
        assert_eq!(
            reserve_with_card(&placed, &in_flight, 4 * MIB, 2 * GIB, Some((budget, limit)), 80),
            None
        );
        // After other VRAM is freed, a new 4 MiB allocation succeeds again.
        assert_eq!(
            reserve_with_card(&placed, &in_flight, 4 * MIB, 2 * GIB, Some((budget, limit - 4 * MIB)), 80),
            Some(8 * MIB)
        );
    }

    #[test]
    fn card_limit_bytes_is_exact_without_truncation() {
        assert_eq!(card_limit_bytes(10 * GIB, 80), 8 * GIB);
        // `105 / 10 * 8 = 80`, whereas exact `105 * 80 / 100 = 84`.
        assert_eq!(card_limit_bytes(105, 80), 84);
        assert_eq!(
            card_limit_bytes(12 * GIB, 80),
            ((u128::from(12 * GIB) * 80) / 100) as u64
        );
        assert_eq!(card_limit_bytes(u64::MAX, 80), (u128::from(u64::MAX) * 80 / 100) as u64);
    }
}
