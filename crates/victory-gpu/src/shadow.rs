//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Inline-first buffer for coalesced dirty page runs produced by
/// [`CopiedPages::plan_stale`]: avoids heap allocation for up to 8 runs per
/// refresh (the overwhelming majority of refreshes).
struct SmallRuns {
    inline: [(usize, usize); 8],
    inline_len: usize,
    spill: Vec<(usize, usize)>,
}

impl SmallRuns {
    fn new() -> Self {
        Self {
            inline: [(0, 0); 8],
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    fn push(&mut self, from: usize, to: usize) {
        if self.inline_len < self.inline.len() {
            self.inline[self.inline_len] = (from, to);
            self.inline_len += 1;
        } else {
            self.spill.push((from, to));
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inline[..self.inline_len]
            .iter()
            .copied()
            .chain(self.spill.iter().copied())
    }
}

/// Per-page last copied write-watch sequence, paired with a 64-page-per-word
/// bitmask of uncopied pages so partially-initialized buffers and windows
/// still hit the O(1)/O(|written|) fast path on initialized subranges and skip
/// initialized regions 64 pages at a time when uncopied pages exist.
struct CopiedPages {
    at: Vec<u64>,
    /// Bit `i` (word `i / 64`, bit `i % 64`) is 1 iff `at[i] == 0`.
    /// Unused high bits in the final word are always 0.
    uncopied_bits: Vec<u64>,
    uncopied: usize,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        let words = pages.div_ceil(64);
        let mut uncopied_bits = vec![u64::MAX; words];
        let rem = pages % 64;
        if rem != 0 {
            uncopied_bits[words - 1] = (1u64 << rem) - 1;
        }
        Self {
            at: vec![0; pages],
            uncopied_bits,
            uncopied: pages,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        self.at.get(index).copied()
    }

    fn set(&mut self, index: usize, now: u64) {
        if let Some(slot) = self.at.get_mut(index) {
            if *slot == 0 && now != 0 {
                self.uncopied -= 1;
                self.uncopied_bits[index / 64] &= !(1u64 << (index % 64));
            } else if *slot != 0 && now == 0 {
                self.uncopied += 1;
                self.uncopied_bits[index / 64] |= 1u64 << (index % 64);
            }
            *slot = now;
        }
    }

    #[cfg(test)]
    fn all_copied(&self) -> bool {
        self.uncopied == 0
    }

    /// Whether every page index in `lo..=hi` has been copied whole at least once.
    fn range_all_copied(&self, lo: usize, hi: usize) -> bool {
        if self.uncopied == 0 {
            return true;
        }
        let first_w = lo / 64;
        let last_w = hi / 64;
        let lo_mask = u64::MAX << (lo % 64);
        let hi_mask = u64::MAX >> (63 - (hi % 64));
        if first_w == last_w {
            return (self.uncopied_bits[first_w] & lo_mask & hi_mask) == 0;
        }
        if (self.uncopied_bits[first_w] & lo_mask) != 0 {
            return false;
        }
        for &word in &self.uncopied_bits[first_w + 1..last_w] {
            if word != 0 {
                return false;
            }
        }
        (self.uncopied_bits[last_w] & hi_mask) == 0
    }

    /// Lowest page index `>= from_idx` and `<= hi` that has never been copied whole.
    fn next_uncopied(&self, from_idx: usize, hi: usize) -> Option<usize> {
        if from_idx > hi || self.uncopied == 0 {
            return None;
        }
        let mut w = from_idx / 64;
        let last_w = hi / 64;
        let mut word = self.uncopied_bits[w] & (u64::MAX << (from_idx % 64));
        loop {
            if w == last_w {
                word &= u64::MAX >> (63 - (hi % 64));
            }
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
            if w == last_w {
                return None;
            }
            w += 1;
            word = self.uncopied_bits[w];
        }
    }

    /// Plans the coalesced byte ranges `[run_from, run_to)` inside `[from, to)`
    /// that must be copied from title memory, updating whole-page copied
    /// timestamps to `now`. Runs purely on metadata so callers can release the
    /// `copied` mutex before invoking memory copies.
    #[allow(clippy::too_many_arguments)]
    fn plan_stale(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
    ) -> SmallRuns {
        let mut runs = SmallRuns::new();
        if from >= to {
            return runs;
        }
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let lo = first_page - base;
        let hi = last - base;

        let push_page = |runs: &mut SmallRuns, run: &mut Option<(usize, usize)>, page: usize| {
            match *run {
                Some((_, run_end)) if page < run_end => {}
                Some((run_start, run_end)) if page == run_end => {
                    *run = Some((run_start, page + 1));
                }
                Some((run_start, run_end)) => {
                    runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                    *run = Some((page, page + 1));
                }
                None => *run = Some((page, page + 1)),
            }
        };

        let mut run: Option<(usize, usize)> = None;
        let w_start = if written.len() > 8 {
            written.partition_point(|&(p, _)| p < first_page)
        } else {
            0
        };

        // Fast path: every page in `[from, to)` has been initialized at least
        // once (even if other pages of the same buffer/window have not).
        if self.range_all_copied(lo, hi) {
            if written.is_empty() {
                return runs;
            }
            for &(page, write) in &written[w_start..] {
                if page < first_page {
                    continue;
                }
                if page > last {
                    break;
                }
                let idx = page - base;
                if write <= self.at[idx] {
                    continue;
                }
                // A page only partly in the range is copied in part: it stays
                // stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    self.set(idx, now);
                }
                push_page(&mut runs, &mut run, page);
            }
            if let Some((run_start, run_end)) = run {
                runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
            }
            return runs;
        }

        // Sparse merge of uncopied pages (via 64-page bitmask) and dirty
        // `written` entries in `[first_page, last]`, skipping already-clean
        // initialized spans without touching `self.at`.
        let mut wi = w_start;
        while wi < written.len() && written[wi].0 < first_page {
            wi += 1;
        }
        let mut next_u = self.next_uncopied(lo, hi);
        loop {
            while wi < written.len() && written[wi].0 <= last {
                let (wp, write) = written[wi];
                let idx = wp - base;
                if self.at[idx] == 0 || write > self.at[idx] {
                    break;
                }
                wi += 1;
            }
            let next_w = (wi < written.len() && written[wi].0 <= last).then(|| written[wi].0 - base);
            let idx = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (Some(u), Some(w)) if w < u => {
                    wi += 1;
                    w
                }
                (Some(u), Some(_)) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    wi += 1;
                    u
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (None, Some(w)) => {
                    wi += 1;
                    w
                }
                (None, None) => break,
            };
            let page = base + idx;
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                self.set(idx, now);
            }
            push_page(&mut runs, &mut run, page);
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((run_start, run_end)) = run {
            runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
        }
        runs
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn copied_count(&self) -> usize {
        self.at.len() - self.uncopied
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<CopiedPages>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!pending.is_empty()` so read-only windows and
    /// buffers skip acquiring `pending` on every refresh and flight check.
    has_pending: AtomicBool,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// Bytes reserved in `crate::rebar` (0 if not placed in GPU upload heap).
    rebar: u64,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!dirty.is_empty()`; cleared only after `sent` and
    /// `has_sent` have been updated so concurrent `card_merge_relay` readers
    /// observing `has_dirty == false` are guaranteed to observe `has_sent == true`.
    has_dirty: AtomicBool,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// Lock-free mirror of `!sent.is_empty()` so clean carded copies skip
    /// locking `sent` on every submission.
    has_sent: AtomicBool,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The least size of a title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" hashes not counted), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_KB` (default 256 KiB; `VICTORY_CARD_BUFFERS_MIN_MB`
/// overrides in MiB if explicitly set) and `VICTORY_CARD_BUFFERS_MB` (default
/// 2048 MiB).
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default) << 20
        };
        // From 256 KiB up (checked against actual buffer length, not 64 KiB
        // rounded size) and 2,048 MiB total by default: moves medium-to-large
        // shader-written buffers (640-828 KiB+) onto VRAM + GPU CARD_COMPARE
        // while keeping tiny buffers out of card-compare dispatch churn.
        let min_bytes = if let Ok(v) = std::env::var("VICTORY_CARD_BUFFERS_MIN_MB")
            && let Ok(n) = v.trim().parse::<u64>()
        {
            (n << 20).max(4 << 10)
        } else {
            let kb = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .unwrap_or(256);
            (kb << 10).max(4 << 10)
        };
        (min_bytes, mb("VICTORY_CARD_BUFFERS_MB", 2048))
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// Maximum share of VRAM budget (%) up to which card buffers may be allocated
/// (`VICTORY_CARD_VRAM_PERCENT`, default 92; clamped 50..=95). Kept higher
/// than [`crate::resource::room_percent`] so hot compute/skinning buffers keep
/// VRAM priority over streamed textures.
fn card_vram_percent() -> u64 {
    static PERCENT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *PERCENT.get_or_init(|| {
        std::env::var("VICTORY_CARD_VRAM_PERCENT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map_or(92, |p| p.clamp(50, 95))
    })
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(len: u64, size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if len < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room under `card_vram_percent()` (default 88%,
    // above textures' 75% cap so hot compute buffers like skinning stay in
    // local VRAM instead of spilling onto PCIe).
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * card_vram_percent()
    {
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; default 1536 MiB, `=0` disables).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while VRAM usage stays below 72% of budget, leaving textures (75%) and
/// title buffers (`card_vram_percent`, 92%) their room. Tiny windows under
/// 64 KiB stay in shared memory to avoid cluttering `CARDED`.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1536) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() || !((64 << 10)..(1 << 30)).contains(&size) {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * 72
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let relay_on = relay();
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let Some(c) = shadow.card.as_ref() else {
            return true;
        };
        let has_dirty = c.has_dirty.load(Ordering::Acquire);
        let has_sent = relay_on && c.has_sent.load(Ordering::Acquire);
        if !has_dirty && !has_sent {
            return true;
        }
        let ranges = if has_dirty {
            let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let ranges = std::mem::take(&mut *dirty);
            if relay_on {
                let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                sent.retain(|s| s.2 > fetched);
                sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                // Publish `has_sent` BEFORE clearing `has_dirty` so a concurrent
                // `card_merge_relay` checking `has_dirty` then `has_sent` never
                // observes both false while a range is transitioning.
                c.has_sent.store(!sent.is_empty(), Ordering::Release);
            }
            c.has_dirty.store(false, Ordering::Release);
            ranges
        } else {
            // Only pruning stale `sent` entries (respects `dirty` -> `sent` lock order).
            let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
            sent.retain(|s| s.2 > fetched);
            c.has_sent.store(!sent.is_empty(), Ordering::Release);
            Vec::new()
        };
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
#[cfg(test)]
pub(crate) fn joined_piece_runs(runs: Vec<CardSpan>) -> Vec<CardSpan> {
    joined_piece_runs_within(runs, 0)
}

/// Joins piece runs that touch or are separated by at most `max_gap` 64 KiB
/// pieces. Re-hashing a few unchanged pieces in parallel inside one compute
/// dispatch costs the same GPU wall-clock as hashing a single piece while
/// cutting hundreds of small `Dispatch` launches per frame.
pub(crate) fn joined_piece_runs_within(mut runs: Vec<CardSpan>, max_gap: u32) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last)
                if run.first <= last.first.saturating_add(last.pieces).saturating_add(max_gap) =>
            {
                let end = (last.first + last.pieces).max(run.first + run.pieces);
                last.pieces = end - last.first;
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Drains write-combining buffers after sequential CPU stores into a ReBAR
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`) mapping so subsequent reads or GPU work
/// observe the completed writes in order.
#[inline]
fn wc_sfence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SFENCE is part of baseline SSE2 on every x86-64 CPU.
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// Copies `len` bytes from write-combined (USWC) memory (`src`, e.g. a ReBAR
/// `D3D12_HEAP_TYPE_GPU_UPLOAD` mapping) into cached system memory (`dst`).
/// Uses AVX2 `VMOVNTDQA` streaming loads on 32-byte aligned addresses bracketed
/// by `_mm_mfence()` so streaming loads cannot bypass prior same-core WC stores
/// or reorder past following WC stores or synchronization loads.
///
/// # Safety
/// `src` must be valid for reads of `len` bytes, `dst` valid for writes of
/// `len` bytes, and the two ranges must not overlap.
unsafe fn copy_from_wc(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        unsafe { copy_from_wc_avx2(src, dst, len) };
        return;
    }
    // SAFETY: caller's bounds and non-overlap guarantee.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_from_wc_avx2(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m256i, _mm_mfence, _mm256_storeu_si256, _mm256_stream_load_si256};
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    // Order VMOVNTDQA after any prior same-core WC stores / sync loads, and
    // guarantee an exit MFENCE before any following WC stores or reads.
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = src.align_offset(32).min(len);
    if prefix > 0 {
        // SAFETY: `prefix <= len`.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, prefix) };
    }
    let mut at = prefix;
    // Issue 8 aligned VMOVNTDQA streaming loads (256 bytes = 4 cache lines)
    // back-to-back before storing so the CPU's fill buffers overlap multiple
    // PCIe read bursts simultaneously.
    while at + 256 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 256 <= len`.
        unsafe {
            let s = src.add(at);
            let d = dst.add(at);
            let v0 = _mm256_stream_load_si256(s.cast::<__m256i>());
            let v1 = _mm256_stream_load_si256(s.add(32).cast::<__m256i>());
            let v2 = _mm256_stream_load_si256(s.add(64).cast::<__m256i>());
            let v3 = _mm256_stream_load_si256(s.add(96).cast::<__m256i>());
            let v4 = _mm256_stream_load_si256(s.add(128).cast::<__m256i>());
            let v5 = _mm256_stream_load_si256(s.add(160).cast::<__m256i>());
            let v6 = _mm256_stream_load_si256(s.add(192).cast::<__m256i>());
            let v7 = _mm256_stream_load_si256(s.add(224).cast::<__m256i>());
            _mm256_storeu_si256(d.cast::<__m256i>(), v0);
            _mm256_storeu_si256(d.add(32).cast::<__m256i>(), v1);
            _mm256_storeu_si256(d.add(64).cast::<__m256i>(), v2);
            _mm256_storeu_si256(d.add(96).cast::<__m256i>(), v3);
            _mm256_storeu_si256(d.add(128).cast::<__m256i>(), v4);
            _mm256_storeu_si256(d.add(160).cast::<__m256i>(), v5);
            _mm256_storeu_si256(d.add(192).cast::<__m256i>(), v6);
            _mm256_storeu_si256(d.add(224).cast::<__m256i>(), v7);
        }
        at += 256;
    }
    while at + 32 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        unsafe {
            let v = _mm256_stream_load_si256(src.add(at).cast::<__m256i>());
            _mm256_storeu_si256(dst.add(at).cast::<__m256i>(), v);
        }
        at += 32;
    }
    if at < len {
        // SAFETY: remaining tail within `len`.
        unsafe { std::ptr::copy_nonoverlapping(src.add(at), dst.add(at), len - at) };
    }
}

/// Compares `len` bytes of write-combined memory (`wc_src`) against cached
/// memory (`cached`) using AVX2 `VMOVNTDQA` streaming loads on aligned
/// addresses bracketed by `_mm_mfence()`, avoiding uncached PCIe scalar loads
/// and scratch allocations.
///
/// # Safety
/// Both `wc_src` and `cached` must be valid for reads of `len` bytes.
unsafe fn slices_equal_wc(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        return unsafe { slices_equal_wc_avx2(wc_src, cached, len) };
    }
    // SAFETY: caller's bounds guarantee.
    unsafe {
        std::slice::from_raw_parts(wc_src, len) == std::slice::from_raw_parts(cached, len)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn slices_equal_wc_avx2(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm_mfence, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_stream_load_si256,
    };
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    // Order VMOVNTDQA after any prior same-core WC stores / sync loads, and
    // guarantee an exit MFENCE on every return path (including early `false`).
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = wc_src.align_offset(32).min(len);
    for i in 0..prefix {
        // SAFETY: `i < prefix <= len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    let mut at = prefix;
    // Branch ONLY once per 4 KiB block (16 x 256-byte chunks) so the CPU does
    // not serialize USWC streaming loads behind a per-cache-line branch.
    const BLOCK: usize = 4096;
    while at + 256 <= len {
        let block_end = at.saturating_add(BLOCK).min(len - (len - at) % 256);
        let mut acc0 = _mm256_set1_epi8(-1);
        let mut acc1 = _mm256_set1_epi8(-1);
        while at < block_end {
            // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 256 <= len`.
            unsafe {
                let w = wc_src.add(at);
                let c = cached.add(at);
                let w0 = _mm256_stream_load_si256(w.cast::<__m256i>());
                let w1 = _mm256_stream_load_si256(w.add(32).cast::<__m256i>());
                let w2 = _mm256_stream_load_si256(w.add(64).cast::<__m256i>());
                let w3 = _mm256_stream_load_si256(w.add(96).cast::<__m256i>());
                let w4 = _mm256_stream_load_si256(w.add(128).cast::<__m256i>());
                let w5 = _mm256_stream_load_si256(w.add(160).cast::<__m256i>());
                let w6 = _mm256_stream_load_si256(w.add(192).cast::<__m256i>());
                let w7 = _mm256_stream_load_si256(w.add(224).cast::<__m256i>());
                let e0 = _mm256_cmpeq_epi8(w0, _mm256_loadu_si256(c.cast::<__m256i>()));
                let e1 = _mm256_cmpeq_epi8(w1, _mm256_loadu_si256(c.add(32).cast::<__m256i>()));
                let e2 = _mm256_cmpeq_epi8(w2, _mm256_loadu_si256(c.add(64).cast::<__m256i>()));
                let e3 = _mm256_cmpeq_epi8(w3, _mm256_loadu_si256(c.add(96).cast::<__m256i>()));
                let e4 = _mm256_cmpeq_epi8(w4, _mm256_loadu_si256(c.add(128).cast::<__m256i>()));
                let e5 = _mm256_cmpeq_epi8(w5, _mm256_loadu_si256(c.add(160).cast::<__m256i>()));
                let e6 = _mm256_cmpeq_epi8(w6, _mm256_loadu_si256(c.add(192).cast::<__m256i>()));
                let e7 = _mm256_cmpeq_epi8(w7, _mm256_loadu_si256(c.add(224).cast::<__m256i>()));
                acc0 = _mm256_and_si256(acc0, _mm256_and_si256(_mm256_and_si256(e0, e1), _mm256_and_si256(e2, e3)));
                acc1 = _mm256_and_si256(acc1, _mm256_and_si256(_mm256_and_si256(e4, e5), _mm256_and_si256(e6, e7)));
            }
            at += 256;
        }
        if _mm256_movemask_epi8(_mm256_and_si256(acc0, acc1)) != -1i32 {
            return false;
        }
    }
    while at + 32 <= len {
        // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        let eq = unsafe {
            let w = _mm256_stream_load_si256(wc_src.add(at).cast::<__m256i>());
            let c = _mm256_loadu_si256(cached.add(at).cast::<__m256i>());
            _mm256_cmpeq_epi8(w, c)
        };
        if _mm256_movemask_epi8(eq) != -1i32 {
            return false;
        }
        at += 32;
    }
    for i in at..len {
        // SAFETY: `i < len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    true
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let n = now.len().min(held.len()).min(title.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_piece_avx2(&now[..n], &mut held[..n], &mut title[..n]) };
    }
    merge_piece_scalar(&now[..n], &mut held[..n], &mut title[..n])
}

fn merge_piece_scalar(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((&n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        if n != *h {
            differs |= n ^ *t;
            *t = n;
            *h = n;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_piece_avx2(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };
    let n = now.len();
    let (np, hp, tp) = (now.as_ptr(), held.as_mut_ptr(), title.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vn = _mm256_loadu_si256(np.add(i).cast::<__m256i>());
            let vh = _mm256_loadu_si256(hp.add(i).cast::<__m256i>());
            let eq = _mm256_cmpeq_epi8(vn, vh);
            if _mm256_movemask_epi8(eq) != -1i32 {
                let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
                let merged = _mm256_blendv_epi8(vn, vt, eq);
                vdiff = _mm256_or_si256(vdiff, _mm256_xor_si256(merged, vt));
                _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                _mm256_storeu_si256(hp.add(i).cast::<__m256i>(), vn);
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_piece_scalar(&now[i..], &mut held[i..], &mut title[i..]) || differs
}

/// Synchronous copy-back merge for [`Shadow::copy_back_with`]: for each byte,
/// takes `host` where `host != old` (GPU wrote), else `title` (keeping title
/// CPU writes); writes the merged bytes into both `title` and `host` while
/// skipping stores to unchanged 32-byte vectors so clean GPU-shared cache lines
/// are not dirtied. Returns whether `title` changed.
fn merge_sync_host(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let n = title.len().min(host.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_sync_host_avx2(&mut title[..n], &mut host[..n], &old[..n]) };
    }
    merge_sync_host_scalar(&mut title[..n], &mut host[..n], &old[..n])
}

fn merge_sync_host_scalar(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    let mut differs = 0u8;
    for ((t, h), &o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
        if *h != o {
            differs |= *h ^ *t;
            *t = *h;
        } else if *h != *t {
            *h = *t;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_sync_host_avx2(title: &mut [u8], host: &mut [u8], old: &[u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_testz_si256, _mm256_xor_si256,
    };
    let n = title.len();
    let (tp, hp, op) = (title.as_mut_ptr(), host.as_mut_ptr(), old.as_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vh = _mm256_loadu_si256(hp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_ho = _mm256_cmpeq_epi8(vh, vo);
            let eq_th = _mm256_cmpeq_epi8(vt, vh);
            if _mm256_movemask_epi8(_mm256_and_si256(eq_ho, eq_th)) != -1i32 {
                let merged = _mm256_blendv_epi8(vh, vt, eq_ho);
                let diff_t = _mm256_xor_si256(merged, vt);
                if _mm256_testz_si256(diff_t, diff_t) == 0 {
                    vdiff = _mm256_or_si256(vdiff, diff_t);
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(merged, vh)) != -1i32 {
                    _mm256_storeu_si256(hp.add(i).cast::<__m256i>(), merged);
                }
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_sync_host_scalar(&mut title[i..], &mut host[i..], &old[i..]) || differs
}

/// Synchronous card-buffer copy-back merge for [`Shadow::card_merge`]: takes
/// `gpu` where `gpu != old`, else `title`, writing merged bytes into both
/// `title` and `old` without dirtying unchanged 32-byte vectors. Returns
/// whether `title` changed.
fn merge_sync_card(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let n = title.len().min(gpu.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_sync_card_avx2(&mut title[..n], &gpu[..n], &mut old[..n]) };
    }
    merge_sync_card_scalar(&mut title[..n], &gpu[..n], &mut old[..n])
}

fn merge_sync_card_scalar(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((t, &g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
        if g != *o {
            differs |= g ^ *t;
            *t = g;
            *o = g;
        } else if *o != *t {
            *o = *t;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_sync_card_avx2(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_testz_si256, _mm256_xor_si256,
    };
    let n = title.len();
    let (tp, gp, op) = (title.as_mut_ptr(), gpu.as_ptr(), old.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vg = _mm256_loadu_si256(gp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_go = _mm256_cmpeq_epi8(vg, vo);
            let eq_to = _mm256_cmpeq_epi8(vt, vo);
            if _mm256_movemask_epi8(_mm256_and_si256(eq_go, eq_to)) != -1i32 {
                let merged = _mm256_blendv_epi8(vg, vt, eq_go);
                let diff_t = _mm256_xor_si256(merged, vt);
                if _mm256_testz_si256(diff_t, diff_t) == 0 {
                    vdiff = _mm256_or_si256(vdiff, diff_t);
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(merged, vo)) != -1i32 {
                    _mm256_storeu_si256(op.add(i).cast::<__m256i>(), merged);
                }
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_sync_card_scalar(&mut title[i..], &gpu[i..], &mut old[i..]) || differs
}

/// Merges GPU-written bytes (`gpu != old`) into `title` without modifying `gpu`
/// or `old`. Used by the WC ReBAR path of [`Shadow::copy_back_with`] so `gpu`
/// stays intact as the baseline for [`changed_runs`]. Returns whether `title`
/// changed.
fn merge_gpu_into_title(title: &mut [u8], gpu: &[u8], old: &[u8]) -> bool {
    let n = title.len().min(gpu.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_gpu_into_title_avx2(&mut title[..n], &gpu[..n], &old[..n]) };
    }
    merge_gpu_into_title_scalar(&mut title[..n], &gpu[..n], &old[..n])
}

fn merge_gpu_into_title_scalar(title: &mut [u8], gpu: &[u8], old: &[u8]) -> bool {
    let mut differs = 0u8;
    for ((t, &g), &o) in title.iter_mut().zip(gpu).zip(old) {
        if g != o {
            differs |= g ^ *t;
            *t = g;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_gpu_into_title_avx2(title: &mut [u8], gpu: &[u8], old: &[u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };
    let n = title.len();
    let (tp, gp, op) = (title.as_mut_ptr(), gpu.as_ptr(), old.as_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vg = _mm256_loadu_si256(gp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq = _mm256_cmpeq_epi8(vg, vo);
            if _mm256_movemask_epi8(eq) != -1i32 {
                let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
                let merged = _mm256_blendv_epi8(vg, vt, eq);
                let diff_t = _mm256_xor_si256(merged, vt);
                if _mm256_testz_si256(diff_t, diff_t) == 0 {
                    vdiff = _mm256_or_si256(vdiff, diff_t);
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_gpu_into_title_scalar(&mut title[i..], &gpu[i..], &old[i..]) || differs
}

/// `[from, to)` without the ranges of `skip`, in order.
fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;
/// Physical-page view memory tracked by `watch::writes_guarded` (skips the
/// failing `GetWriteWatch` kernel syscall on every refresh).
const VIEW_GUARDED: u8 = 3;

/// Where `a` and `b` (of one length) first differ (AVX2 32 bytes at a time,
/// falling back to 8 bytes at a time).
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { first_difference_avx2(&a[..n], &b[..n]) };
    }
    first_difference_scalar(&a[..n], &b[..n])
}

fn first_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let whole = a.len().min(b.len()) / 8 * 8;
    let from = a[..whole]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(b[..whole].chunks_exact(8))
        .position(|(x, y)| x != y)
        .map_or(whole, |word| word * 8);
    a[from..]
        .iter()
        .zip(&b[from..])
        .position(|(x, y)| x != y)
        .map(|i| from + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + (!mask).trailing_zeros() as usize);
        }
        i += 32;
    }
    first_difference_scalar(&a[i..], &b[i..]).map(|off| i + off)
}

/// Where `a` and `b` (of one length) last differ (AVX2 32 bytes at a time,
/// falling back to 8 bytes at a time).
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let n = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { last_difference_avx2(&a[..n], &b[..n]) };
    }
    last_difference_scalar(&a[..n], &b[..n])
}

fn last_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let whole = a.len().min(b.len()) / 8 * 8;
    if let Some(i) = a[whole..].iter().zip(&b[whole..]).rposition(|(x, y)| x != y) {
        return Some(whole + i);
    }
    let word = a[..whole]
        .as_chunks::<8>()
        .0
        .iter()
        .zip(b[..whole].chunks_exact(8))
        .rposition(|(x, y)| x != y)?
        * 8;
    a[word..word + 8]
        .iter()
        .zip(&b[word..word + 8])
        .rposition(|(x, y)| x != y)
        .map(|i| word + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn last_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let whole = n & !31;
    if let Some(i) = a[whole..].iter().zip(&b[whole..]).rposition(|(x, y)| x != y) {
        return Some(whole + i);
    }
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = whole;
    while i >= 32 {
        i -= 32;
        // SAFETY: `i + 32 <= whole <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + 31 - (!mask).leading_zeros() as usize);
        }
    }
    None
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The word-wise searches agree with the byte-wise ones they replace.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..61u32).map(|i| (i * 7 + 3) as u8).collect();
        let same = base.clone();
        assert_eq!(first_difference(&base, &same), None);
        assert_eq!(last_difference(&base, &same), None);
        for len in [0usize, 1, 7, 8, 9, 16, 23, 61] {
            for at in 0..len {
                for other in at..len {
                    let (a, mut b) = (&base[..len], base[..len].to_vec());
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on the kept worker pool ([`wide`]) if they are many:
/// a snapshot of the world's big heap is 100-170 MB per submission, 2-4 GB a
/// frame, and one thread copies some 8 GB/s. When `src_wc` is true (`src` is
/// write-combined ReBAR memory), uses AVX2 `VMOVNTDQA` streaming loads.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize, src_wc: bool) {
    // Keep WC PCIe BAR reads single-threaded: concurrent multi-core reads to
    // distinct BAR offsets thrash the PCIe read queue and drop bandwidth ~10x.
    if src_wc {
        // SAFETY: caller's bounds and non-overlap guarantee.
        unsafe { copy_from_wc(src, dst, len) };
        return;
    }
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: caller's bounds and non-overlap guarantee.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    let mut jobs: Vec<Job> = Vec::with_capacity(threads);
    for i in 0..threads {
        let at = (i * each).min(len);
        let n = each.min(len - at);
        if n == 0 {
            continue;
        }
        jobs.push(Box::new(move || {
            // SAFETY: disjoint sub-ranges within the caller's valid ranges.
            unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
        }));
    }
    wide(jobs);
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Lock-free completion state for one [`wide`] batch: workers decrement `left`
/// atomically and only lock `lock` to notify `ready` on the final job (`1 -> 0`).
/// Any panic in a worker job is captured so the worker thread stays alive, the
/// batch finishes without deadlocking, and the caller re-throws the panic.
struct WideDone {
    left: AtomicUsize,
    lock: Mutex<()>,
    ready: std::sync::Condvar,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl WideDone {
    fn finish_one(&self) {
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.ready.notify_all();
        }
    }

    fn run_job(&self, job: Job) {
        struct FinishGuard<'a>(&'a WideDone);
        impl Drop for FinishGuard<'_> {
            fn drop(&mut self) {
                self.0.finish_one();
            }
        }
        let _finish = FinishGuard(self);
        let _in_wide = InWideGuard::enter();
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
            let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
            }
        }
    }

    fn wait_and_rethrow(&self) {
        if self.left.load(Ordering::Acquire) > 0 {
            let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            while self.left.load(Ordering::Acquire) > 0 {
                guard = self.ready.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
        }
        if let Some(payload) = self.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
            std::panic::resume_unwind(payload);
        }
    }
}

thread_local! {
    /// Prevents nested [`wide`] calls (e.g. from inside a worker job) from
    /// starving the pool; nested calls execute inline instead.
    static IN_WIDE: Cell<bool> = const { Cell::new(false) };
    /// Reusable 64 KiB scratch buffer for [`Shadow::copy_back_with`] and
    /// [`Shadow::merge_under_writes`], avoiding per-call heap allocation and
    /// zero-initialization.
    static SCRATCH_TITLE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Reusable 64 KiB scratch buffer for copying WC ReBAR pieces into cached
    /// RAM once before diffing and merging in [`Shadow::copy_back_with`].
    static SCRATCH_WC: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

struct InWideGuard(bool);

impl InWideGuard {
    fn enter() -> Self {
        let prev = IN_WIDE.with(|c| c.replace(true));
        Self(prev)
    }
}

impl Drop for InWideGuard {
    fn drop(&mut self) {
        IN_WIDE.with(|c| c.set(self.0));
    }
}

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<(Job, Arc<WideDone>)>>,
    more: std::sync::Condvar,
    threads: usize,
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let n = std::thread::available_parallelism()
        .map_or(4, |n| n.get() / 4)
        .clamp(2, 6);
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
        threads: n,
    });
    STARTED.call_once(|| {
        for i in 0..workers.threads {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                let workers = WORKERS.get().expect("made above");
                loop {
                    let (job, done) = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(item) = jobs.pop_front() {
                                break item;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    done.run_job(job);
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. Any panic in a job is propagated to the caller without
/// deadlocking or losing worker threads.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 || IN_WIDE.with(Cell::get) {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let count = jobs.len();
    let done = Arc::new(WideDone {
        left: AtomicUsize::new(count),
        lock: Mutex::new(()),
        ready: std::sync::Condvar::new(),
        panic: Mutex::new(None),
    });
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            queue.push_back((job, done.clone()));
        }
    }
    if count == 2 {
        workers.more.notify_one();
    } else {
        workers.more.notify_all();
    }
    loop {
        let next = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match next {
            Some((job, item_done)) => item_done.run_job(job),
            None => break,
        }
    }
    done.wait_and_rethrow();
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;
/// Minimum total bytes before splitting work across [`wide`] workers.
const WIDE_MIN_SPLIT: usize = 256 << 10;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes, source is write-combined), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize, bool)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs_wc(snaps));
}

/// `copies` (from, to, bytes) as pieces of work for [`wide`].
#[cfg(test)]
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    copy_jobs_wc(
        copies
            .into_iter()
            .map(|(src, dst, len)| (src, dst, len, false))
            .collect(),
    )
}

/// `copies` (from, to, bytes, src_is_wc) as pieces of work up to [`WIDE_JOB`]
/// bytes each: long ones cut, short ones grouped; medium totals split across
/// workers instead of serializing into a single job.
fn copy_jobs_wc(copies: Vec<(usize, usize, usize, bool)>) -> Vec<Job> {
    let mut cached = Vec::new();
    let mut wc = Vec::new();
    for entry in copies {
        if entry.3 {
            wc.push((entry.0, entry.1, entry.2));
        } else {
            cached.push((entry.0, entry.1, entry.2));
        }
    }
    let total: usize = cached.iter().map(|c| c.2).sum();
    let target = if total < WIDE_MIN_SPLIT {
        WIDE_JOB
    } else {
        (total / 5).clamp(WIDE_MIN_SPLIT, WIDE_JOB)
    };
    let mut jobs: Vec<Job> = Vec::new();
    // Keep all WC BAR reads serialized on one worker so multiple cores never
    // thrash the PCIe read queue while cached copies run in parallel.
    if !wc.is_empty() {
        jobs.push(Box::new(move || {
            for (src, dst, len) in wc {
                // SAFETY: a ReBAR WC mapping and its cached snapshot, non-overlapping.
                unsafe { copy_from_wc(src as *const u8, dst as *mut u8, len) };
            }
        }));
    }
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: a host copy's mapping and its snapshot, non-overlapping.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in cached {
        let mut at = 0;
        while at < len {
            let n = (target - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= target {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (
                    shadow.data as usize,
                    before.as_ptr() as usize,
                    shadow.start as usize,
                    from,
                    to,
                    shadow.rebar != 0,
                ),
                None => (0, 0, 0, 0, 0, false),
            }
        })
        .collect();
    find_pieces_wc(scans)
}

#[cfg(test)]
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    find_pieces_wc(
        scans
            .into_iter()
            .map(|(data, before, start, from, to)| (data, before, start, from, to, false))
            .collect(),
    )
}

/// For each of `scans` (copy pointer, snapshot pointer, copy title start,
/// offsets from and to, whether the copy pointer is write-combined): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work that hold whole pieces.
fn find_pieces_wc(scans: Vec<(usize, usize, usize, usize, usize, bool)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, bool, usize, usize);
    let total_cached: usize = scans
        .iter()
        .filter(|s| !s.5)
        .map(|s| s.4.saturating_sub(s.3))
        .sum();
    let target = if total_cached < WIDE_MIN_SPLIT {
        WIDE_JOB
    } else {
        (total_cached / 5).clamp(WIDE_MIN_SPLIT, WIDE_JOB)
    };
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut wc_group: Vec<Part> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, is_wc, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let same = if is_wc {
                        unsafe { slices_equal_wc((data + at) as *const u8, (before + at) as *const u8, n) }
                    } else {
                        unsafe {
                            std::slice::from_raw_parts((data + at) as *const u8, n)
                                == std::slice::from_raw_parts((before + at) as *const u8, n)
                        }
                    };
                    if !same {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to, is_wc)) in scans.into_iter().enumerate() {
        if is_wc {
            // Keep all WC BAR scans in one sequential job so concurrent CPU
            // cores never contend on the PCIe BAR read queue (~10x faster).
            if from < to {
                wc_group.push((data, before, start, from, to, true, which, 0));
            }
            continue;
        }
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (target - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, false, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= target {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut wc_group, &mut jobs);
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { changed_runs_avx2(&now[..n], &was[..n]) };
    }
    changed_runs_scalar(&now[..n], &was[..n])
}

fn changed_runs_scalar(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn changed_runs_avx2(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = now.len();
    let (np, wp) = (now.as_ptr(), was.as_ptr());
    let mut runs = Vec::new();
    let mut run_start: Option<usize> = None;
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within both slices.
        let diff = unsafe {
            let vn = _mm256_loadu_si256(np.add(i).cast::<__m256i>());
            let vw = _mm256_loadu_si256(wp.add(i).cast::<__m256i>());
            !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(vn, vw)) as u32)
        };
        if diff == 0 {
            if let Some(start) = run_start.take() {
                runs.push((start, i - start));
            }
        } else if diff == u32::MAX {
            run_start.get_or_insert(i);
        } else {
            let mut pos = 0usize;
            while pos < 32 {
                let rem = diff >> pos;
                if run_start.is_some() {
                    let ones = (rem.trailing_ones() as usize).min(32 - pos);
                    pos += ones;
                    if pos < 32 {
                        let start = run_start.take().expect("checked is_some above");
                        runs.push((start, i + pos - start));
                    }
                } else {
                    if rem == 0 {
                        break;
                    }
                    pos += rem.trailing_zeros() as usize;
                    run_start = Some(i + pos);
                }
            }
        }
        i += 32;
    }
    while i < n {
        if now[i] != was[i] {
            run_start.get_or_insert(i);
        } else if let Some(start) = run_start.take() {
            runs.push((start, i - start));
        }
        i += 1;
    }
    if let Some(start) = run_start {
        runs.push((start, n - start));
    }
    runs
}

#[cfg(test)]
mod merge_tests {
    use super::changed_runs;

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.rebar != 0 {
            crate::rebar::released(self.rebar);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.copied_count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter().filter_map(|key| map.remove(&key)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether the memory behind the copy is physical-page view memory tracked
    /// by guards rather than `GetWriteWatch` (`ask_ahead` can skip it).
    pub(crate) fn is_view_guarded(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == VIEW_GUARDED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Whether the copy is placed over title memory in place (other
    /// buffers may alias it).
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages as usize)),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            has_pending: AtomicBool::new(false),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            rebar: 0,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        if !crate::readback::copy_back(address, len) {
            return;
        }
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            self.has_pending.store(true, Ordering::Release);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let (init_from, init_to) = self.inside(address, len);
        let mut parts = Vec::with_capacity(4);
        if init_from < init_to {
            parts.push((init_from, init_to));
        }
        let mut rest = Vec::with_capacity(4);
        for &(a, e) in pending.iter() {
            if parts.is_empty() {
                break;
            }
            let (a, e) = self.inside(a, e - a);
            rest.clear();
            for &(from, to) in &parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            std::mem::swap(&mut parts, &mut rest);
        }
        pending.push(range);
        self.has_pending.store(true, Ordering::Release);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        if taken == 0 {
            return;
        }
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let is_wc = self.rebar != 0;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        if is_wc {
                            copy_from_wc(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                        } else {
                            std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                        }
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(
                parts
                    .into_iter()
                    .filter(|&(from, to)| from < to)
                    .map(|(from, to)| (data + from, snapshot + from, to - from, is_wc)),
            );
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from, is_wc);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            if differing.is_empty() {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
                return 0;
            }
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            let is_wc = self.rebar != 0;
            SCRATCH_TITLE.with(|title_cell| {
                let mut title_buf = title_cell.borrow_mut();
                if title_buf.len() < PIECE {
                    title_buf.resize(PIECE, 0);
                }
                SCRATCH_WC.with(|wc_cell| {
                    let mut wc_buf = wc_cell.borrow_mut();
                    if is_wc && wc_buf.len() < PIECE {
                        wc_buf.resize(PIECE, 0);
                    }
                    for (at, n) in differing {
                        let old = &before[at - from..at - from + n];
                        if is_wc {
                            // Copy WC GPU upload memory into L1-cached scratch ONCE
                            // via AVX2 streaming loads (`VMOVNTDQA`), diff and merge
                            // in cached memory, and write back only title-changed
                            // runs so PCIe never sees uncached byte reads or
                            // redundant stores.
                            let wc_slice = &mut wc_buf[..n];
                            // SAFETY: inside the ReBAR mapping; `n <= PIECE`.
                            unsafe { copy_from_wc(self.data.add(at), wc_slice.as_mut_ptr(), n) };
                            let Some(first) = first_difference(wc_slice, old) else {
                                continue;
                            };
                            let last = last_difference(wc_slice, old).unwrap_or(first);
                            let (at, n) = (at + first, last + 1 - first);
                            let host_orig = &wc_slice[first..=last];
                            let old = &old[first..=last];
                            let title = &mut title_buf[..n];
                            if !crate::host::read_into((start + at) as *const u8, title) {
                                continue;
                            }
                            // Merge GPU-written bytes into `title` without
                            // touching `host_orig`: after the merge, `title`
                            // holds the exact merged bytes (`host_orig` where
                            // GPU wrote, original `title` elsewhere), so
                            // `changed_runs(title, host_orig)` yields the exact
                            // title-modified runs to store back to WC memory.
                            let differs = merge_gpu_into_title(title, host_orig, old);
                            let wc_runs = changed_runs(title, host_orig);
                            for &(run_at, run_len) in &wc_runs {
                                // SAFETY: inside the ReBAR mapping; sequential WC store.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        title.as_ptr().add(run_at),
                                        self.data.add(at + run_at),
                                        run_len,
                                    );
                                }
                            }
                            if !wc_runs.is_empty() {
                                wc_sfence();
                            }
                            if differs {
                                crate::watch::writing(start + at, n);
                                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                                written += n;
                            }
                        } else {
                            // SAFETY: inside the mapping; only the submitting thread
                            // writes it.
                            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                            // From the first byte that differs to the last: title
                            // memory is read and written there only (a shader that sets
                            // a few words of a piece cost a read, a walk and a write of
                            // all 64 KiB of it).
                            let Some(first) = first_difference(host, old) else {
                                continue;
                            };
                            let last = last_difference(host, old).unwrap_or(first);
                            let (at, n) = (at + first, last + 1 - first);
                            let (host, old) = (&mut host[first..=last], &old[first..=last]);
                            let title = &mut title_buf[..n];
                            if !crate::host::read_into((start + at) as *const u8, title) {
                                continue;
                            }
                            if merge_sync_host(title, host, old) {
                                crate::watch::writing(start + at, n);
                                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                                written += n;
                            }
                        }
                    }
                });
            });
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            if self.rebar != 0 {
                copy_from_wc(self.data.add(at), now.as_mut_ptr(), n);
            } else {
                std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            }
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on the kept worker
    /// pool when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data, is_wc) = (self.start as usize, self.data as usize, self.rebar != 0);
        let before_ptr = before.as_ptr() as usize;
        let scan = move |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping and `before`, which nothing
                // writes while a submission's results are compared.
                let same = if is_wc {
                    unsafe {
                        slices_equal_wc(
                            (data + at) as *const u8,
                            (before_ptr + (at - from)) as *const u8,
                            n,
                        )
                    }
                } else {
                    unsafe {
                        std::slice::from_raw_parts((data + at) as *const u8, n)
                            == std::slice::from_raw_parts((before_ptr + (at - from)) as *const u8, n)
                    }
                };
                if !same {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        // Keep WC PCIe BAR scans single-threaded to avoid multi-core PCIe read thrashing.
        let threads = if is_wc { 1 } else { ((to - from) / PER_THREAD).clamp(1, 8) };
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every worker sees whole pieces.
        let cut = |i: usize| {
            if i == 0 {
                return from;
            }
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            at.saturating_sub((start + at) % PIECE).max(from)
        };
        let parts: Arc<Mutex<Vec<(usize, Vec<(usize, usize)>)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(threads)));
        let mut jobs: Vec<Job> = Vec::with_capacity(threads);
        for i in 0..threads {
            let (a, b) = (cut(i), cut(i + 1));
            if a >= b {
                continue;
            }
            let parts = parts.clone();
            jobs.push(Box::new(move || {
                let mine = scan(a, b);
                parts.lock().unwrap_or_else(|e| e.into_inner()).push((i, mine));
            }));
        }
        wide(jobs);
        let mut parts = std::mem::take(&mut *parts.lock().unwrap_or_else(|e| e.into_inner()));
        parts.sort_unstable_by_key(|p| p.0);
        parts.into_iter().flat_map(|(_, found)| found).collect()
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        self.has_pending.store(!pending.is_empty(), Ordering::Release);
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        self.has_pending.load(Ordering::Acquire)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => written.into_iter().any(|(page, write)| {
                page >= base && copied.get(page - base).is_none_or(|at| at == 0 || write > at)
            }),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        SCRATCH_TITLE.with(|title_cell| {
            let mut title_buf = title_cell.borrow_mut();
            for (a, e) in busy {
                // As the refresh itself asks (`update_free`): the journal if
                // the submission's asking covered the range, else the kernel,
                // else the guards. The journal alone said nothing for copies
                // of view memory, which is guarded and not write-watched, and
                // those were all of run 99's cases: the flights were settled
                // 4-5 times a frame for "0 pages looked at".
                let Some((now, written)) = crate::watch::writes(a, e, true) else {
                    return true;
                };
                let stale: Vec<usize> = {
                    let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    written
                        .into_iter()
                        .filter(|&(page, write)| page >= base && copied.get(page - base).is_none_or(|at| at == 0 || write > at))
                        .map(|(page, _)| page)
                        .collect()
                };
                if stale.is_empty() {
                    continue;
                }
                if title_buf.len() < PAGE {
                    title_buf.resize(PAGE, 0);
                }
                for page in stale {
                    let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                    if lo >= hi {
                        continue;
                    }
                    let n = hi - lo;
                    let title = &mut title_buf[..n];
                    // The lock of the copy's reference (the snapshot, or the
                    // CPU's side of a copy in the card's memory), from before
                    // title memory is read until the title's bytes are in: with
                    // the copy-back on its own thread ([`relay`]) that thread
                    // puts what the card wrote into title memory and into the
                    // reference under the same lock, and a page read before
                    // such a merge and compared after it would take the card's
                    // bytes for the title's.
                    let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                    // A page that is not there is left as it is, and looked at
                    // again next time.
                    if !crate::host::read_into(lo as *const u8, title) {
                        continue;
                    }
                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                    let at = lo - start;
                    if self.card.is_some() {
                        // SAFETY: inside the mapping of the copy's CPU side,
                        // which only the submitting thread writes.
                        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        let runs = changed_runs(title, held);
                        if runs.len() > *runs_left {
                            return true;
                        }
                        *runs_left -= runs.len();
                        for (r, m) in runs {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    } else {
                        let Some(snapshot) = before.as_mut() else {
                            return true;
                        };
                        // SAFETY: the snapshot's allocation holds the copy's
                        // bytes at the copy's offsets (`writing`), and these
                        // are in a declared range.
                        let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                        let runs = changed_runs(title, held);
                        for &(r, m) in &runs {
                            // SAFETY: inside the copy's mapping. The card may be
                            // writing the copy beside these bytes; they are
                            // written here and nothing around them.
                            unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                        if self.rebar != 0 && !runs.is_empty() {
                            wc_sfence();
                        }
                    }
                    drop(before);
                    // Taken, if the range holds all of the page that is in the
                    // copy; else the rest of it is the refresh's, which notes
                    // only whole pages, and the page is looked at again.
                    if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                        copied.set(page - base, now);
                    }
                }
            }
            false
        })
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if !self.has_pending.load(Ordering::Acquire) {
            return self.update_free(from, to);
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        let mode = self.watched.load(Ordering::Relaxed);
        if mode != NOT_WATCHED {
            let result = if mode == VIEW_GUARDED {
                crate::watch::writes_guarded(from, to).map(|(now, w)| (now, w, true))
            } else {
                crate::watch::writes_kind(from, to, true)
            };
            if let Some((now, written, is_view)) = result {
                let next = if is_view { VIEW_GUARDED } else { 1 };
                if mode != next {
                    self.watched.store(next, Ordering::Relaxed);
                }
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1") || crate::host::walls_trace()) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let is_wc = self.rebar != 0;
        let mut title = [0u8; SAMPLE];
        let mut wc_sample = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = if is_wc {
                    unsafe { copy_from_wc(self.data.add(at - start).cast_const(), wc_sample.as_mut_ptr(), n) };
                    &wc_sample[..n]
                } else {
                    unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) }
                };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer. Plans coalesced runs under
    /// `self.copied`, then releases the mutex before executing the copies.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let runs = {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.plan_stale(base, start, end, from, to, now, written)
        };
        let mut bytes = 0;
        for (run_from, run_to) in runs.iter() {
            bytes += self.copy(run_from, run_to);
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if self.rebar != 0 && to > from {
            wc_sfence();
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
        card.has_dirty.store(true, Ordering::Release);
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            if merge_sync_card(title, gpu, old) {
                crate::watch::writing(start + at, n);
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        // Check `has_dirty` FIRST, then `has_sent`: `card_dirty` sets
        // `has_sent` (Release) before clearing `has_dirty` (Release), so
        // observing `has_dirty == false` guarantees observing `has_sent == true`
        // for any range transitioning from `dirty` to `sent`.
        let skip: Vec<(usize, usize)> =
            if !card.has_dirty.load(Ordering::Acquire) && !card.has_sent.load(Ordering::Acquire) {
                Vec::new()
            } else {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                dirty
                    .iter()
                    .copied()
                    .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                    .filter(|&(a, b)| a < at + n && b > at)
                    .collect()
            };
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::watch::writing(start + at, n);
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: RwLock<BTreeMap<(u64, usize), Entry>> = RwLock::new(BTreeMap::new());

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

fn buffers_write() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.write().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers_write().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers_write().remove(&(start, resource));
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers_read()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers_write();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read()
        .iter()
        .map(|(key, entry)| (key.0, entry.end - key.0, false))
        .collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    // Shared-read fast path: 99.99% of calls either hit an already-created
    // buffer shadow or miss into `loose()`, so parallel recording threads never
    // serialize on an exclusive lock.
    {
        let map = buffers_read();
        let found = map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end);
        match found {
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
            Some((_, entry)) if entry.shadow.is_some() => {
                return entry.shadow.clone();
            }
            Some(_) => {}
        }
    }
    let mut map = buffers_write();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && len >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        if !big && let Some(shadow) = rebar_copy(crate::rebar::Kind::Buffer, key.0, len, size, false) {
            entry.shadow = Some(shadow);
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(len, size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    has_dirty: AtomicBool::new(false),
                    sent: Mutex::default(),
                    has_sent: AtomicBool::new(false),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// A host copy in CPU-visible card memory (`D3D12_HEAP_TYPE_GPU_UPLOAD`,
/// `crate::rebar`), if enabled, supported and within budget.
fn rebar_copy(kind: crate::rebar::Kind, start: u64, len: u64, size: u64, window: bool) -> Option<Arc<Shadow>> {
    // Small title buffers (`Kind::Buffer`) are snapshotted and diffed on the
    // CPU after shader writes whenever `VICTORY_COPY_BACK_ONLY_READBACK` is
    // off. Reading write-combined GPU upload memory over PCIe from the CPU
    // stalls copy-back severely (~80 ms/frame for 17 MB), so keep small
    // buffers in system RAM unless readback filtering is active.
    let cpu_read = crate::readback::tagged_in(start, len)
        || (kind == crate::rebar::Kind::Buffer && !crate::readback::enabled());
    if cpu_read && kind == crate::rebar::Kind::Buffer && !crate::readback::enabled() && crate::rebar::level() >= 2 {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "Resizable BAR: keeping small title buffers in system RAM because VICTORY_COPY_BACK_ONLY_READBACK=0 (avoids uncached PCIe CPU readback stalls)"
            );
        });
    }
    let reservation = crate::rebar::reserve(kind, size, cpu_read)?;
    let (buffer, data) = host()?.gpu_upload_buffer(size)?;
    let reserved = reservation.commit();
    crate::host::name(&buffer, || {
        format!(
            "title {} {start:#x}+{len:#x} (rebar)",
            if window { "memory" } else { "buffer" }
        )
    });
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = window;
    shadow.rebar = reserved;
    Some(Arc::new(shadow))
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: RwLock<BTreeMap<(u64, u64), Arc<Shadow>>> = RwLock::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let resolve = |map: &BTreeMap<(u64, u64), Arc<Shadow>>| -> (u64, u64) {
        if unsized_ask && tails() {
            let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
            longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
        } else if any && fine() && !unsized_ask {
            // The windows that begin in the ask's cell at or below the ask (one
            // that begins in a cell below and reaches this far is passed over:
            // the ask then makes its own).
            let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
            holder(there, address, end).unwrap_or(named)
        } else {
            named
        }
    };
    // Under the lock (shared or exclusive), with the handing out: see `let_go_idle`
    // (which takes `LOOSE.write()`).
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    // Shared-read fast path: 99.99% of lookups hit an existing window.
    {
        let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        let (start, len) = resolve(&map);
        if let Some(shadow) = map.get(&(start, len)) {
            note(shadow);
            return Some(shadow.clone());
        }
    }
    let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
    let (start, len) = resolve(&map);
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    if let Some(shadow) = rebar_copy(crate::rebar::Kind::Window, start, len, len, true) {
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the one just made is the CPU's side.
    let (buffer, card) = match window_card_for(len) {
        Some(on_card) => {
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card {
                staging: buffer,
                before: Mutex::default(),
                dirty: Mutex::default(),
                has_dirty: AtomicBool::new(false),
                sent: Mutex::default(),
                has_sent: AtomicBool::new(false),
                size: len,
            };
            (on_card, Some(card))
        }
        None => (buffer, None),
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }

    #[test]
    fn copied_pages_tracks_uncopied_count_and_fast_path_condition() {
        let mut pages = super::CopiedPages::new(4);
        assert!(!pages.all_copied());
        assert_eq!(pages.copied_count(), 0);
        pages.set(0, 10);
        pages.set(0, 20); // updating an already-copied page keeps count stable
        assert_eq!(pages.copied_count(), 1);
        assert!(!pages.all_copied());
        pages.set(1, 10);
        pages.set(2, 10);
        pages.set(3, 10);
        assert!(pages.all_copied());
        assert_eq!(pages.copied_count(), 4);
        assert_eq!(pages.get(2), Some(10));
        assert_eq!(pages.get(4), None);
        pages.set(1, 0);
        assert!(!pages.all_copied());
        assert_eq!(pages.copied_count(), 3);
    }

    /// Reference oracle matching the original O(range) page-by-page `copy_stale`
    /// loop, used to prove `CopiedPages::plan_stale` is bit-for-bit identical
    /// across all alignments, partial pages, duplicates, and word boundaries.
    #[allow(clippy::too_many_arguments)]
    fn oracle_plan_stale(
        at_vec: &mut [u64],
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
    ) -> Vec<(usize, usize)> {
        use super::PAGE;
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let mut writes = written.iter().peekable();
        let mut out = Vec::new();
        let mut run: Option<usize> = None;
        for page in first_page..=last {
            while writes.peek().is_some_and(|&&(p, _)| p < page) {
                writes.next();
            }
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    // Consume any duplicates for the same page, keeping max write seq.
                    let mut best = w;
                    while let Some(&&(p2, w2)) = writes.peek() {
                        if p2 != page {
                            break;
                        }
                        writes.next();
                        best = best.max(w2);
                    }
                    best
                }
                _ => 0,
            };
            let cur = at_vec[page - base];
            if cur == 0 || write > cur {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    at_vec[page - base] = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                out.push(((first * PAGE).max(from), (page * PAGE).min(to)));
            }
        }
        if let Some(first) = run {
            out.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to)));
        }
        out
    }

    #[test]
    fn plan_stale_matches_oracle_across_boundaries_and_partial_init() {
        use super::{CopiedPages, PAGE};
        // Span 3 bitmask words (> 128 pages) with unaligned start & end.
        let start = 10 * PAGE + 0x1234;
        let len = 140 * PAGE + 0x5678;
        let end = start + len;
        let base = start / PAGE;
        let pages_total = (end.div_ceil(PAGE)) - base;

        let mut cp = CopiedPages::new(pages_total);
        let mut oracle = vec![0u64; pages_total];

        // Partially initialize several subranges across 64-page word boundaries.
        for idx in (0..64).chain(65..128).chain(130..135) {
            cp.set(idx, 10);
            oracle[idx] = 10;
        }
        assert!(!cp.all_copied());
        assert!(cp.range_all_copied(0, 63));
        assert!(!cp.range_all_copied(0, 64));
        assert!(cp.range_all_copied(65, 127));
        assert!(!cp.range_all_copied(64, 127));
        assert_eq!(cp.next_uncopied(0, pages_total - 1), Some(64));
        assert_eq!(cp.next_uncopied(65, pages_total - 1), Some(128));

        let cases: &[((usize, usize), u64, &[(usize, u64)])] = &[
            // Fully initialized subrange, no writes -> empty runs.
            ((start, start + 60 * PAGE), 20, &[]),
            // Fully initialized subrange with stale and fresh writes + out-of-range writes.
            (
                (start + 2 * PAGE, start + 62 * PAGE),
                21,
                &[
                    (base, 99),
                    (base + 5, 10),
                    (base + 6, 15),
                    (base + 7, 16),
                    (base + 20, 19),
                    (base + 139, 99),
                ],
            ),
            // Partial page edges (should copy bytes without marking partial pages copied).
            (
                (start + 64 * PAGE + 128, start + 64 * PAGE + 4096),
                22,
                &[(base + 64, 22)],
            ),
            // Mixed uncopied + written + already-copied across all 3 words.
            (
                (start + 50 * PAGE, start + 136 * PAGE),
                25,
                &[
                    (base + 52, 23),
                    (base + 63, 24),
                    (base + 64, 24),
                    (base + 65, 24),
                    (base + 127, 25),
                    (base + 128, 25),
                    (base + 131, 5),
                ],
            ),
            // Whole buffer sweep, including tail partial page.
            ((start, end), 30, &[(base + 10, 29), (base + 140, 29)]),
        ];

        for &((from, to), now, written) in cases {
            let got: Vec<(usize, usize)> = cp
                .plan_stale(base, start, end, from, to, now, written)
                .iter()
                .collect();
            let want = oracle_plan_stale(&mut oracle, base, start, end, from, to, now, written);
            assert_eq!(got, want, "range {from:#x}..{to:#x} now={now}");
            assert_eq!(cp.at, oracle, "copied timestamps diverged at now={now}");
            let expected_uncopied = oracle.iter().filter(|&&v| v == 0).count();
            assert_eq!(cp.uncopied, expected_uncopied);
        }
        assert!(cp.all_copied());
    }

    #[test]
    fn simd_diff_merge_and_wc_helpers_match_scalar_oracles() {
        use super::{
            changed_runs, changed_runs_scalar, copy_from_wc, first_difference,
            first_difference_scalar, last_difference, last_difference_scalar,
            merge_gpu_into_title, merge_gpu_into_title_scalar, merge_piece, merge_piece_scalar,
            merge_sync_card, merge_sync_card_scalar, merge_sync_host, merge_sync_host_scalar,
            slices_equal_wc,
        };

        // Test lengths covering <32, exact 32, unaligned tails, and multi-vector spans.
        for len in [0usize, 1, 7, 15, 31, 32, 33, 63, 64, 65, 95, 96, 129, 257, 1024] {
            let old: Vec<u8> = (0..len).map(|i| ((i * 37 + 11) & 0xff) as u8).collect();
            let mut gpu = old.clone();
            let mut title = old.clone();
            for i in 0..len {
                if i % 19 == 0 || i == len.saturating_sub(1) || ((30..66).contains(&i) && i % 3 == 0) {
                    gpu[i] ^= 0x5a;
                }
                if i % 23 == 0 || ((16..80).contains(&i) && i % 5 == 0) {
                    title[i] ^= 0xa5;
                }
            }

            // first_difference / last_difference
            assert_eq!(first_difference(&gpu, &old), first_difference_scalar(&gpu, &old), "len={len}");
            assert_eq!(last_difference(&gpu, &old), last_difference_scalar(&gpu, &old), "len={len}");
            assert_eq!(first_difference(&old, &old), None);
            assert_eq!(last_difference(&old, &old), None);

            // changed_runs
            assert_eq!(changed_runs(&gpu, &old), changed_runs_scalar(&gpu, &old), "len={len}");
            assert_eq!(changed_runs(&old, &old), Vec::<(usize, usize)>::new());

            // merge_piece
            let (mut h1, mut t1) = (old.clone(), title.clone());
            let (mut h2, mut t2) = (old.clone(), title.clone());
            assert_eq!(
                merge_piece(&gpu, &mut h1, &mut t1),
                merge_piece_scalar(&gpu, &mut h2, &mut t2),
                "merge_piece ret len={len}"
            );
            assert_eq!((&h1, &t1), (&h2, &t2), "merge_piece bytes len={len}");

            // merge_sync_host
            let (mut host1, mut t1) = (gpu.clone(), title.clone());
            let (mut host2, mut t2) = (gpu.clone(), title.clone());
            assert_eq!(
                merge_sync_host(&mut t1, &mut host1, &old),
                merge_sync_host_scalar(&mut t2, &mut host2, &old),
                "merge_sync_host ret len={len}"
            );
            assert_eq!((&host1, &t1), (&host2, &t2), "merge_sync_host bytes len={len}");

            // merge_sync_card
            let (mut old1, mut t1) = (old.clone(), title.clone());
            let (mut old2, mut t2) = (old.clone(), title.clone());
            assert_eq!(
                merge_sync_card(&mut t1, &gpu, &mut old1),
                merge_sync_card_scalar(&mut t2, &gpu, &mut old2),
                "merge_sync_card ret len={len}"
            );
            assert_eq!((&old1, &t1), (&old2, &t2), "merge_sync_card bytes len={len}");

            // merge_gpu_into_title + changed_runs WC writeback equivalence
            let (mut t1, mut t2) = (title.clone(), title.clone());
            assert_eq!(
                merge_gpu_into_title(&mut t1, &gpu, &old),
                merge_gpu_into_title_scalar(&mut t2, &gpu, &old),
                "merge_gpu_into_title ret len={len}"
            );
            assert_eq!(t1, t2);
            let mut wc_reconstructed = gpu.clone();
            for (r, m) in changed_runs(&t1, &gpu) {
                wc_reconstructed[r..r + m].copy_from_slice(&t1[r..r + m]);
            }
            assert_eq!(wc_reconstructed, host1, "WC writeback must equal merge_sync_host");

            // copy_from_wc / slices_equal_wc across misaligned source offsets 0..32
            for shift in [0usize, 1, 7, 16, 31] {
                if shift > len {
                    continue;
                }
                let sub = &gpu[shift..];
                let mut copied = vec![0xeeu8; sub.len()];
                // SAFETY: valid non-overlapping slices of length `sub.len()`.
                unsafe { copy_from_wc(sub.as_ptr(), copied.as_mut_ptr(), sub.len()) };
                assert_eq!(&copied[..], sub, "copy_from_wc len={len} shift={shift}");
                // SAFETY: valid readable slices of length `sub.len()`.
                assert!(unsafe { slices_equal_wc(sub.as_ptr(), copied.as_ptr(), sub.len()) });
                if !copied.is_empty() {
                    let flip = copied.len() / 2;
                    copied[flip] ^= 1;
                    // SAFETY: valid readable slices of length `sub.len()`.
                    assert!(!unsafe { slices_equal_wc(sub.as_ptr(), copied.as_ptr(), sub.len()) });
                }
            }
        }
    }

    #[test]
    fn nested_wide_runs_without_deadlock() {
        use super::{Job, wide};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let sum = Arc::new(AtomicUsize::new(0));
        let outer: Vec<Job> = (0..8)
            .map(|_| {
                let sum = sum.clone();
                Box::new(move || {
                    let inner: Vec<Job> = (0..4)
                        .map(|_| {
                            let sum = sum.clone();
                            Box::new(move || {
                                sum.fetch_add(1, Ordering::Relaxed);
                            }) as Job
                        })
                        .collect();
                    wide(inner);
                }) as Job
            })
            .collect();
        wide(outer);
        assert_eq!(sum.load(Ordering::Relaxed), 32);
    }

    #[test]
    fn wide_propagates_worker_panic_without_deadlocking_pool() {
        use super::{Job, wide};
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let panic_res = std::panic::catch_unwind(|| {
            let jobs: Vec<Job> = (0..8)
                .map(|i| {
                    Box::new(move || {
                        if i == 3 {
                            panic!("synthetic wide worker panic");
                        }
                    }) as Job
                })
                .collect();
            wide(jobs);
        });
        assert!(panic_res.is_err(), "worker panic must propagate to caller");

        // Pool must remain healthy and non-deadlocked on subsequent batches.
        let ran = Arc::new(AtomicUsize::new(0));
        let jobs: Vec<Job> = (0..16)
            .map(|_| {
                let ran = ran.clone();
                Box::new(move || {
                    ran.fetch_add(1, Ordering::Relaxed);
                }) as Job
            })
            .collect();
        wide(jobs);
        assert_eq!(ran.load(Ordering::Relaxed), 16);
    }

    #[test]
    fn rebar_wc_write_read_compare_roundtrip() {
        use super::{PIECE, copy_from_wc, slices_equal_wc, wc_sfence};
        let Some(host) = crate::host::host() else {
            return;
        };
        if !crate::rebar::supported() {
            return;
        }
        const LEN: usize = 2 * PIECE; // 128 KiB
        let Some((_buf, wc_ptr)) = host.gpu_upload_buffer(LEN as u64) else {
            return;
        };
        let pattern: Vec<u8> = (0..LEN).map(|i| ((i * 29 + (i >> 8) * 7 + 3) & 0xff) as u8).collect();
        // Store into WC VRAM, drain WCBs, then immediately read & compare via VMOVNTDQA.
        // SAFETY: mapped 128 KiB ReBAR buffer.
        unsafe { std::ptr::copy_nonoverlapping(pattern.as_ptr(), wc_ptr, LEN) };
        wc_sfence();

        let mut out = vec![0u8; LEN];
        for (offset, sub_len) in [
            (0usize, LEN),
            (0, 17),
            (1, 31),
            (1, 300),
            (17, 4096 + 53),
            (31, PIECE + 129),
            (32, PIECE),
            (LEN - 33, 33),
        ] {
            out[..sub_len].fill(0);
            // SAFETY: sub-range lies strictly within the 128 KiB mapped ReBAR buffer.
            unsafe { copy_from_wc(wc_ptr.add(offset), out.as_mut_ptr(), sub_len) };
            assert_eq!(&out[..sub_len], &pattern[offset..offset + sub_len], "copy_from_wc off={offset} len={sub_len}");
            // SAFETY: valid readable ranges of length `sub_len`.
            assert!(
                unsafe { slices_equal_wc(wc_ptr.add(offset), pattern.as_ptr().add(offset), sub_len) },
                "slices_equal_wc eq off={offset} len={sub_len}"
            );
            // Flip first, mid, and last bytes in the WC buffer and verify immediate detection.
            for flip in [0usize, sub_len / 2, sub_len - 1] {
                let target = offset + flip;
                // SAFETY: inside mapped ReBAR buffer.
                unsafe {
                    wc_ptr.add(target).write(pattern[target] ^ 0xff);
                }
                wc_sfence();
                // SAFETY: valid readable ranges of length `sub_len`.
                assert!(
                    !unsafe { slices_equal_wc(wc_ptr.add(offset), pattern.as_ptr().add(offset), sub_len) },
                    "slices_equal_wc missed flip off={offset} len={sub_len} flip={flip}"
                );
                // Restore original byte and confirm equality returns immediately.
                // SAFETY: inside mapped ReBAR buffer.
                unsafe {
                    wc_ptr.add(target).write(pattern[target]);
                }
                wc_sfence();
                // SAFETY: valid readable ranges of length `sub_len`.
                assert!(
                    unsafe { slices_equal_wc(wc_ptr.add(offset), pattern.as_ptr().add(offset), sub_len) },
                    "slices_equal_wc stale after restore off={offset} len={sub_len} flip={flip}"
                );
            }
        }
    }

    /// End-to-end release benchmarks comparing BEFORE (baseline scalar/unindexed)
    /// vs AFTER (bitmask + AVX2 + VMOVNTDQA) on CPU and real D3D12 ReBAR /
    /// shared buffers. Run with:
    /// `cargo test --release -p victory-gpu --locked -- --ignored --nocapture bench_shadow_and_rebar`
    #[test]
    #[ignore = "benchmark"]
    fn bench_shadow_and_rebar() {
        use super::{
            CopiedPages, PAGE, PIECE, changed_runs, changed_runs_scalar, copy_from_wc,
            first_difference, first_difference_scalar, last_difference, last_difference_scalar,
            merge_piece, merge_sync_host, slices_equal_wc,
        };
        use std::hint::black_box;
        use std::time::Instant;

        // 1. `copy_stale` planning on a partially-initialized 174 MiB buffer
        //    (2,784 pages; 2,500 initialized, 284 uncopied at the tail).
        const PAGES: usize = 2784;
        let base = 0x1000usize;
        let start = base * PAGE;
        let end = start + PAGES * PAGE;
        let query_from = start;
        let query_to = start + 2400 * PAGE; // 150 MiB initialized subrange
        let dirty_8: Vec<(usize, u64)> = (0..8).map(|i| (base + i * 250 + 17, 50)).collect();

        let mut cp = CopiedPages::new(PAGES);
        let mut oracle_at = vec![0u64; PAGES];
        for i in 0..2500 {
            cp.set(i, 10);
            oracle_at[i] = 10;
        }
        const PLAN_ITERS: usize = 10_000;
        // Baseline O(range) scan when buffer is not 100% initialized.
        let t0 = Instant::now();
        let mut acc = 0usize;
        for iter in 0..PLAN_ITERS {
            let now = 100 + iter as u64;
            let runs = oracle_plan_stale(
                &mut oracle_at,
                base,
                start,
                end,
                query_from,
                query_to,
                now,
                &[],
            );
            acc += runs.len();
        }
        let before_clean = t0.elapsed();
        black_box(acc);

        let t0 = Instant::now();
        let mut acc = 0usize;
        for iter in 0..PLAN_ITERS {
            let now = 100 + iter as u64;
            let runs = cp.plan_stale(base, start, end, query_from, query_to, now, &[]);
            acc += runs.iter().count();
        }
        let after_clean = t0.elapsed();
        black_box(acc);

        // Dirty 8 pages in the 150 MiB subrange.
        let t0 = Instant::now();
        let mut acc = 0usize;
        for iter in 0..PLAN_ITERS {
            let now = 200_000 + iter as u64 * 2;
            let w: Vec<(usize, u64)> = dirty_8.iter().map(|&(p, _)| (p, now - 1)).collect();
            let runs = oracle_plan_stale(
                &mut oracle_at,
                base,
                start,
                end,
                query_from,
                query_to,
                now,
                &w,
            );
            acc += runs.len();
        }
        let before_dirty = t0.elapsed();
        black_box(acc);

        let t0 = Instant::now();
        let mut acc = 0usize;
        for iter in 0..PLAN_ITERS {
            let now = 200_000 + iter as u64 * 2;
            let w: Vec<(usize, u64)> = dirty_8.iter().map(|&(p, _)| (p, now - 1)).collect();
            let runs = cp.plan_stale(base, start, end, query_from, query_to, now, &w);
            acc += runs.iter().count();
        }
        let after_dirty = t0.elapsed();
        black_box(acc);

        println!(
            "[BENCH] copy_stale 150MiB clean (10k iters): before={before_clean:?}, after={after_clean:?} ({:.1}x)",
            before_clean.as_secs_f64() / after_clean.as_secs_f64()
        );
        println!(
            "[BENCH] copy_stale 150MiB 8-dirty (10k iters): before={before_dirty:?}, after={after_dirty:?} ({:.1}x)",
            before_dirty.as_secs_f64() / after_dirty.as_secs_f64()
        );

        // 2. Copy-back piece diff + merge (970 x 64 KiB pieces ~= 60 MiB/frame,
        //    sparse shader writes in the middle and edges of each piece).
        //    Keep buffer resets outside the timed merge loops so setup memcpy
        //    does not dilute the actual merge measurement.
        const PIECES: usize = 970;
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 13 + 7) as u8).collect();
        let mut gpu_mid = old.clone();
        for at in 28 * 1024..36 * 1024 {
            gpu_mid[at] ^= 0x3c;
        }
        let mut gpu_sparse = old.clone();
        for at in (64..320).chain(PIECE - 320..PIECE - 64) {
            gpu_sparse[at] ^= 0x3c;
        }

        // first_difference + last_difference on mid-piece shader writes (28 KiB..36 KiB)
        let t0 = Instant::now();
        let mut s = 0usize;
        for _ in 0..PIECES * 4 {
            s += first_difference_scalar(black_box(&gpu_mid), black_box(&old)).unwrap_or(0);
            s += last_difference_scalar(black_box(&gpu_mid), black_box(&old)).unwrap_or(0);
        }
        let before_diff = t0.elapsed();
        black_box(s);

        let t0 = Instant::now();
        let mut s = 0usize;
        for _ in 0..PIECES * 4 {
            s += first_difference(black_box(&gpu_mid), black_box(&old)).unwrap_or(0);
            s += last_difference(black_box(&gpu_mid), black_box(&old)).unwrap_or(0);
        }
        let after_diff = t0.elapsed();
        black_box(s);
        println!(
            "[BENCH] first+last_difference mid-piece (3880x64KiB): before={before_diff:?}, after={after_diff:?} ({:.1}x)",
            before_diff.as_secs_f64() / after_diff.as_secs_f64()
        );

        // merge_piece (relay_merge / card_merge_relay) — baseline branchless byte loop vs AVX2
        let baseline_merge_piece = |now: &[u8], held: &mut [u8], title: &mut [u8]| -> bool {
            let mut differs = 0u8;
            for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
                let wrote = if *n != *h { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*n & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *h = (*h & !wrote) | (*n & wrote);
            }
            differs != 0
        };
        let mut held = old.clone();
        let mut title = old.clone();
        let t0 = Instant::now();
        let mut hits = 0usize;
        for _ in 0..PIECES {
            // Reset only the 512 changed bytes so setup stays O(512) instead of O(128 KiB).
            held[64..320].copy_from_slice(&old[64..320]);
            held[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            title[64..320].copy_from_slice(&old[64..320]);
            title[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            if baseline_merge_piece(black_box(&gpu_sparse), black_box(&mut held), black_box(&mut title)) {
                hits += 1;
            }
        }
        let before_mp = t0.elapsed();
        black_box(hits);

        held.copy_from_slice(&old);
        title.copy_from_slice(&old);
        let t0 = Instant::now();
        let mut hits = 0usize;
        for _ in 0..PIECES {
            held[64..320].copy_from_slice(&old[64..320]);
            held[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            title[64..320].copy_from_slice(&old[64..320]);
            title[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            if merge_piece(black_box(&gpu_sparse), black_box(&mut held), black_box(&mut title)) {
                hits += 1;
            }
        }
        let after_mp = t0.elapsed();
        black_box(hits);
        println!(
            "[BENCH] merge_piece sparse (970x64KiB ~= 60MiB): before={before_mp:?}, after={after_mp:?} ({:.1}x)",
            before_mp.as_secs_f64() / after_mp.as_secs_f64()
        );

        // merge_sync_host (copy_back_with) — baseline byte loop vs AVX2
        let baseline_sync = |title: &mut [u8], host: &mut [u8], old: &[u8]| -> bool {
            let mut differs = 0u8;
            for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                let gpu = if *h != *o { 0xffu8 } else { 0 };
                let merged = (*t & !gpu) | (*h & gpu);
                differs |= merged ^ *t;
                *t = merged;
                *h = merged;
            }
            differs != 0
        };
        let mut host_buf = gpu_sparse.clone();
        title.copy_from_slice(&old);
        let t0 = Instant::now();
        let mut hits = 0usize;
        for _ in 0..PIECES {
            title[64..320].copy_from_slice(&old[64..320]);
            title[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            if baseline_sync(black_box(&mut title), black_box(&mut host_buf), black_box(&old)) {
                hits += 1;
            }
        }
        let before_sync = t0.elapsed();
        black_box(hits);

        host_buf.copy_from_slice(&gpu_sparse);
        title.copy_from_slice(&old);
        let t0 = Instant::now();
        let mut hits = 0usize;
        for _ in 0..PIECES {
            title[64..320].copy_from_slice(&old[64..320]);
            title[PIECE - 320..PIECE - 64].copy_from_slice(&old[PIECE - 320..PIECE - 64]);
            if merge_sync_host(black_box(&mut title), black_box(&mut host_buf), black_box(&old)) {
                hits += 1;
            }
        }
        let after_sync = t0.elapsed();
        black_box(hits);
        println!(
            "[BENCH] copy_back merge_sync_host sparse (970x64KiB ~= 60MiB): before={before_sync:?}, after={after_sync:?} ({:.1}x)",
            before_sync.as_secs_f64() / after_sync.as_secs_f64()
        );

        // changed_runs (merge_under_writes)
        let t0 = Instant::now();
        let mut total_runs = 0usize;
        for _ in 0..PIECES * 2 {
            total_runs += changed_runs_scalar(black_box(&gpu_sparse), black_box(&old)).len();
        }
        let before_cr = t0.elapsed();
        black_box(total_runs);

        let t0 = Instant::now();
        let mut total_runs = 0usize;
        for _ in 0..PIECES * 2 {
            total_runs += changed_runs(black_box(&gpu_sparse), black_box(&old)).len();
        }
        let after_cr = t0.elapsed();
        black_box(total_runs);
        println!(
            "[BENCH] changed_runs (1940x64KiB): before={before_cr:?}, after={after_cr:?} ({:.1}x)",
            before_cr.as_secs_f64() / after_cr.as_secs_f64()
        );

        // 3. Real GPU hardware ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) vs shared
        //    heap (`D3D12_HEAP_TYPE_CUSTOM` L0 WB) read/write/compare/copy_back
        //    throughput on this exact GPU.
        let Some(host) = crate::host::host() else {
            println!("[BENCH] no host D3D12 device");
            return;
        };
        const BUF_LEN: usize = 16 << 20; // 16 MiB
        let src = vec![0x5au8; BUF_LEN];
        let mut dst = vec![0u8; BUF_LEN];
        let mut cmp_ref = vec![0x5au8; BUF_LEN];

        if let Some((_shared, shared_ptr)) = host.shared_buffer(BUF_LEN as u64) {
            crate::host::copy_mapped(shared_ptr, src.as_ptr(), BUF_LEN, false);
            let t0 = Instant::now();
            for _ in 0..8 {
                crate::host::copy_mapped(shared_ptr, src.as_ptr(), BUF_LEN, false);
            }
            let shared_write = t0.elapsed();
            let t0 = Instant::now();
            for _ in 0..8 {
                // SAFETY: mapped 16 MiB shared buffer.
                unsafe { std::ptr::copy_nonoverlapping(shared_ptr, dst.as_mut_ptr(), BUF_LEN) };
                black_box(dst[0]);
            }
            let shared_read = t0.elapsed();
            println!(
                "[BENCH] shared_buffer 128MiB: write={shared_write:?} ({:.2} GB/s), read={shared_read:?} ({:.2} GB/s)",
                (8 * BUF_LEN) as f64 / shared_write.as_secs_f64() / 1e9,
                (8 * BUF_LEN) as f64 / shared_read.as_secs_f64() / 1e9,
            );
        }

        if crate::rebar::supported()
            && let Some((_rebar, rebar_ptr)) = host.gpu_upload_buffer(BUF_LEN as u64)
        {
            // Write via copy_mapped (NtReadVirtualMemory directly to WC)
            crate::host::copy_mapped(rebar_ptr, src.as_ptr(), BUF_LEN, false);
            let t0 = Instant::now();
            for _ in 0..8 {
                crate::host::copy_mapped(rebar_ptr, src.as_ptr(), BUF_LEN, false);
            }
            let rebar_write_nt = t0.elapsed();

            // Write via direct copy_nonoverlapping to WC
            let t0 = Instant::now();
            for _ in 0..8 {
                // SAFETY: mapped 16 MiB ReBAR buffer.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), rebar_ptr, BUF_LEN) };
            }
            let rebar_write_memcpy = t0.elapsed();

            // Read BEFORE: normal copy_nonoverlapping from WC (uncached loads)
            let t0 = Instant::now();
            // SAFETY: mapped 16 MiB ReBAR buffer.
            unsafe { std::ptr::copy_nonoverlapping(rebar_ptr, dst.as_mut_ptr(), BUF_LEN) };
            black_box(dst[0]);
            let rebar_read_before = t0.elapsed();

            // Read AFTER: 4-cache-line unrolled AVX2 VMOVNTDQA streaming loads
            let t0 = Instant::now();
            // SAFETY: mapped 16 MiB ReBAR buffer.
            unsafe { copy_from_wc(rebar_ptr, dst.as_mut_ptr(), BUF_LEN) };
            black_box(dst[0]);
            let rebar_read_after = t0.elapsed();
            assert_eq!(dst[0], 0x5a);

            // Read AFTER (wide multi-threaded VMOVNTDQA via copy_wide)
            let t0 = Instant::now();
            // SAFETY: mapped 16 MiB ReBAR buffer.
            unsafe { super::copy_wide(rebar_ptr, dst.as_mut_ptr(), BUF_LEN, true) };
            black_box(dst[0]);
            let rebar_read_wide = t0.elapsed();

            // Compare BEFORE (`slice == slice`) vs AFTER (branchless-per-4KiB `slices_equal_wc`)
            cmp_ref.fill(0x5a);
            let t0 = Instant::now();
            // SAFETY: mapped 16 MiB ReBAR buffer.
            let eq_before = black_box(unsafe { std::slice::from_raw_parts(rebar_ptr, BUF_LEN) } == &cmp_ref[..]);
            let rebar_cmp_before = t0.elapsed();
            assert!(eq_before);

            let t0 = Instant::now();
            // SAFETY: mapped 16 MiB ReBAR buffer.
            let eq_after = black_box(unsafe { slices_equal_wc(rebar_ptr, cmp_ref.as_ptr(), BUF_LEN) });
            let rebar_cmp_after = t0.elapsed();
            assert!(eq_after);

            // Compare AFTER (wide multi-threaded `find_pieces_wc`)
            let t0 = Instant::now();
            let pieces_after = super::find_pieces_wc(vec![(
                rebar_ptr as usize,
                cmp_ref.as_ptr() as usize,
                0,
                0,
                BUF_LEN,
                true,
            )]);
            let rebar_cmp_wide = t0.elapsed();
            assert!(pieces_after[0].is_empty());

            // Simulate ReBAR copy-back of 16 x 64 KiB (1 MiB) differing pieces:
            // BEFORE (direct WC first_diff + last_diff + byte merge loop on WC) vs
            // AFTER (single VMOVNTDQA read into L1 + L1 diff/merge + changed_runs storeback)
            const CB_PIECES: usize = 16;
            let mut old_1m = vec![0x5au8; CB_PIECES * PIECE];
            let mut title_1m = old_1m.clone();
            for p in 0..CB_PIECES {
                old_1m[p * PIECE + 64] ^= 0x11;
                old_1m[p * PIECE + PIECE - 65] ^= 0x22;
            }
            let t0 = Instant::now();
            for p in 0..CB_PIECES {
                let off = p * PIECE;
                // SAFETY: mapped 16 MiB ReBAR buffer.
                let host_s = unsafe { std::slice::from_raw_parts_mut(rebar_ptr.add(off), PIECE) };
                let old_s = &old_1m[off..off + PIECE];
                let first = first_difference_scalar(host_s, old_s).unwrap();
                let last = last_difference_scalar(host_s, old_s).unwrap();
                let _ = baseline_sync(
                    &mut title_1m[off + first..=off + last],
                    &mut host_s[first..=last],
                    &old_s[first..=last],
                );
            }
            let rebar_cb_before = t0.elapsed();

            // Reset title_1m and rebar bytes cleanly
            title_1m.copy_from_slice(&old_1m);
            // SAFETY: mapped 16 MiB ReBAR buffer.
            unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), rebar_ptr, CB_PIECES * PIECE) };
            let mut wc_scratch = vec![0u8; PIECE];
            let t0 = Instant::now();
            for p in 0..CB_PIECES {
                let off = p * PIECE;
                // SAFETY: mapped 16 MiB ReBAR buffer.
                unsafe { copy_from_wc(rebar_ptr.add(off), wc_scratch.as_mut_ptr(), PIECE) };
                let old_s = &old_1m[off..off + PIECE];
                let first = first_difference(&wc_scratch, old_s).unwrap();
                let last = last_difference(&wc_scratch, old_s).unwrap();
                let host_orig = &wc_scratch[first..=last];
                let t_slice = &mut title_1m[off + first..=off + last];
                let _ = super::merge_gpu_into_title(t_slice, host_orig, &old_s[first..=last]);
                let runs = changed_runs(t_slice, host_orig);
                for &(run_at, run_len) in &runs {
                    // SAFETY: sequential WC write of changed run only.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            t_slice.as_ptr().add(run_at),
                            rebar_ptr.add(off + first + run_at),
                            run_len,
                        );
                    }
                }
                if !runs.is_empty() {
                    super::wc_sfence();
                }
            }
            let rebar_cb_after = t0.elapsed();

            println!(
                "[BENCH] ReBAR GPU_UPLOAD write 128MiB: copy_mapped={rebar_write_nt:?} ({:.2} GB/s), memcpy={rebar_write_memcpy:?} ({:.2} GB/s)",
                (8 * BUF_LEN) as f64 / rebar_write_nt.as_secs_f64() / 1e9,
                (8 * BUF_LEN) as f64 / rebar_write_memcpy.as_secs_f64() / 1e9,
            );
            println!(
                "[BENCH] ReBAR GPU_UPLOAD read 16MiB: before(memcpy)={rebar_read_before:?} ({:.3} GB/s), after(VMOVNTDQA 1T)={rebar_read_after:?} ({:.3} GB/s, {:.1}x), copy_wide(WC-1T)={rebar_read_wide:?} ({:.3} GB/s, {:.1}x)",
                BUF_LEN as f64 / rebar_read_before.as_secs_f64() / 1e9,
                BUF_LEN as f64 / rebar_read_after.as_secs_f64() / 1e9,
                rebar_read_before.as_secs_f64() / rebar_read_after.as_secs_f64(),
                BUF_LEN as f64 / rebar_read_wide.as_secs_f64() / 1e9,
                rebar_read_before.as_secs_f64() / rebar_read_wide.as_secs_f64(),
            );
            println!(
                "[BENCH] ReBAR GPU_UPLOAD cmp 16MiB: before(slice==)={rebar_cmp_before:?} ({:.3} GB/s), after(slices_equal_wc 1T)={rebar_cmp_after:?} ({:.3} GB/s, {:.1}x), find_pieces_wc(WC-1T)={rebar_cmp_wide:?} ({:.3} GB/s, {:.1}x)",
                BUF_LEN as f64 / rebar_cmp_before.as_secs_f64() / 1e9,
                BUF_LEN as f64 / rebar_cmp_after.as_secs_f64() / 1e9,
                rebar_cmp_before.as_secs_f64() / rebar_cmp_after.as_secs_f64(),
                BUF_LEN as f64 / rebar_cmp_wide.as_secs_f64() / 1e9,
                rebar_cmp_before.as_secs_f64() / rebar_cmp_wide.as_secs_f64(),
            );
            println!(
                "[BENCH] ReBAR GPU_UPLOAD copy_back 1MiB (16x64KiB): before(direct WC)={rebar_cb_before:?}, after(VMOVNTDQA+L1)={rebar_cb_after:?} ({:.1}x)",
                rebar_cb_before.as_secs_f64() / rebar_cb_after.as_secs_f64(),
            );
        } else {
            println!("[BENCH] ReBAR GPU_UPLOAD not supported on this adapter");
        }
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers_read()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
