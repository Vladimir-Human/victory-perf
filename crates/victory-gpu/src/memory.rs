//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, drained on new
//!   frames and fresh command lists so newly faulted pages become visible
//!   without waiting for an extra frame. A pointer a shader takes out of
//!   memory cannot be known before it runs; the addresses a dispatch is
//!   given as root views and root constants are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! subsequent submissions. For the structure builder that is the same as a
//! dropped stage, so before a dispatch the host makes ready everything it
//! can see the dispatch name: the whole copy around each root view, around
//! each pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{Ordering, fence};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// Frames a retired copy slot stays untouched before reuse so no in-flight
/// command list can observe a replaced view-heap descriptor.
const RETIRE_DELAY_FRAMES: u64 = 4;

#[derive(Default, Clone, Copy)]
struct ListSync {
    read_epoch: u64,
    write_epoch: u64,
    frame: u64,
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Number of directory pages currently pointing to each copy slot.
    copy_pages: Vec<u32>,
    /// Monotonic epoch at which each copy slot was assigned or revived.
    slot_epoch: Vec<u64>,
    /// Frame at which a copy slot's page count dropped to 0 (`u64::MAX` while live).
    slot_retired_frame: Vec<u64>,
    /// Global epoch incremented whenever a copy slot is allocated, replaced, or revived.
    epoch: u64,
    /// Per host view heap (by address): `(base_slot, synced_epoch)`.
    heaps: HashMap<usize, (u32, u64)>,
    /// Per command list (by address): high-water refresh/write epochs so
    /// copies added by concurrent recording threads are never missed.
    lists: HashMap<usize, ListSync>,
    /// The frame the refused cache was last cleared in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

/// Extracts candidate title addresses encoded in root constants:
/// - two adjacent constants `(lo, hi)` where `hi` is in `1..=0xFF` and `lo` is
///   either `>= 64 KiB` or 64-byte aligned (catching 4 GiB arena bases such as
///   `0x10_0000_0000` while skipping small unaligned integers);
/// - one constant `c >= 0x0100_0000` encoding an address in 256-byte units.
fn extract_constant_addresses(constants: &[u32]) -> Vec<u64> {
    let mut out = Vec::new();
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && (c >= 0x1_0000 || (c & 0x3F) == 0)
        {
            out.push(u64::from(c) | (u64::from(high) << 32));
        }
        if c >= 0x0100_0000 {
            out.push(u64::from(c) << 8);
        }
    }
    out
}

/// Merges sorted `(start, end)` intervals into disjoint sorted spans so
/// overlap queries run in exact O(log N) regardless of how many small buffers
/// sit inside a large allocation.
fn merge_sorted_spans(spans: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for &(start, end) in spans {
        if start >= end {
            continue;
        }
        if let Some(last) = merged.last_mut()
            && start <= last.1
        {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Returns whether `[at, at + PAGE)` overlaps any merged disjoint sorted
/// `(start, end)` range of registered non-window title buffers.
fn overlaps_any_buffer(at: u64, merged_buffers: &[(u64, u64)]) -> bool {
    let end = at.saturating_add(PAGE);
    let idx = merged_buffers.partition_point(|&(start, _)| start < end);
    idx > 0 && merged_buffers[idx - 1].1 > at
}

/// Sorted `(start, end)` spans of registered title buffers (non-window copies),
/// paired with their disjoint merged form for exact O(log N) overlap queries.
fn registered_buffer_spans() -> (Vec<(u64, u64)>, Vec<(u64, u64)>) {
    let mut spans: Vec<(u64, u64)> = crate::shadow::spans()
        .into_iter()
        .filter(|&(_, _, is_window)| !is_window)
        .map(|(start, len, _)| (start, start.saturating_add(len)))
        .collect();
    spans.sort_unstable();
    let merged = merge_sorted_spans(&spans);
    (spans, merged)
}

/// Computes the 32-bit byte offset of page `at` inside `shadow`'s underlying
/// host `ID3D12Resource` (accounting for any in-place placement bias) and
/// verifies the entire 64 KiB page fits within a 32-bit byte offset.
fn page_buffer_offset(shadow: &Shadow, at: u64) -> Option<u32> {
    let offset = shadow.offset(at);
    let end = offset.checked_add(PAGE)?;
    (end <= u64::from(u32::MAX)).then_some(offset as u32)
}

/// Writes `(copy_id, byte_offset)` into a directory slot. When replacing an
/// existing non-zero entry, clears `copy_id` to 0 first so a concurrent GPU
/// reader never observes a torn `(old_copy_id, new_offset)` pair.
///
/// # Safety
/// `entry` must point to a valid 2-dword slot inside the directory mapping.
unsafe fn write_directory_entry(entry: *mut u32, old_copy: u32, new_copy: u32, offset: u32) {
    unsafe {
        if old_copy != 0 && old_copy != new_copy {
            entry.write_volatile(0);
            fence(Ordering::Release);
        }
        entry.add(1).write_volatile(offset);
        fence(Ordering::Release);
        entry.write_volatile(new_copy);
    }
}

/// Finds the active [`Shadow`] copy covering the full 64 KiB page at `start`
/// (`start` page-aligned) that is writable as a raw UAV and permitted under
/// `VICTORY_BVH_QUARANTINE`.
fn find_page_shadow(start: u64) -> Option<Arc<Shadow>> {
    let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
    let shadow = crate::shadow::find(start, 1)
        .filter(whole)
        .or_else(|| crate::shadow::find(start, PAGE).filter(whole))?;
    // Under `VICTORY_BVH_QUARANTINE=1`, never map an in-place placed buffer:
    // shaders writing through `SLOT_MEMORY` would mutate title memory directly
    // and bypass quarantine.
    if crate::pipeline::quarantine_enabled() && shadow.in_place() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                start = format_args!("{start:#x}"),
                "title memory by address: skipping in-place buffer while VICTORY_BVH_QUARANTINE=1 to keep builder stores isolated from title memory"
            );
        });
        return None;
    }
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        return None;
    }
    Some(shadow)
}

/// Selects a copy slot to hold a new shadow: reuses an existing slot if
/// already present, appends while under `COPIES`, or recycles a retired slot
/// whose page count has been zero for at least [`RETIRE_DELAY_FRAMES`] frames.
fn select_copy_slot(
    copies_len: usize,
    copy_pages: &[u32],
    slot_retired_frame: &[u64],
    frame: u64,
) -> Option<(usize, bool)> {
    if copies_len < COPIES as usize {
        return Some((copies_len, false));
    }
    for (n, (&pages, &retired_at)) in copy_pages.iter().zip(slot_retired_frame).enumerate() {
        if pages == 0 && frame.saturating_sub(retired_at) > RETIRE_DELAY_FRAMES {
            return Some((n, true));
        }
    }
    None
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            copy_pages: Vec::new(),
            slot_epoch: Vec::new(),
            slot_retired_frame: Vec::new(),
            epoch: 0,
            heaps: HashMap::new(),
            lists: HashMap::new(),
            frame: u64::MAX,
            refused: HashSet::new(),
        })
    }

    /// Registers `shadow` in `self.copies` (or returns its existing slot),
    /// returning `(slot_index, is_newly_assigned_or_revived)`.
    fn ensure_slot(&mut self, shadow: &Arc<Shadow>) -> Option<(usize, bool)> {
        if let Some(n) = self.copies.iter().position(|c| Arc::ptr_eq(c, shadow)) {
            if self.copy_pages[n] == 0 {
                // Reviving a previously retired slot: bump epoch so command
                // lists that skipped it while retired refresh/write-track it.
                self.epoch = self.epoch.wrapping_add(1).max(1);
                self.slot_epoch[n] = self.epoch;
                self.slot_retired_frame[n] = u64::MAX;
                return Some((n, true));
            }
            return Some((n, false));
        }
        let Some((slot, recycled)) = select_copy_slot(
            self.copies.len(),
            &self.copy_pages,
            &self.slot_retired_frame,
            self.frame,
        ) else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    COPIES,
                    "title memory by address: no number left for another copy; its pages read zeros"
                );
            });
            return None;
        };
        self.epoch = self.epoch.wrapping_add(1).max(1);
        if recycled {
            self.copies[slot] = shadow.clone();
            self.copy_pages[slot] = 0;
            self.slot_epoch[slot] = self.epoch;
            self.slot_retired_frame[slot] = u64::MAX;
        } else {
            self.copies.push(shadow.clone());
            self.copy_pages.push(0);
            self.slot_epoch.push(self.epoch);
            self.slot_retired_frame.push(u64::MAX);
        }
        Some((slot, true))
    }

    /// Assigns directory page `page` to `new_slot` at `offset`, maintaining
    /// per-slot reference counts and retiring slots that lose their last page.
    fn assign_page(&mut self, page: u64, new_slot: usize, offset: u32) {
        // SAFETY: callers check `page < MEMORY_PAGES - 1`.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: inside the directory's mapping.
        let old_copy = unsafe { entry.read() };
        let new_copy = new_slot as u32 + 1;
        if old_copy == new_copy {
            // SAFETY: inside the directory's mapping.
            let old_off = unsafe { entry.add(1).read() };
            if old_off == offset {
                return;
            }
        }
        // SAFETY: inside the directory's mapping.
        unsafe { write_directory_entry(entry, old_copy, new_copy, offset) };
        if old_copy != 0 && old_copy != new_copy {
            let old_slot = (old_copy - 1) as usize;
            if let Some(count) = self.copy_pages.get_mut(old_slot) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.slot_retired_frame[old_slot] = self.frame;
                }
            }
        }
        if old_copy != new_copy {
            self.copy_pages[new_slot] = self.copy_pages[new_slot].saturating_add(1);
            self.slot_retired_frame[new_slot] = u64::MAX;
        }
    }

    /// Gives the page of `address` its copy, if it has none (or if
    /// `revalidate` is true and the page's active `Shadow` changed, e.g. a
    /// title buffer was registered inside a loose window or reallocated after
    /// `unregister`/`in_place_let_go`) and the memory is the title's.
    fn map(&mut self, address: u64, revalidate: bool) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let existing = unsafe { entry.read() };
        if existing != 0 && !revalidate {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if existing == 0 && self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            if existing == 0 {
                self.refused.insert(page);
            }
            return;
        }
        let Some(shadow) = find_page_shadow(start) else {
            return;
        };
        if existing != 0
            && self
                .copies
                .get((existing - 1) as usize)
                .is_some_and(|c| Arc::ptr_eq(c, &shadow))
        {
            return;
        }
        let Some(start_offset) = page_buffer_offset(&shadow, start) else {
            return;
        };
        let Some((n, fresh_slot)) = self.ensure_slot(&shadow) else {
            return;
        };
        self.assign_page(page, n, start_offset);

        // Eagerly fill the remaining full pages of a newly registered copy so
        // multi-page BVH structures do not fault page by page. When `shadow`
        // is a loose window, never overwrite pages belonging to a registered
        // title buffer inside that window (otherwise root-UAV writes to the
        // buffer and raw-address reads through the window alias different GPU
        // copies).
        if fresh_slot || (existing != 0 && revalidate) {
            let (first, len) = shadow.span();
            let (buffers, merged_buffers) = registered_buffer_spans();
            let is_buffer = buffers
                .binary_search(&(first, first.saturating_add(len)))
                .is_ok();
            let mut at = first.next_multiple_of(PAGE);
            while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                let other_page = at / PAGE;
                if other_page != page {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add(other_page as usize * 2) };
                    // SAFETY: as above.
                    let cur_copy = unsafe { other.read() };
                    let can_claim = if cur_copy == 0 {
                        // A loose window must not claim pages covered by a
                        // registered title buffer.
                        is_buffer || !overlaps_any_buffer(at, &merged_buffers)
                    } else if is_buffer && cur_copy != n as u32 + 1 {
                        // A registered title buffer always supersedes a loose
                        // window or unregistered stale copy on its pages, and
                        // on revalidation supersedes an older replaced copy on
                        // sibling pages too.
                        let cur_not_registered_buffer =
                            self.copies.get((cur_copy - 1) as usize).is_some_and(|c| {
                                let (cf, cl) = c.span();
                                buffers.binary_search(&(cf, cf.saturating_add(cl))).is_err()
                            });
                        cur_not_registered_buffer || (revalidate && cur_copy == existing)
                    } else {
                        false
                    };
                    if can_claim
                        && let Some(off) = page_buffer_offset(&shadow, at)
                    {
                        self.assign_page(other_page, n, off);
                    }
                }
                at += PAGE;
            }
        }
    }

    /// The pages shaders found empty since the last drain. Uses 64-bit loads
    /// to scan the 16 KiB fault table in <1 us on write-back shared memory.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        // Dword 0 is the unused sink; scan slot 1 alone, then 2-dword aligned
        // pairs `(2, 3), (4, 5), ...` up to `MEMORY_FAULTS`.
        let faults_len = MEMORY_FAULTS as usize;
        if faults_len > 1 {
            // SAFETY: inside the faults mapping.
            let at1 = unsafe { self.faults.1.add(1) };
            // SAFETY: as above; the GPU may be writing concurrently.
            let p1 = unsafe { at1.read_volatile() };
            if p1 != 0 {
                pages.push(p1);
                // SAFETY: as above.
                unsafe { at1.write_volatile(0) };
            }
        }
        let mut i = 2usize;
        while i + 1 < faults_len {
            // SAFETY: `i` is even and `i + 1 < MEMORY_FAULTS`; the mapping is
            // 64 KiB aligned so `self.faults.1.add(i)` is 8-byte aligned.
            let pair = unsafe { self.faults.1.add(i).cast::<u64>().read_volatile() };
            if pair != 0 {
                let lo = pair as u32;
                let hi = (pair >> 32) as u32;
                if lo != 0 {
                    pages.push(lo);
                    // SAFETY: inside the faults mapping.
                    unsafe { self.faults.1.add(i).write_volatile(0) };
                }
                if hi != 0 {
                    pages.push(hi);
                    // SAFETY: inside the faults mapping.
                    unsafe { self.faults.1.add(i + 1).write_volatile(0) };
                }
            }
            i += 2;
        }
        if i < faults_len {
            // SAFETY: `i < MEMORY_FAULTS`.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: inside the faults mapping.
            let p = unsafe { at.read_volatile() };
            if p != 0 {
                pages.push(p);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE, false);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64, revalidate: bool) {
        self.map(address, revalidate);
        if let Some(next) = (address & !(CELL - 1)).checked_add(CELL) {
            self.map(next, false);
        }
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    // Once per list reset: reset per-list generation watermarks and keep the
    // directory/fault buffers alive for the list's lifetime.
    const READ: usize = usize::MAX - 0x6d;
    let first_read = list.first_use(READ);
    let new_frame = m.frame != frame;
    if new_frame {
        m.frame = frame;
        m.refused.clear();
        m.lists
            .retain(|_, s| frame.saturating_sub(s.frame) <= 16);
    }
    if new_frame || first_read {
        m.take_faults();
    }
    // Root-view addresses revalidate existing directory entries so a newly
    // registered or reallocated title buffer immediately supersedes a loose
    // window or stale copy covering the same pages.
    for &address in addresses {
        m.name(address, true);
    }
    for address in extract_constant_addresses(constants) {
        m.name(address, false);
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, synced_epoch) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    if synced_epoch < m.epoch {
        for (n, shadow) in m.copies.iter().enumerate() {
            if m.slot_epoch[n] > synced_epoch {
                let (first, len) = shadow.span();
                // Account for any in-place placement bias so the raw UAV from
                // offset 0 covers every byte up to `bias + len`.
                let view_len = shadow.offset(first).saturating_add(len);
                heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, view_len);
            }
        }
        m.heaps.insert(key, (base, m.epoch));
    } else if !m.heaps.contains_key(&key) {
        m.heaps.insert(key, (base, m.epoch));
    }
    let list_key = std::ptr::from_ref(list) as usize;
    let entry = m.lists.entry(list_key).or_default();
    if first_read {
        entry.read_epoch = 0;
        entry.write_epoch = 0;
    }
    entry.frame = frame;
    let read_epoch = entry.read_epoch;
    let write_epoch = entry.write_epoch;
    if read_epoch < m.epoch || (writes && write_epoch < m.epoch) {
        for (n, shadow) in m.copies.iter().enumerate() {
            if m.copy_pages[n] == 0 {
                continue;
            }
            let e = m.slot_epoch[n];
            let (start, len) = shadow.span();
            if e > read_epoch {
                shadow.refresh(list, start, len);
            }
            if writes && e > write_epoch {
                shadow.gpu_writes(list, start, len);
            }
        }
        if let Some(entry) = m.lists.get_mut(&list_key) {
            entry.read_epoch = m.epoch;
            if writes {
                entry.write_epoch = m.epoch;
            }
        }
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_constant_addresses_catches_4gib_boundaries_and_256b_units() {
        // 4 GiB-aligned 64-bit pair (0x10_0000_0000) was previously dropped by `c >= 0x1_0000`.
        let got = extract_constant_addresses(&[0x0000_0000, 0x10, 0x0002_0000, 0x20]);
        assert!(got.contains(&0x10_0000_0000), "{got:#x?}");
        assert!(got.contains(&0x20_0002_0000), "{got:#x?}");

        // Small unaligned integers (e.g. dispatch/loop counts like (3, 1)) are ignored.
        assert!(extract_constant_addresses(&[3, 1, 17, 2]).is_empty());

        // 64-byte-aligned low offset under 64 KiB (e.g. 0x10_0000_0040) is accepted.
        assert_eq!(
            extract_constant_addresses(&[0x40, 0x10]),
            vec![0x10_0000_0040]
        );

        // Single constant in 256-byte units (`c >= 0x0100_0000`).
        assert_eq!(
            extract_constant_addresses(&[0x0123_4567]),
            vec![0x01_2345_6700]
        );
    }

    #[test]
    fn overlaps_any_buffer_protects_title_buffers_inside_loose_windows() {
        let mut raw = vec![
            (0x10_0010_0000, 0x10_0020_0000), // 1 MiB BVH buffer
            (0x10_0020_8000, 0x10_0020_C000), // 16 KiB sub-page buffer
            (0x20_0000_0000, 0x20_0B00_0000), // 176 MiB heap enclosing >100 nested buffers
        ];
        for i in 0..128u64 {
            let s = 0x20_0001_0000 + i * 0x1000;
            raw.push((s, s + 0x400));
        }
        raw.sort_unstable();
        let merged = merge_sorted_spans(&raw);
        // Loose window pages outside buffers are free to claim.
        assert!(!overlaps_any_buffer(0x10_0000_0000, &merged));
        assert!(!overlaps_any_buffer(0x10_000F_0000, &merged));
        assert!(!overlaps_any_buffer(0x10_0021_0000, &merged));
        assert!(!overlaps_any_buffer(0x20_0B00_0000, &merged));
        // Pages inside or overlapping registered buffers (including deep inside
        // the 176 MiB enclosing heap past 128 smaller nested buffers) are protected.
        assert!(overlaps_any_buffer(0x10_0010_0000, &merged));
        assert!(overlaps_any_buffer(0x10_001F_0000, &merged));
        assert!(overlaps_any_buffer(0x10_0020_0000, &merged));
        assert!(overlaps_any_buffer(0x20_0A00_0000, &merged));
    }

    #[test]
    fn select_copy_slot_recycles_only_after_retire_delay() {
        // Under capacity: appends directly.
        assert_eq!(select_copy_slot(10, &[1; 10], &[u64::MAX; 10], 50), Some((10, false)));

        // At full capacity with one retired slot at frame 100:
        let mut pages = vec![1u32; COPIES as usize];
        let mut retired = vec![u64::MAX; COPIES as usize];
        pages[7] = 0;
        retired[7] = 100;
        // Too soon (in-flight lists could still reference slot 7): refused.
        assert_eq!(
            select_copy_slot(COPIES as usize, &pages, &retired, 100 + RETIRE_DELAY_FRAMES),
            None
        );
        // After retire delay: cleanly recycled.
        assert_eq!(
            select_copy_slot(
                COPIES as usize,
                &pages,
                &retired,
                100 + RETIRE_DELAY_FRAMES + 1
            ),
            Some((7, true))
        );
    }

    #[test]
    fn write_directory_entry_updates_offset_and_copy() {
        let mut dir = [0u32; 4];
        unsafe {
            write_directory_entry(dir.as_mut_ptr(), 0, 5, 0x3_0000);
        }
        assert_eq!((dir[0], dir[1]), (5, 0x3_0000));
        unsafe {
            write_directory_entry(dir.as_mut_ptr(), 5, 9, 0x1_0000);
        }
        assert_eq!((dir[0], dir[1]), (9, 0x1_0000));
    }
}
