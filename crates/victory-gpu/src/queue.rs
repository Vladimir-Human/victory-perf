//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A signal of the queue's, due after what it holds before it.
    Signal(usize, u64),
}

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present runs what queues still hold (on by default: a wait
/// is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for
/// finding out whether the title's waits reach across its presents; a
/// wait nothing ever meets then stops the title for good.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

#[cfg(all(windows, test))]
static QUEUE_SUBMISSIONS: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_JOIN_PARTS` (on by default, `=0` disables): coalesces consecutive
/// ready submission parts in a queue into fewer `ExecuteCommandLists`
/// submissions when no unsatisfied wait, fence signal, or copy-back/readback
/// hazard separates them.
#[cfg(windows)]
pub(crate) fn join_parts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_JOIN_PARTS").is_none_or(|v| v != "0");
        if !on {
            tracing::info!("consecutive submission part coalescing off (VICTORY_JOIN_PARTS=0)");
        }
        on
    })
}

/// Coalesces consecutive ready `Held::Run` entries at the front of `held` into
/// `(lists, order)`, crossing only already-satisfied `Held::Wait`/`Held::Value`
/// gates (or forced waits when `all` is true) and stopping before any
/// `Held::Signal`, unsatisfied wait, or copy-back/readback hazard. Returns the
/// number of additional parts joined into the initial run.
#[cfg(windows)]
fn coalesce_held_runs(
    q: usize,
    all: bool,
    held: &mut Vec<Held>,
    lists: &mut Vec<crate::list::Detached>,
    order: &mut Vec<usize>,
) -> usize {
    let mut joined = 0usize;
    loop {
        let earlier_hosts: Vec<&crate::host::HostList> =
            order.iter().map(|&i| lists[i].host()).collect();
        // Peek past any satisfied `Wait` / `Value` gates to the next `Run`.
        let mut scan = 0usize;
        let can_cross = loop {
            match held.get(scan) {
                Some(&Held::Wait(fence, value)) => {
                    if !all && crate::fence::completed(fence as This) < value {
                        break false;
                    }
                    scan += 1;
                }
                Some(Held::Value(wait)) => {
                    if !all {
                        // If `earlier` itself writes to the waited-on memory
                        // (e.g. resetting a flag before a wait), `value_met`
                        // would observe stale CPU memory; run `earlier` first.
                        let (addr, bytes) = wait.place();
                        if crate::host::lists_write_range(&earlier_hosts, addr, bytes)
                            || !value_met(wait)
                        {
                            break false;
                        }
                    }
                    scan += 1;
                }
                Some(Held::Run(..)) => break true,
                Some(Held::Signal(..)) | None => break false,
            }
        };
        if !can_cross {
            break;
        }
        let Some(Held::Run(next_lists, next_order)) = held.get(scan) else {
            break;
        };
        let safe = {
            let next_hosts: Vec<&crate::host::HostList> =
                next_order.iter().map(|&i| next_lists[i].host()).collect();
            crate::host::parts_can_join(&earlier_hosts, &next_hosts)
        };
        if !safe {
            break;
        }
        // Consume the satisfied gates and the next `Held::Run`, preserving
        // forced-wait diagnostics when `all` is true.
        for item in held.drain(..scan) {
            if let Held::Value(wait) = item
                && all
                && !value_met(&wait)
            {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                    tracing::warn!(
                        queue = format_args!("{q:#x}"),
                        ?wait,
                        "a wait for a memory value was not met at a present: the queue runs on"
                    );
                }
            }
        }
        if let Held::Run(next_lists, next_order) = held.remove(0) {
            let base = lists.len();
            lists.extend(next_lists);
            order.extend(next_order.into_iter().map(|i| base + i));
            joined += 1;
        }
    }
    joined
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let first = if lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 && crate::host::late() {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they are settled.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    } else if first > 0 {
        let now: Vec<_> = lists[..first]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
        // They have run: their descriptor tables' slots are free again, as
        // on the path without cuts ([`execute`]). This path replaced that
        // one without the call, so the lists that run at once kept their
        // slots until the title's reset and the ring ran out as before
        // (run 45: 3,143 of 15,942 tables refused in one frame).
        crate::list::tables_done(&lists[..first]);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        lock(&queue(this).held).extend(pieces);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let held = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(held);
            loop {
                match held.first() {
                    None => break,
                    Some(&Held::Wait(fence, value))
                        if !all && crate::fence::completed(fence as This) < value =>
                    {
                        break;
                    }
                    Some(Held::Value(wait)) if !all && !value_met(wait) => break,
                    Some(_) => {}
                }
                match held.remove(0) {
                    Held::Wait(..) => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if all && !value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(mut lists, mut order) => {
                        let joined = if join_parts() {
                            coalesce_held_runs(q, all, &mut held, &mut lists, &mut order)
                        } else {
                            0
                        };
                        #[cfg(feature = "diagnostics")]
                        {
                            if joined > 0 {
                                crate::copystat::add(
                                    crate::copystat::C::PartsTogether,
                                    joined as u64,
                                );
                            } else if !join_parts() {
                                let waits =
                                    held.iter().take_while(|h| matches!(h, Held::Value(_))).count();
                                if waits > 0
                                    && matches!(held.get(waits), Some(Held::Run(..)))
                                    && held[..waits]
                                        .iter()
                                        .all(|h| matches!(h, Held::Value(w) if w.met()))
                                {
                                    crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                                }
                            }
                            crate::capture::op(q, || {
                                format!(
                                    "RUN held submission of {} lists (joined {joined}, forced {all})",
                                    order.len()
                                )
                            });
                        }
                        let _ = joined;
                        #[cfg(test)]
                        QUEUE_SUBMISSIONS.fetch_add(1, Ordering::Relaxed);
                        if crate::host::late()
                            && let Some(host) = crate::host::host()
                        {
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            host.execute_late(lists, order);
                        } else {
                            let ran = {
                                let runs: Vec<&crate::host::HostList> =
                                    order.iter().map(|&i| lists[i].host()).collect();
                                match crate::host::host() {
                                    Some(host) => {
                                        host.execute(&runs);
                                        true
                                    }
                                    None => false,
                                }
                            };
                            // They have run: their host lists are used again.
                            if ran {
                                crate::list::recycle(lists, &order);
                            }
                        }
                        if all {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Forced, 1);
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !crate::host::late_signal(fence, value) {
                            crate::fence::set(fence as This, value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            break;
        }
    }
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || (crate::host::late() && crate::host::late_met(wait))
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            Held::Run(taken, order)
        });
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(fence as usize, value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence as usize, value) {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || now < value;
            if behind && waits() {
                held.push(Held::Wait(fence as usize, value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    _: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    #[cfg(windows)]
    if crate::host::late() && !crate::host::late_free_present() {
        crate::host::settle_for_present();
        release_held(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here.
        release_held(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    release_held(forces());
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = unsafe { *plane.resources }
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        // A second plane (the title's interface over its game image, in
        // its Series S mode) is drawn over the first: `crate::over`.
        const READABLE: u32 = 0x80;
        // SAFETY: the title passes `planes` plane descriptions.
        let second = unsafe { second_plane(planes, plane_parameters, resource) }
            .and_then(|r| Some((r, crate::resource::host_resource(r)?)));
        let over = second.map(|(r, t)| {
            let was = crate::resource::host_transition(r, READABLE).map_or(READABLE, |(_, before)| before);
            (r, t, was)
        });
        let shown = host.present(texture, state, frame, over.map(|(_, t, was)| (t, was)));
        if let Some((r, _, was)) = over {
            crate::resource::host_transition(r, was);
        }
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}

/// The bytes from one plane's description to the next in a present.
#[cfg(windows)]
const PLANE_STRIDE: usize = 0x48;

/// The first resource of a present's second plane, if it has one that is not
/// the first plane's (`first`): what is drawn over the first.
///
/// # Safety
/// `parameters` is null or points at `planes` plane descriptions,
/// [`PLANE_STRIDE`] bytes apart, each listing `count` resources.
#[cfg(windows)]
unsafe fn second_plane(planes: u32, parameters: *const Plane, first: This) -> Option<This> {
    if planes < 2 || parameters.is_null() {
        return None;
    }
    // SAFETY: the caller's: a second description follows the first.
    let plane = unsafe { &*parameters.cast::<u8>().add(PLANE_STRIDE).cast::<Plane>() };
    if plane.count == 0 || plane.resources.is_null() {
        return None;
    }
    // SAFETY: the caller's: the plane lists `count` resources.
    let resource = unsafe { *plane.resources };
    (!resource.is_null() && resource != first).then_some(resource)
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run.
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }

    #[cfg(windows)]
    static QUEUE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Inspects `list`'s own vtable to decide whether it uses the newer
    /// console layout (+2 slots from `0x1d8` onward), avoiding races with
    /// parallel device-layout tests.
    #[cfg(windows)]
    fn list_slot_offset(list: This, old_offset: usize) -> usize {
        if old_offset < 0x1d8 {
            return old_offset;
        }
        // SAFETY: one of our command lists whose first field is its 160-entry
        // vtable; `sync` occupies two adjacent slots: (129, 130) in the older
        // layout and (131, 132) in the newer layout (unused slots have distinct
        // per-slot `missing` stubs).
        let newer = unsafe {
            let table = *list.cast::<*const *const ()>();
            *table.add(131) == *table.add(132)
        };
        if newer { old_offset + 0x10 } else { old_offset }
    }

    /// Real-GPU test of `VICTORY_JOIN_PARTS` coalescing and negative controls:
    /// 1. Consecutive ready parts across satisfied `ValueWait`s coalesce into
    ///    one `ExecuteCommandLists` submission when `join_parts()` is true.
    /// 2. Unsatisfied `ValueWait` blocks coalescing until satisfied.
    /// 3. Intervening `Held::Signal` blocks coalescing and preserves fence order.
    /// 4. Readback / overlapping bottom-of-pipe write hazard blocks coalescing
    ///    (`parts_can_join` == false), while non-overlapping writes coalesce.
    #[cfg(windows)]
    #[test]
    fn hw_coalesce_ready_parts_and_negative_controls() {
        let _guard = QUEUE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let require_hw = std::env::var_os("VICTORY_REQUIRE_HW_GPU").is_some_and(|v| v != "0");
        let Some(host) = crate::host::host() else {
            assert!(!require_hw, "VICTORY_REQUIRE_HW_GPU=1 set, but no host GPU");
            return;
        };

        let null = std::ptr::null();
        let mut allocator = null;
        let alloc_iid = guid(
            0x6102dee4,
            0xaf59,
            0x4b09,
            [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
        );
        let list_iid = guid(
            0x5b160d0f,
            0xac1b,
            0x4185,
            [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
        );
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut allocator), S_OK);
        let make_list = || -> This {
            let mut l = null;
            assert_eq!(crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut l), S_OK);
            l
        };
        let write32_on = |l: This, addr: u64, val: u32| {
            let f = slot::<extern "system" fn(This, u64, u32, u32, u32)>(l, list_slot_offset(l, 0x228));
            f(l, addr, val, 0, 0);
        };
        let wait32_on = |l: This, addr: u64, cmp: u32, val: u64| {
            let f = slot::<extern "system" fn(This, u64, u32, u64)>(l, list_slot_offset(l, 0x238));
            f(l, addr, cmp, val);
        };
        let close_on = |l: This| {
            let f = slot::<extern "system" fn(This) -> HRESULT>(l, 0x48);
            assert_eq!(f(l), S_OK);
        };
        let release_obj = |o: This| {
            slot::<extern "system" fn(This) -> u32>(o, 0x10)(o);
        };

        // Create queue + fence.
        let qdesc = QueueDesc { kind: 0, priority: 0, flags: 0, nodes: 0 };
        let mut q = null;
        assert_eq!(create(null, &qdesc, &QUEUE_IIDS[0], &mut q), S_OK);
        let mut fence = null;
        assert_eq!(
            crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence),
            S_OK
        );

        // --- Case 1: 4 consecutive ready parts behind satisfied ValueWaits ---
        let ready_flags = Box::new([1u32; 4]);
        let mut out_words = Box::new([0u32; 4]);
        let l1 = make_list();
        for i in 0..4usize {
            if i > 0 {
                // Comparison 3 == EQUAL: memory already holds 1.
                wait32_on(l1, &raw const ready_flags[i] as u64, 3, 1);
            }
            write32_on(l1, &raw mut out_words[i] as u64, (i as u32 + 1) * 10);
        }
        close_on(l1);

        let before_subs = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        execute(q, 1, &l1);
        crate::host::settle_for_present();
        let after_subs = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        assert_eq!(*out_words, [10, 20, 30, 40], "all 4 parts must execute in order");
        if cuts() && join_parts() {
            assert_eq!(
                after_subs - before_subs,
                1,
                "4 ready parts must coalesce into 1 host submission when VICTORY_JOIN_PARTS=1"
            );
        } else if cuts() {
            assert_eq!(
                after_subs - before_subs,
                4,
                "4 parts must stay 4 separate host submissions when VICTORY_JOIN_PARTS=0"
            );
        }
        release_obj(l1);

        // --- Case 2 (Negative control): unsatisfied ValueWait blocks coalescing ---
        let mut gate_flag = Box::new(0u32);
        let mut neg_out = Box::new([0u32; 2]);
        let l2 = make_list();
        write32_on(l2, &raw mut neg_out[0] as u64, 111);
        // Comparison 3 == EQUAL: currently 0 != 1, so part 2 MUST stay held!
        wait32_on(l2, &raw const *gate_flag as u64, 3, 1);
        write32_on(l2, &raw mut neg_out[1] as u64, 222);
        close_on(l2);

        let before_neg = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        execute(q, 1, &l2);
        crate::host::settle_for_present();
        // Part 1 ran, part 2 MUST NOT have run yet!
        assert_eq!(neg_out[0], 111, "prefix before unsatisfied wait must run");
        assert_eq!(neg_out[1], 0, "suffix behind unsatisfied wait must stay held");
        let mid_neg = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        assert_eq!(mid_neg - before_neg, 1);

        // Now satisfy the memory wait and release held work!
        *gate_flag = 1;
        release_held(false);
        crate::host::settle_for_present();
        assert_eq!(neg_out[1], 222, "suffix must run once wait value is satisfied");
        let end_neg = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        assert_eq!(end_neg - before_neg, 2);
        release_obj(l2);

        // --- Case 3 (Negative control): intervening Held::Signal blocks coalescing ---
        let mut sig_out = Box::new([0u32; 2]);
        let mut fence_gate = null;
        assert_eq!(
            crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence_gate),
            S_OK
        );
        // Hold the queue behind `fence_gate == 1` so we can enqueue `[RunA, Signal(fence, 9), RunB]`.
        let wait_fn = slot::<extern "system" fn(This, This, u64) -> HRESULT>(
            q,
            signal_slot(q) + 8,
        );
        let signal_fn = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(wait_fn(q, fence_gate, 1), S_OK);
        let la = make_list();
        write32_on(la, &raw mut sig_out[0] as u64, 1);
        close_on(la);
        let lb = make_list();
        write32_on(lb, &raw mut sig_out[1] as u64, 2);
        close_on(lb);
        execute(q, 1, &la);
        assert_eq!(signal_fn(q, fence, 9), S_OK);
        execute(q, 1, &lb);
        assert_eq!(*sig_out, [0, 0], "both held behind fence_gate");
        assert_eq!(crate::fence::completed(fence), 0);

        let before_sig = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        assert_eq!(crate::fence::signal(fence_gate, 1), S_OK);
        crate::host::settle_for_present();
        let after_sig = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        assert_eq!(*sig_out, [1, 2]);
        assert_eq!(crate::fence::completed(fence), 9);
        assert_eq!(
            after_sig - before_sig,
            2,
            "intervening Held::Signal must prevent coalescing RunA and RunB"
        );
        release_obj(la);
        release_obj(lb);
        release_obj(fence_gate);

        // --- Case 4 (Negative control): readback / overlapping write hazard ---
        let (hl_a, hl_b) = (host.new_list().unwrap(), host.new_list().unwrap());
        let mut shared_mem = [0u8; 64];
        let mut other_mem = [0u8; 64];
        // Clean lists with no hazards can join.
        assert!(crate::host::parts_can_join(&[&hl_a], &[&hl_b]));
        // Non-overlapping bottom-of-pipe write in A vs Fill in B -> safe to join.
        hl_a.defer_write(shared_mem.as_mut_ptr(), Written::Value(7), 4);
        hl_b.defer_fill(other_mem.as_mut_ptr(), other_mem.as_ptr(), 4);
        assert!(crate::host::parts_can_join(&[&hl_a], &[&hl_b]));
        // Overlapping bottom-of-pipe write in A vs Fill reading the same bytes in B -> hazard!
        hl_b.defer_fill(other_mem.as_mut_ptr(), shared_mem.as_ptr(), 4);
        assert!(
            !crate::host::parts_can_join(&[&hl_a], &[&hl_b]),
            "overlapping bottom-of-pipe write and Fill must block coalescing"
        );
        // Readback in A always blocks coalescing with B.
        let (hl_rb, hl_next) = (host.new_list().unwrap(), host.new_list().unwrap());
        let (rb_buf, _) = host.shared_buffer(256).unwrap();
        hl_rb.defer_read_back(rb_buf, shared_mem.as_mut_ptr(), [16, 1, 16]);
        assert!(
            !crate::host::parts_can_join(&[&hl_rb], &[&hl_next]),
            "GPU->CPU readback in earlier part must block coalescing"
        );
        hl_a.close();
        hl_b.close();
        hl_rb.close();
        hl_next.close();

        // --- Case 5 (F4 regression): earlier part resets a currently-met ValueWait ---
        // Initially `*reset_flag == 1` (wait condition EQUAL 1 appears met on CPU).
        // Part 0 resets `*reset_flag = 0` via bottom-of-pipe write, followed by
        // `Wait(reset_flag == 1)`, followed by Part 1 writing `reset_out = 999`.
        // `coalesce_held_runs` MUST NOT bypass the wait using stale CPU memory!
        let mut reset_flag = Box::new(1u32);
        let mut reset_out = Box::new(0u32);
        let l5 = make_list();
        write32_on(l5, &raw mut *reset_flag as u64, 0);
        wait32_on(l5, &raw const *reset_flag as u64, 3, 1);
        write32_on(l5, &raw mut *reset_out as u64, 999);
        close_on(l5);

        let before_f4 = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
        execute(q, 1, &l5);
        crate::host::settle_for_present();
        assert_eq!(*reset_flag, 0, "part 0 must execute and reset flag to 0");
        assert_eq!(
            *reset_out, 0,
            "part 1 must NOT bypass ValueWait using stale pre-part-0 CPU memory"
        );
        assert_eq!(QUEUE_SUBMISSIONS.load(Ordering::Relaxed) - before_f4, 1);

        // Once another queue/CPU restores `*reset_flag = 1`, part 1 runs.
        *reset_flag = 1;
        release_held(false);
        crate::host::settle_for_present();
        assert_eq!(*reset_out, 999, "part 1 must execute once flag becomes 1 again");
        assert_eq!(QUEUE_SUBMISSIONS.load(Ordering::Relaxed) - before_f4, 2);
        release_obj(l5);

        // --- Case 6 (F4/F3 helper regression): `lists_write_range` detects
        // overlapping `defer_copy` destinations and ignores disjoint ones ---
        let (hl_wr_overlap, hl_wr_disjoint) = (
            host.new_list().unwrap(),
            host.new_list().unwrap(),
        );
        let mut war_buf_a = [0u8; 64];
        let mut war_buf_b = [0u8; 64];
        hl_wr_overlap.defer_copy(war_buf_a.as_mut_ptr(), war_buf_b.as_ptr(), 32);
        hl_wr_disjoint.defer_copy(war_buf_b.as_mut_ptr(), war_buf_a.as_ptr(), 32);
        assert!(crate::host::lists_write_range(
            &[&hl_wr_overlap],
            war_buf_a.as_ptr() as usize,
            4,
        ));
        assert!(!crate::host::lists_write_range(
            &[&hl_wr_disjoint],
            war_buf_a.as_ptr() as usize,
            4,
        ));
        hl_wr_overlap.close();
        hl_wr_disjoint.close();

        release_obj(fence);
        release_obj(q);
        release_obj(allocator);
    }

    /// Real-GPU benchmark comparing BEFORE (16 split `ExecuteCommandLists`
    /// submissions) vs AFTER (1 coalesced `ExecuteCommandLists` submission)
    /// for a 16-part cut submission whose waits are already satisfied.
    #[cfg(windows)]
    #[test]
    #[ignore = "hardware benchmark: run with --ignored --nocapture"]
    fn hw_bench_join_parts_before_vs_after() {
        let _guard = QUEUE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(host) = crate::host::host() else {
            eprintln!("no host GPU; skipped");
            return;
        };
        const PARTS: usize = 16;
        const ITERS: usize = 25;

        let mut split_us = Vec::with_capacity(ITERS);
        let mut joined_us = Vec::with_capacity(ITERS);
        let mut lists: Vec<crate::host::HostList> = (0..PARTS)
            .map(|_| {
                let l = host.new_list().unwrap();
                l.close();
                l
            })
            .collect();
        let refs: Vec<&crate::host::HostList> = lists.iter().collect();

        // Warmup
        host.execute(&refs);

        for _ in 0..ITERS {
            let t0 = std::time::Instant::now();
            for r in &refs {
                host.execute(&[*r]);
            }
            split_us.push(t0.elapsed().as_secs_f64() * 1e6);

            let t1 = std::time::Instant::now();
            host.execute(&refs);
            joined_us.push(t1.elapsed().as_secs_f64() * 1e6);
        }

        // Also verify end-to-end `queue::execute` submission count with current VICTORY_JOIN_PARTS.
        let null = std::ptr::null();
        let alloc_iid = guid(0x6102dee4, 0xaf59, 0x4b09, [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24]);
        let list_iid = guid(0x5b160d0f, 0xac1b, 0x4185, [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55]);
        let mut allocator = null;
        assert_eq!(crate::list::create_allocator(null, 0, &alloc_iid, &mut allocator), S_OK);
        let mut q = null;
        let qdesc = QueueDesc { kind: 0, priority: 0, flags: 0, nodes: 0 };
        assert_eq!(create(null, &qdesc, &QUEUE_IIDS[0], &mut q), S_OK);
        let ready = Box::new(1u32);
        let mut sink = Box::new([0u32; PARTS]);
        let mut e2e_us = Vec::with_capacity(ITERS);
        let mut e2e_subs = 0u64;
        for _ in 0..ITERS {
            let mut l = null;
            assert_eq!(crate::list::create_list(null, 0, 0, allocator, null, &list_iid, &mut l), S_OK);
            let write32 = slot::<extern "system" fn(This, u64, u32, u32, u32)>(l, list_slot_offset(l, 0x228));
            let wait32 = slot::<extern "system" fn(This, u64, u32, u64)>(l, list_slot_offset(l, 0x238));
            for p in 0..PARTS {
                if p > 0 {
                    wait32(l, &raw const *ready as u64, 3, 1);
                }
                write32(l, &raw mut sink[p] as u64, p as u32 + 1, 0, 0);
            }
            let close = slot::<extern "system" fn(This) -> HRESULT>(l, 0x48);
            assert_eq!(close(l), S_OK);
            let s0 = QUEUE_SUBMISSIONS.load(Ordering::Relaxed);
            let t = std::time::Instant::now();
            execute(q, 1, &l);
            crate::host::settle_for_present();
            e2e_us.push(t.elapsed().as_secs_f64() * 1e6);
            e2e_subs = QUEUE_SUBMISSIONS.load(Ordering::Relaxed) - s0;
            slot::<extern "system" fn(This) -> u32>(l, 0x10)(l);
        }
        slot::<extern "system" fn(This) -> u32>(q, 0x10)(q);
        slot::<extern "system" fn(This) -> u32>(allocator, 0x10)(allocator);
        lists.clear();

        let median = |v: &mut Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let s_med = median(&mut split_us);
        let j_med = median(&mut joined_us);
        let e2e_med = median(&mut e2e_us);
        eprintln!(
            "Multi-part submission coalescing benchmark ({PARTS} ready parts, {ITERS} iterations, VICTORY_JOIN_PARTS={}):\n  BEFORE (16 split host submissions) : {s_med:.1} us ({:.3} ms)\n  AFTER  (1 coalesced submission)    : {j_med:.1} us ({:.3} ms) ({:.2}x speedup)\n  E2E queue::execute (subs={e2e_subs:2})      : {e2e_med:.1} us ({:.3} ms)",
            u8::from(join_parts()),
            s_med / 1e3,
            j_med / 1e3,
            s_med / j_med.max(1e-6),
            e2e_med / 1e3,
        );
    }
}
