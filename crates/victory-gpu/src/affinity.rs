//! Hybrid CPU-set partitioning and thread pinning for P-core / E-core systems.
//!
//! * `VICTORY_FAST_CORES`: default on (`=0` disables); keeps frame-critical
//!   GPU threads on the highest-efficiency cores (`EfficiencyClass == max`).
//! * `VICTORY_DISK_CORES`: opt-in `slow` (`VICTORY_DISK_CORES=slow`); keeps
//!   background storage decompression workers off the fast cores
//!   (`EfficiencyClass < max`).

use std::ffi::OsStr;
use std::sync::OnceLock;

/// Splits `(cpu_set_id, efficiency_class)` entries into `(fast_ids, slow_ids)`
/// when the processor reports at least two distinct efficiency classes.
///
/// Returns `None` if `entries` is empty or uniform (`min_eff == max_eff`), so
/// homogeneous CPUs and virtualized environments remain no-ops.
pub fn partition_cpu_sets(entries: &[(u32, u8)]) -> Option<(Vec<u32>, Vec<u32>)> {
    if entries.is_empty() {
        return None;
    }
    let mut min_eff = u8::MAX;
    let mut max_eff = u8::MIN;
    for &(_, eff) in entries {
        min_eff = min_eff.min(eff);
        max_eff = max_eff.max(eff);
    }
    if min_eff == max_eff {
        return None;
    }
    let mut fast = Vec::new();
    let mut slow = Vec::new();
    for &(id, eff) in entries {
        if eff == max_eff {
            fast.push(id);
        } else {
            slow.push(id);
        }
    }
    Some((fast, slow))
}

fn fast_cores_from(value: Option<&OsStr>) -> bool {
    value.is_none_or(|v| v.to_str().is_none_or(|s| s.trim() != "0"))
}

fn disk_cores_slow_from(value: Option<&OsStr>) -> bool {
    value.is_some_and(|v| v.to_str().is_some_and(|s| s.trim().eq_ignore_ascii_case("slow")))
}

/// Whether `VICTORY_FAST_CORES` allows pinning hot GPU threads to fast cores.
/// On by default; `VICTORY_FAST_CORES=0` disables.
pub fn fast_cores_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| fast_cores_from(std::env::var_os("VICTORY_FAST_CORES").as_deref()))
}

/// Whether `VICTORY_DISK_CORES=slow` asks background storage workers to stay on
/// efficiency cores. Off by default.
pub fn disk_cores_slow() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| disk_cores_slow_from(std::env::var_os("VICTORY_DISK_CORES").as_deref()))
}

#[cfg(windows)]
mod imp {
    use std::sync::OnceLock;

    use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
    use windows_sys::Win32::System::Kernel::PROCESSOR_NUMBER;
    use windows_sys::Win32::System::SystemInformation::{
        CpuSetInformation, GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentProcessorNumberEx, GetCurrentThread, SetThreadSelectedCpuSets,
    };

    use super::{disk_cores_slow, fast_cores_enabled, partition_cpu_sets};

    #[derive(Clone, Copy)]
    struct CpuSetEntry {
        id: u32,
        group: u16,
        logical_processor: u8,
        efficiency: u8,
    }

    pub(super) struct Topology {
        pub(super) fast_ids: Vec<u32>,
        pub(super) slow_ids: Vec<u32>,
        fast_lps: Vec<(u16, u8)>,
        slow_lps: Vec<(u16, u8)>,
    }

    fn query_system_cpu_sets() -> Option<Vec<CpuSetEntry>> {
        let mut needed: u32 = 0;
        // SAFETY: null buffer with length 0 queries the required byte size.
        let initial_ok = unsafe {
            GetSystemCpuSetInformation(
                std::ptr::null_mut(),
                0,
                &mut needed,
                GetCurrentProcess(),
                0,
            )
        };
        if initial_ok != 0 {
            return None;
        }
        // SAFETY: reads thread-local Win32 error code immediately after failed call.
        if unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER || needed == 0 {
            return None;
        }

        const MAX_RETRIES: usize = 4;
        for _ in 0..MAX_RETRIES {
            let words = usize::try_from(needed).ok()?.div_ceil(size_of::<u64>());
            // 8-byte aligned backing storage matching `SYSTEM_CPU_SET_INFORMATION`.
            let mut buf = vec![0u64; words];
            let cap = u32::try_from(buf.len().saturating_mul(size_of::<u64>())).ok()?;
            let mut returned: u32 = 0;
            // SAFETY: `buf` is 8-byte aligned and writable for `cap` bytes.
            let ok = unsafe {
                GetSystemCpuSetInformation(
                    buf.as_mut_ptr().cast::<SYSTEM_CPU_SET_INFORMATION>(),
                    cap,
                    &mut returned,
                    GetCurrentProcess(),
                    0,
                )
            };
            if ok != 0 {
                let used = (returned as usize).min(cap as usize);
                let base = buf.as_ptr().cast::<u8>();
                let mut entries = Vec::new();
                let mut offset = 0usize;
                while offset + size_of::<u32>() <= used {
                    // SAFETY: `offset + 4 <= used <= cap`, within `buf`.
                    let entry_size =
                        unsafe { base.add(offset).cast::<u32>().read_unaligned() } as usize;
                    if entry_size == 0 || offset.saturating_add(entry_size) > used {
                        break;
                    }
                    if entry_size >= size_of::<SYSTEM_CPU_SET_INFORMATION>() {
                        // SAFETY: `offset + entry_size <= used` and `entry_size >= struct size`.
                        let info = unsafe {
                            base.add(offset)
                                .cast::<SYSTEM_CPU_SET_INFORMATION>()
                                .read_unaligned()
                        };
                        if info.Type == CpuSetInformation {
                            // SAFETY: `Type == CpuSetInformation` selects the `CpuSet` union variant.
                            let cs = unsafe { info.Anonymous.CpuSet };
                            entries.push(CpuSetEntry {
                                id: cs.Id,
                                group: cs.Group,
                                logical_processor: cs.LogicalProcessorIndex,
                                efficiency: cs.EfficiencyClass,
                            });
                        }
                    }
                    offset += entry_size;
                }
                return Some(entries);
            }
            // SAFETY: reads thread-local Win32 error code immediately after failed call.
            if unsafe { GetLastError() } == ERROR_INSUFFICIENT_BUFFER && returned > cap {
                needed = returned;
                continue;
            }
            return None;
        }
        None
    }

    pub(super) fn topology() -> Option<&'static Topology> {
        static TOPOLOGY: OnceLock<Option<Topology>> = OnceLock::new();
        TOPOLOGY
            .get_or_init(|| {
                let entries = query_system_cpu_sets()?;
                let raw: Vec<(u32, u8)> = entries.iter().map(|e| (e.id, e.efficiency)).collect();
                let (fast_ids, slow_ids) = partition_cpu_sets(&raw)?;
                let max_eff = entries.iter().map(|e| e.efficiency).max()?;
                let mut fast_lps = Vec::with_capacity(fast_ids.len());
                let mut slow_lps = Vec::with_capacity(slow_ids.len());
                for e in &entries {
                    let lp = (e.group, e.logical_processor);
                    if e.efficiency == max_eff {
                        fast_lps.push(lp);
                    } else {
                        slow_lps.push(lp);
                    }
                }
                tracing::info!(
                    fast = fast_ids.len(),
                    slow = slow_ids.len(),
                    fast_enabled = fast_cores_enabled(),
                    disk_slow = disk_cores_slow(),
                    "hybrid CPU topology detected"
                );
                Some(Topology {
                    fast_ids,
                    slow_ids,
                    fast_lps,
                    slow_lps,
                })
            })
            .as_ref()
    }

    pub(super) fn pin_current_thread(role: &str, tier: &str, ids: &[u32]) -> bool {
        let Ok(count) = u32::try_from(ids.len()) else {
            return false;
        };
        if count == 0 {
            return false;
        }
        // SAFETY: `GetCurrentThread()` is valid for the calling thread and `ids` is valid for `count` u32s.
        let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), count) } != 0;
        if ok {
            tracing::debug!(role, tier, cores = count, "pinned thread to CPU sets");
        } else {
            // SAFETY: reads thread-local Win32 error code immediately after failed call.
            let err = unsafe { GetLastError() };
            tracing::warn!(role, tier, error = err, "SetThreadSelectedCpuSets failed");
        }
        ok
    }

    pub(super) fn current_processor() -> (u16, u8) {
        let mut pn = PROCESSOR_NUMBER {
            Group: 0,
            Number: 0,
            Reserved: 0,
        };
        // SAFETY: `pn` is a valid out-pointer.
        unsafe { GetCurrentProcessorNumberEx(&mut pn) };
        (pn.Group, pn.Number)
    }

    impl Topology {
        pub(super) fn is_fast_lp(&self, group: u16, lp: u8) -> Option<bool> {
            if self.fast_lps.contains(&(group, lp)) {
                Some(true)
            } else if self.slow_lps.contains(&(group, lp)) {
                Some(false)
            } else {
                None
            }
        }
    }
}

/// Pins the calling thread to the highest-efficiency CPU sets on hybrid CPUs.
///
/// Enabled by default; set `VICTORY_FAST_CORES=0` to leave scheduling to the OS.
/// Returns `true` if thread CPU-set selection was applied.
pub fn keep_on_fast_cores(role: &str) -> bool {
    #[cfg(windows)]
    {
        if !fast_cores_enabled() {
            return false;
        }
        let Some(topo) = imp::topology() else {
            return false;
        };
        imp::pin_current_thread(role, "fast", &topo.fast_ids)
    }
    #[cfg(not(windows))]
    {
        let _ = role;
        false
    }
}

/// Pins the calling thread to lower-efficiency CPU sets on hybrid CPUs when
/// `VICTORY_DISK_CORES=slow` is set.
///
/// Off by default; returns `true` if thread CPU-set selection was applied.
pub fn keep_on_slow_cores(role: &str) -> bool {
    #[cfg(windows)]
    {
        if !disk_cores_slow() {
            return false;
        }
        let Some(topo) = imp::topology() else {
            return false;
        };
        imp::pin_current_thread(role, "slow", &topo.slow_ids)
    }
    #[cfg(not(windows))]
    {
        let _ = role;
        false
    }
}

/// Returns `(fast_lp_count, slow_lp_count)` if running on a hybrid Windows CPU.
pub fn hybrid_cpu_counts() -> Option<(usize, usize)> {
    #[cfg(windows)]
    {
        imp::topology().map(|t| (t.fast_ids.len(), t.slow_ids.len()))
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Returns `(group, logical_processor, Some(is_fast))` for the calling thread's
/// current processor, or `None` on non-Windows / uniform CPUs.
#[cfg(windows)]
pub fn current_processor_tier() -> Option<(u16, u8, bool)> {
    let topo = imp::topology()?;
    let (group, lp) = imp::current_processor();
    topo.is_fast_lp(group, lp).map(|fast| (group, lp, fast))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_cpu_sets_handles_empty_and_uniform() {
        assert_eq!(partition_cpu_sets(&[]), None);
        assert_eq!(partition_cpu_sets(&[(0x100, 0)]), None);
        assert_eq!(
            partition_cpu_sets(&[(0x100, 1), (0x101, 1), (0x102, 1)]),
            None
        );
    }

    #[test]
    fn partition_cpu_sets_splits_hybrid_and_multi_tier() {
        let mut entries: Vec<(u32, u8)> = (0..12).map(|i| (0x100 + i, 1)).collect();
        entries.extend((12..16).map(|i| (0x100 + i, 0)));
        let (fast, slow) = partition_cpu_sets(&entries).expect("hybrid");
        assert_eq!(fast, (0..12).map(|i| 0x100 + i).collect::<Vec<_>>());
        assert_eq!(slow, (12..16).map(|i| 0x100 + i).collect::<Vec<_>>());

        let three_tier = [(10, 0), (11, 2), (12, 1), (13, 2)];
        let (fast, slow) = partition_cpu_sets(&three_tier).expect("3-tier");
        assert_eq!(fast, vec![11, 13]);
        assert_eq!(slow, vec![10, 12]);
    }

    #[test]
    fn env_var_parsing_respects_defaults_and_overrides() {
        assert!(fast_cores_from(None));
        assert!(fast_cores_from(Some(OsStr::new(""))));
        assert!(fast_cores_from(Some(OsStr::new("1"))));
        assert!(!fast_cores_from(Some(OsStr::new("0"))));
        assert!(!fast_cores_from(Some(OsStr::new(" 0 "))));

        assert!(!disk_cores_slow_from(None));
        assert!(!disk_cores_slow_from(Some(OsStr::new(""))));
        assert!(!disk_cores_slow_from(Some(OsStr::new("0"))));
        assert!(!disk_cores_slow_from(Some(OsStr::new("fast"))));
        assert!(disk_cores_slow_from(Some(OsStr::new("slow"))));
        assert!(disk_cores_slow_from(Some(OsStr::new(" SLOW "))));
    }

    #[cfg(windows)]
    #[test]
    fn live_hybrid_pinning_schedules_fast_and_slow_threads() {
        const CHILD: &str = "VICTORY_AFFINITY_TEST_CHILD";
        match std::env::var(CHILD).ok().as_deref() {
            Some("slow") => {
                let Some((fast, slow)) = hybrid_cpu_counts() else {
                    return;
                };
                assert!(fast > 0 && slow > 0);
                assert!(fast_cores_enabled());
                assert!(disk_cores_slow());

                let fast_handle = std::thread::spawn(|| {
                    assert!(keep_on_fast_cores("test-fast"));
                    let mut fast_hits = 0usize;
                    let mut slow_hits = 0usize;
                    for _ in 0..64 {
                        std::hint::spin_loop();
                        if let Some((_, _, is_fast)) = current_processor_tier() {
                            if is_fast {
                                fast_hits += 1;
                            } else {
                                slow_hits += 1;
                            }
                        }
                        std::thread::yield_now();
                    }
                    (fast_hits, slow_hits)
                });
                let slow_handle = std::thread::spawn(|| {
                    assert!(keep_on_slow_cores("test-slow"));
                    let mut fast_hits = 0usize;
                    let mut slow_hits = 0usize;
                    for _ in 0..64 {
                        std::hint::spin_loop();
                        if let Some((_, _, is_fast)) = current_processor_tier() {
                            if is_fast {
                                fast_hits += 1;
                            } else {
                                slow_hits += 1;
                            }
                        }
                        std::thread::yield_now();
                    }
                    (fast_hits, slow_hits)
                });

                let (ff, fs) = fast_handle.join().unwrap();
                let (sf, ss) = slow_handle.join().unwrap();
                assert_eq!(ff, 64, "fast thread hit slow core ({fs} slow, {ff} fast)");
                assert_eq!(ss, 64, "slow thread hit fast core ({sf} fast, {ss} slow)");
                return;
            }
            Some("fast_off") => {
                assert!(!fast_cores_enabled());
                assert!(!disk_cores_slow());
                assert!(!keep_on_fast_cores("test-disabled"));
                assert!(!keep_on_slow_cores("test-disabled"));
                return;
            }
            _ => {}
        }

        let Some((fast, slow)) = hybrid_cpu_counts() else {
            eprintln!("uniform CPU topology: skipping live hybrid pinning check");
            return;
        };
        assert!(fast > 0 && slow > 0);

        let fast_handle = std::thread::spawn(|| {
            assert!(keep_on_fast_cores("test-default-fast"));
            let mut fast_hits = 0usize;
            for _ in 0..32 {
                if let Some((_, _, is_fast)) = current_processor_tier()
                    && is_fast
                {
                    fast_hits += 1;
                }
                std::thread::yield_now();
            }
            fast_hits
        });
        assert_eq!(fast_handle.join().unwrap(), 32);

        let exe = std::env::current_exe().unwrap();
        let test_name = "affinity::tests::live_hybrid_pinning_schedules_fast_and_slow_threads";

        let slow_status = std::process::Command::new(&exe)
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD, "slow")
            .env("VICTORY_FAST_CORES", "1")
            .env("VICTORY_DISK_CORES", "slow")
            .status()
            .unwrap();
        assert!(slow_status.success(), "slow child failed: {slow_status}");

        let off_status = std::process::Command::new(&exe)
            .args(["--exact", test_name, "--nocapture"])
            .env(CHILD, "fast_off")
            .env("VICTORY_FAST_CORES", "0")
            .env_remove("VICTORY_DISK_CORES")
            .status()
            .unwrap();
        assert!(off_status.success(), "fast_off child failed: {off_status}");
    }
}
