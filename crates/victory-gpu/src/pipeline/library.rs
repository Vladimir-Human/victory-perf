//! Compiled host pipelines kept on disk, and the title pipelines worth
//! starting early (decision 0007, "Pipeline cache").
//!
//! 1. Every host pipeline goes into the host's pipeline library, named by
//!    a hash of all that its host description holds (the translated
//!    shaders' code, which changes when a translation does, our root
//!    signature, the graphics state). The library is saved under the cache
//!    folder; the host refuses a library of another adapter or driver, and
//!    an entry whose description differs, so neither is ever used.
//! 2. The title pipelines a run drew or dispatched with are listed by a key
//!    known when the title creates them (their shader containers and
//!    graphics state); the next run hands those to the workers at
//!    creation, during loading.
//!
//! Library calls overlap: the host documents the library as safe to use
//! from several threads ("The pipeline library is thread-safe to use, and
//! will internally synchronize as necessary", `CreatePipelineLibrary`,
//! Thread Safety), with one exception kept here: two loads of the same
//! pipeline are made one after the other ([`same_name`]). Under one lock of
//! ours for every call, the eight workers made pipelines one at a time:
//! 87 to 96 of 110 looks at each worker found it waiting for that lock
//! (run 49), and an earlier run's 2,300 pipelines, all in the library, were
//! still being loaded 46 s into the opening cutscene.
//! `VICTORY_LIBRARY_LOCK=1`: one call at a time, as before.
//!
//! The library's file is mapped, not read into memory ([`Blob`]): the host
//! keeps pointing into the bytes it was made from for as long as it lives,
//! and read into a buffer they were 1,352 MB of the process's own memory
//! for the whole run (run 60), all of it in physical memory. Mapped, the
//! pages are the file's: only those the host reads are in memory, and the
//! system can drop them without writing them anywhere. A save writes the
//! new library through a mapping of the new file in the same way (a buffer
//! of the library's size was made, filled and written out at each save).
//! `VICTORY_LIBRARY_MAP=0`: read and written through buffers, as before.
//!
//! The mapped file's handle stays open for as long as the mapping lives
//! ([`holds`]). A save renames the new library over the file this run has
//! mapped, and with the handle closed Windows then gives up the old file's
//! contents: every page of the mapping that is not in memory at that
//! moment, or leaves memory later (the trim below, or Windows taking pages
//! back), reads as zeros from then on. The host's library reads such pages
//! as stored pipelines; the driver, handed a pipeline of zeros, removes the
//! device with DXGI_ERROR_DRIVER_INTERNAL_ERROR (runs 72, 73 and 82, each
//! within 5 ms of the first library call after a save and a trim; and the
//! library run 77 saved from such pages was refused by run 78). Measured
//! with a 64 MB file written past the system's cache, mapped and replaced:
//! 16,352 of 16,384 pages read zeros with the handle closed, none with it
//! open (the test below). `VICTORY_LIBRARY_HANDLE=0`: closed, as before.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, RwLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HSTRING, Interface, PCWSTR};

static DIR: OnceLock<PathBuf> = OnceLock::new();

/// Keeps the pipeline library and the early list in `dir` from now on.
pub fn use_pipeline_cache(dir: PathBuf) {
    let _ = DIR.set(dir);
}

/// Host pipelines created, loaded from the library, and title pipelines
/// started at creation, since the start.
pub(crate) static MADE: AtomicU64 = AtomicU64::new(0);
pub(crate) static LOADED: AtomicU64 = AtomicU64::new(0);
pub(crate) static EARLY: AtomicU64 = AtomicU64::new(0);
/// For the memory lines: the bytes of the library's file, kept in memory
/// while the library lives, and the bytes the last save wrote (a buffer of
/// that size is made, filled and written every time pipelines were added,
/// at most every 20 s).
pub(crate) static BLOB: AtomicU64 = AtomicU64::new(0);
pub(crate) static SAVED: AtomicU64 = AtomicU64::new(0);

/// Whether the library's file is mapped ([`Blob`]) and saved through a
/// mapping. On by default; `VICTORY_LIBRARY_MAP=0` uses buffers, as before.
fn mapped() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_MAP").is_none_or(|v| v != "0"))
}

/// Whether the mapped library file's handle is kept open while it is
/// mapped (see the module's notes: without it a save in the same run turns
/// the mapping's pages into zeros as they leave memory). On by default;
/// `VICTORY_LIBRARY_HANDLE=0` closes it after mapping, as before.
fn holds() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_HANDLE").is_none_or(|v| v != "0"))
}

/// A whole file mapped into memory: read-only (the library's file) or
/// writable (the file a save fills). Unmapped when dropped.
struct Mapped {
    at: windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
    /// The file, for a mapping made by [`Mapped::open`] ([`holds`]): open
    /// until after the view is unmapped, so that the file a rename took
    /// the name from keeps its contents for the view.
    _file: Option<std::fs::File>,
}

impl Mapped {
    /// Maps `file` (`len` bytes, not zero) for reading, or for reading and
    /// writing.
    fn map(file: &std::fs::File, len: usize, write: bool) -> Option<Mapped> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_READ, FILE_MAP_WRITE, MapViewOfFile, PAGE_READONLY,
            PAGE_READWRITE,
        };
        if len == 0 {
            return None;
        }
        // SAFETY: a file we hold open; the view keeps the file's pages
        // after the mapping object and the file are closed.
        let at = unsafe {
            let mapping = CreateFileMappingW(
                file.as_raw_handle(),
                std::ptr::null(),
                if write { PAGE_READWRITE } else { PAGE_READONLY },
                0,
                0,
                std::ptr::null(),
            );
            if mapping.is_null() {
                return None;
            }
            let at = MapViewOfFile(
                mapping,
                if write { FILE_MAP_WRITE } else { FILE_MAP_READ },
                0,
                0,
                0,
            );
            CloseHandle(mapping);
            at
        };
        (!at.Value.is_null()).then_some(Mapped {
            at,
            len,
            _file: None,
        })
    }

    /// The file at `path`, mapped for reading; `None` if it is not there,
    /// is empty or cannot be mapped.
    fn open(path: &std::path::Path) -> Option<Mapped> {
        Mapped::open_holding(path, holds())
    }

    /// As [`Mapped::open`]; `hold` says whether the file stays open with
    /// the mapping.
    fn open_holding(path: &std::path::Path, hold: bool) -> Option<Mapped> {
        // Opened as the standard library opens a file to read: others may
        // read, write, rename and delete it meanwhile, so a save still
        // renames the new library over this one.
        let file = std::fs::File::open(path).ok()?;
        let len = usize::try_from(file.metadata().ok()?.len()).ok()?;
        let mut mapped = Mapped::map(&file, len, false)?;
        if hold {
            mapped._file = Some(file);
        }
        Some(mapped)
    }

    /// A new file of `len` bytes at `path` (replacing what is there),
    /// mapped for writing.
    fn create(path: &std::path::Path, len: usize) -> Option<Mapped> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .ok()?;
        file.set_len(len as u64).ok()?;
        Mapped::map(&file, len, true)
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: the view holds `len` readable bytes until dropped.
        unsafe { std::slice::from_raw_parts(self.at.Value.cast(), self.len) }
    }

    /// Only for a view made by [`Mapped::create`].
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: a writable view of `len` bytes, ours alone.
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: our view.
        unsafe { windows_sys::Win32::System::Memory::UnmapViewOfFile(self.at) };
    }
}

/// The bytes a library is made from: the host reads them for as long as
/// the library lives, so they are kept beside it.
enum Blob {
    /// No file (an empty library).
    None,
    /// The file read into memory (`VICTORY_LIBRARY_MAP=0`, tests, or a file
    /// that could not be mapped).
    Held(Vec<u8>),
    /// The file itself, mapped.
    Mapped(Mapped),
}

impl Blob {
    /// The library file in `dir`: a library a save left beside it
    /// ([`NEXT_FILE`]) takes its place first.
    #[cfg(test)]
    fn of(dir: &std::path::Path) -> Blob {
        Blob::of_files(dir, &LibraryFiles::legacy()).0
    }

    /// Opens the adapter-scoped library file described by `files` in `dir`,
    /// promoting any pending `.next` file first. Returns `(blob, from_legacy)`
    /// where `from_legacy` is `true` if the bytes came read-only from the
    /// legacy unsuffixed [`LIBRARY_FILE`] because this adapter had no keyed
    /// file yet.
    fn of_files(dir: &std::path::Path, files: &LibraryFiles) -> (Blob, bool) {
        let path = dir.join(&files.bin);
        let next = dir.join(&files.next);
        if next.exists() && std::fs::rename(&next, &path).is_err() {
            tracing::warn!(
                file = %files.bin,
                "the host pipeline library saved last could not take the old one's place"
            );
        }
        if let Some(blob) = Blob::open_path(&path) {
            return (blob, false);
        }
        if files.legacy_fallback && !path.exists() && !next.exists() {
            let legacy = dir.join(LIBRARY_FILE);
            if let Some(blob) = Blob::open_path(&legacy) {
                return (blob, true);
            }
        }
        (Blob::None, false)
    }

    fn open_path(path: &std::path::Path) -> Option<Blob> {
        if mapped()
            && let Some(file) = Mapped::open(path)
        {
            return Some(Blob::Mapped(file));
        }
        match std::fs::read(path) {
            Ok(bytes) if !bytes.is_empty() => Some(Blob::Held(bytes)),
            _ => None,
        }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Blob::None => &[],
            Blob::Held(bytes) => bytes,
            Blob::Mapped(file) => file.bytes(),
        }
    }
}

/// The mapped library file's view (address, bytes; 0: not mapped), when a
/// pipeline was last loaded from the library (`crate::host::now_ms`), and
/// whether the view's pages were let go since ([`trim`]).
static VIEW: [std::sync::atomic::AtomicUsize; 2] =
    [const { std::sync::atomic::AtomicUsize::new(0) }; 2];
static LOADED_AT: AtomicU64 = AtomicU64::new(0);
static TRIMMED: AtomicBool = AtomicBool::new(false);
/// The bytes of the view let go so far, for the memory line.
pub(crate) static TRIMMED_BYTES: AtomicU64 = AtomicU64::new(0);
/// When the view's pages were last let go (`crate::host::now_ms`), and how
/// often and when a save of this run last renamed a new library over the
/// mapped file: for [`removal_note`].
static TRIMMED_AT: AtomicU64 = AtomicU64::new(0);
static REPLACED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The library calls running now: thread, what (load, make, store,
/// serialise), the pipeline, and since when (`crate::host::now_us`).
static CALLS: Mutex<Vec<(u32, &'static str, String, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// The pipeline this thread is making (its shaders), for [`CALLS`].
    static LABEL: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) };
}

/// Names the pipeline this thread makes next.
pub(crate) fn label(text: String) {
    LABEL.with(|l| *l.borrow_mut() = text);
}

/// A library call of this thread, listed in [`CALLS`] until dropped.
struct Call(u32, u64);

fn call(what: &'static str, name: &str) -> Call {
    let (thread, since) = (crate::host::thread_id(), crate::host::now_us());
    let label = LABEL.with(|l| l.borrow().clone());
    CALLS.lock().unwrap_or_else(|e| e.into_inner()).push((
        thread,
        what,
        format!("{name} {label}"),
        since,
    ));
    Call(thread, since)
}

impl Drop for Call {
    fn drop(&mut self) {
        CALLS
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|c| (c.0, c.3) != (self.0, self.1));
    }
}

/// For the report when the host device is removed: the library calls
/// running at that moment, whether this run replaced the mapped file and
/// let its pages go, and how many pages of the mapping read as zeros now.
/// A library holds no page of zeros (the 259 MB one of run 83 has no run of
/// 2,048 zero bytes): each such page is one whose contents were lost (see
/// the module's notes), and a pipeline stored there is what the driver was
/// handed.
pub(crate) fn removal_note() -> String {
    use std::fmt::Write;
    let now = crate::host::now_us();
    let mut text = String::from("pipeline library: ");
    {
        let calls = CALLS.lock().unwrap_or_else(|e| e.into_inner());
        if calls.is_empty() {
            text.push_str("no call running");
        }
        for (i, (thread, what, name, since)) in calls.iter().enumerate() {
            let _ = write!(
                text,
                "{}{what} of {name} running for {:.1} ms on thread {thread}",
                if i == 0 { "" } else { ", " },
                now.saturating_sub(*since) as f64 / 1000.0
            );
        }
    }
    let at = VIEW[0].load(Ordering::Acquire);
    let len = VIEW[1].load(Ordering::Relaxed);
    if at == 0 {
        text.push_str("; its file is not mapped");
        return text;
    }
    let ago = |ms: u64| crate::host::now_ms().saturating_sub(ms) as f64 / 1000.0;
    let _ = write!(
        text,
        "; its file is mapped ({len} bytes), the handle {}",
        if holds() {
            "kept open"
        } else {
            "closed (VICTORY_LIBRARY_HANDLE=0)"
        }
    );
    match REPLACED[0].load(Ordering::Relaxed) {
        0 => text.push_str("; this run has not replaced the file"),
        n => {
            let _ = write!(
                text,
                "; this run replaced the file under the mapping {n} times, last {:.1} s ago",
                ago(REPLACED[1].load(Ordering::Relaxed))
            );
        }
    }
    if TRIMMED_BYTES.load(Ordering::Relaxed) > 0 {
        let _ = write!(
            text,
            "; its pages were last let go {:.1} s ago",
            ago(TRIMMED_AT.load(Ordering::Relaxed))
        );
    }
    // Within two seconds: pages not in memory come from the disk.
    let started = std::time::Instant::now();
    // SAFETY: the view lives as long as the library, which is never dropped.
    let bytes = unsafe { std::slice::from_raw_parts(at as *const u8, len) };
    let (mut zero, mut looked, mut first) = (0u64, 0u64, None);
    for (n, page) in bytes.chunks(4096).enumerate() {
        if n % 4096 == 0 && started.elapsed() > std::time::Duration::from_secs(2) {
            break;
        }
        looked += 1;
        if page.iter().all(|&b| b == 0) {
            zero += 1;
            if first.is_none() {
                first = Some(n);
            }
        }
    }
    let _ = write!(
        text,
        "; {zero} of the {looked} pages looked at read as zeros now"
    );
    if let Some(first) = first {
        let _ = write!(
            text,
            " (the first is page {first}): the library's contents were lost there"
        );
    }
    text
}

/// Lets the mapped library file's pages go from the process's memory once
/// no pipeline has been loaded for ten seconds. The host reads nearly the
/// whole file while the early pipelines load (run 65: all 1,490 MB of it
/// were in the process's memory a minute in, and 1,300 MB still at the
/// end, on a PC with 2 GB free); after that it reads the file again only
/// when a new pipeline loads. Pages let go stay in Windows' file cache
/// while nothing else wants the memory, and come back on the next read.
/// Called once a second by the sweep; does its work on a thread of its own
/// (hundreds of thousands of pages), once after each burst of loads.
/// `VICTORY_LIBRARY_TRIM=0`: the pages stay until Windows takes them.
pub(crate) fn trim() {
    static ON: OnceLock<bool> = OnceLock::new();
    let at = VIEW[0].load(Ordering::Acquire);
    let len = VIEW[1].load(Ordering::Relaxed);
    if at == 0
        || TRIMMED.load(Ordering::Relaxed)
        || !*ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_TRIM").is_none_or(|v| v != "0"))
    {
        return;
    }
    let quiet_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    if crate::host::now_ms() < LOADED_AT.load(Ordering::Relaxed) + quiet_ms
        || TRIMMED.swap(true, Ordering::Relaxed)
    {
        return;
    }
    let work = move || {
        // SAFETY: the view lives as long as the library, which is never
        // dropped. Unlocking pages that are not locked fails, as expected,
        // and takes them out of the working set: the documented way to let
        // pages of a mapped file go.
        unsafe { windows_sys::Win32::System::Memory::VirtualUnlock(at as *const _, len) };
        TRIMMED_BYTES.fetch_add(len as u64, Ordering::Relaxed);
        TRIMMED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
        crate::host::trail(|| {
            format!("pipeline library: the mapped file's pages let go from memory ({len} bytes)")
        });
    };
    if std::thread::Builder::new()
        .name("victory-library-trim".into())
        .spawn(work)
        .is_err()
    {
        TRIMMED.store(false, Ordering::Relaxed);
    }
}

/// Stable identity of the host D3D12 adapter, used to separate on-disk
/// pipeline libraries across adapters (e.g. hardware GPU vs software WARP or
/// another GPU) without encoding per-boot LUIDs or driver versions that would
/// orphan old caches on driver updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdapterKey {
    vendor: u32,
    device: u32,
    revision: u32,
    software: bool,
}

impl AdapterKey {
    fn slug(self) -> String {
        format!(
            "{:04x}-{:04x}-r{:02x}-{}",
            self.vendor,
            self.device,
            self.revision,
            if self.software { "sw" } else { "hw" },
        )
    }
}

/// Queries the adapter identity (`VendorId`, `DeviceId`, `Revision`, software
/// flag) behind `device`. Returns `None` if DXGI cannot enumerate the LUID.
fn adapter_key(device: &ID3D12Device1) -> Option<AdapterKey> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
        IDXGIFactory4,
    };
    // SAFETY: plain DXGI query for the device's own adapter LUID.
    unsafe {
        let luid = device.GetAdapterLuid();
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(luid).ok()?;
        let desc = adapter.GetDesc1().ok()?;
        Some(AdapterKey {
            vendor: desc.VendorId,
            device: desc.DeviceId,
            revision: desc.Revision,
            software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        })
    }
}

/// Filenames inside the pipeline cache folder for one adapter's library.
#[derive(Clone, Debug, PartialEq, Eq)]
struct LibraryFiles {
    bin: String,
    next: String,
    tmp: String,
    /// Whether an existing legacy unsuffixed `host-pipelines.bin` may be read
    /// (read-only) when this adapter has no keyed file yet. Disabled for
    /// software adapters so WARP never reads or maps a hardware cache.
    legacy_fallback: bool,
}

impl LibraryFiles {
    fn legacy() -> Self {
        Self {
            bin: LIBRARY_FILE.to_owned(),
            next: NEXT_FILE.to_owned(),
            tmp: "host-pipelines.tmp".to_owned(),
            legacy_fallback: false,
        }
    }

    fn for_adapter(key: Option<AdapterKey>) -> Self {
        match key {
            Some(k) => {
                let slug = k.slug();
                Self {
                    bin: format!("host-pipelines-{slug}.bin"),
                    next: format!("host-pipelines-{slug}.next"),
                    tmp: format!("host-pipelines-{slug}.tmp"),
                    legacy_fallback: !k.software,
                }
            }
            None => Self::legacy(),
        }
    }
}

struct Library {
    library: ID3D12PipelineLibrary1,
    /// The blob the library was made from: it must outlive the library.
    _blob: Blob,
    /// Adapter-scoped filenames this library saves to.
    files: LibraryFiles,
    /// Whether `_blob` was opened read-only from legacy [`LIBRARY_FILE`] rather
    /// than `files.bin` (saving writes `files.bin`, leaving the mapped legacy
    /// file unreplaced).
    from_legacy: bool,
    /// Pipelines stored since the last save.
    added: AtomicU64,
    /// Stores share it and a save takes it alone: the size the library
    /// asks for must still be its size when it is written out.
    saving: RwLock<()>,
}

// SAFETY: D3D12 objects are free-threaded, and the library synchronises
// its own calls (see the module's notes); what it does not, two loads of
// one name, is serialised by `same_name`.
unsafe impl Send for Library {}
// SAFETY: as above; the other fields are an atomic and a lock.
unsafe impl Sync for Library {}

/// Held by a load of the pipeline named `name`: loads of the same name
/// wait for each other (the host's one exception to the library's thread
/// safety: such a load may change that pipeline's entry). Names share 64
/// locks, so two different names seldom wait.
fn same_name(name: &str) -> MutexGuard<'static, ()> {
    static LOADING: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];
    let at = name.bytes().fold(0usize, |h, b| {
        h.wrapping_mul(31).wrapping_add(usize::from(b))
    });
    LOADING[at % LOADING.len()]
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// With `VICTORY_LIBRARY_LOCK=1`: held by every library call, as before.
fn one_call() -> Option<MutexGuard<'static, ()>> {
    static ON: OnceLock<bool> = OnceLock::new();
    static CALLING: Mutex<()> = Mutex::new(());
    ON.get_or_init(|| std::env::var_os("VICTORY_LIBRARY_LOCK").is_some_and(|v| v == "1"))
        .then(|| CALLING.lock().unwrap_or_else(|e| e.into_inner()))
}

const LIBRARY_FILE: &str = "host-pipelines.bin";
/// A saved library that could not replace [`LIBRARY_FILE`] while this run
/// had it mapped (a system that refuses to replace a mapped file): the next
/// start renames it over the library first ([`Blob::of`]).
const NEXT_FILE: &str = "host-pipelines.next";
const EARLY_FILE: &str = "early-pipelines.txt";
/// Title pipelines listed for an early start, at most.
const EARLY_MAX: usize = 8192;

fn library() -> Option<&'static Library> {
    static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let device = crate::host::host()?.device.cast::<ID3D12Device1>().ok()?;
            let files = LibraryFiles::for_adapter(adapter_key(&device));
            let (blob, from_legacy) = DIR
                .get()
                .map_or((Blob::None, false), |d| Blob::of_files(d, &files));
            let library = open_with(&device, blob, files, from_legacy);
            if let Some(lib) = &library {
                let bytes = lib._blob.bytes().len();
                if bytes > 0 {
                    BLOB.store(bytes as u64, Ordering::Relaxed);
                }
                if let Blob::Mapped(file) = &lib._blob {
                    VIEW[1].store(file.len, Ordering::Relaxed);
                    VIEW[0].store(file.at.Value as usize, Ordering::Release);
                    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
                }
            } else {
                tracing::warn!("no host pipeline library: pipelines are not kept");
            }
            library
        })
        .as_ref()
}

/// The library in `blob`, or an empty one if the host rejects the blob (an
/// adapter or driver change, damage).
#[cfg(test)]
fn open(device: &ID3D12Device1, blob: Blob) -> Option<Library> {
    open_with(device, blob, LibraryFiles::legacy(), false)
}

fn open_with(
    device: &ID3D12Device1,
    blob: Blob,
    files: LibraryFiles,
    from_legacy: bool,
) -> Option<Library> {
    let make = |bytes: &[u8]| -> windows::core::Result<ID3D12PipelineLibrary1> {
        // SAFETY: the bytes outlive the library (kept beside it).
        unsafe {
            device
                .CreatePipelineLibrary::<ID3D12PipelineLibrary>(bytes)?
                .cast()
        }
    };
    if !blob.bytes().is_empty() {
        match make(blob.bytes()) {
            Ok(library) => {
                let bytes = blob.bytes().len();
                tracing::info!(
                    bytes,
                    file = %files.bin,
                    from_legacy,
                    mapped = matches!(blob, Blob::Mapped(_)),
                    "host pipeline library loaded"
                );
                return Some(Library {
                    library,
                    _blob: blob,
                    files,
                    from_legacy,
                    added: AtomicU64::new(0),
                    saving: RwLock::new(()),
                });
            }
            Err(e) => {
                tracing::info!(
                    error = %e,
                    file = %files.bin,
                    from_legacy,
                    "host pipeline library rejected; starting an empty one"
                )
            }
        }
    } else {
        tracing::info!(file = %files.bin, "no host pipeline library yet; starting an empty one");
    }
    // The rejected bytes are let go (a mapped file is unmapped) before the
    // empty library is made: nothing reads them any more.
    drop(blob);
    let library = make(&[]).ok()?;
    Some(Library {
        library,
        _blob: Blob::None,
        files,
        from_legacy: false,
        added: AtomicU64::new(0),
        saving: RwLock::new(()),
    })
}

/// A pipeline's name in the library: the hash of what its host
/// description holds.
pub(crate) fn name(kind: &str, root: &[u8], code: &[&[u8]], state: &str) -> String {
    let mut h = blake3::Hasher::new();
    for part in [kind.as_bytes(), root]
        .into_iter()
        .chain(code.iter().copied())
        .chain([state.as_bytes()])
    {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    h.finalize().to_hex()[..32].to_owned()
}

/// The pipeline named `name`: loaded from the library by `load` if it is
/// there with the same description, else made by `create` and stored.
pub(crate) fn pipeline(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<ID3D12PipelineState> {
    pipeline_with_status(name, load, create).map(|(pso, _)| pso)
}

/// Like [`pipeline`], also reporting whether the pipeline was loaded from the
/// library (`true`) or freshly compiled and stored (`false`).
pub(crate) fn pipeline_with_status(
    name: &str,
    load: impl FnOnce(&ID3D12PipelineLibrary1, PCWSTR) -> windows::core::Result<ID3D12PipelineState>,
    create: impl FnOnce() -> windows::core::Result<ID3D12PipelineState>,
) -> windows::core::Result<(ID3D12PipelineState, bool)> {
    let wide = HSTRING::from(name);
    let Some(library) = library() else {
        MADE.fetch_add(1, Ordering::Relaxed);
        return create().map(|pso| (pso, false));
    };
    let began = std::time::Instant::now();
    let loaded = {
        let _all = one_call();
        let _one = same_name(name);
        let _call = call("load", name);
        load(&library.library, PCWSTR(wide.as_ptr()))
    };
    // The load read the library's file (see `trim`).
    LOADED_AT.store(crate::host::now_ms(), Ordering::Relaxed);
    TRIMMED.store(false, Ordering::Relaxed);
    let refused = match loaded {
        Ok(pso) => {
            LOADED.fetch_add(1, Ordering::Relaxed);
            crate::host::trail(|| {
                let label = LABEL.with(|l| l.borrow().clone());
                format!(
                    "pipeline library: loaded {label} in {:.1} ms",
                    began.elapsed().as_secs_f64() * 1e3
                )
            });
            return Ok((pso, true));
        }
        Err(e) => e,
    };
    // Compiled outside any lock: compiles take up to seconds.
    let pso = {
        let _call = call("make", name);
        create()?
    };
    MADE.fetch_add(1, Ordering::Relaxed);
    crate::host::trail(|| {
        let label = LABEL.with(|l| l.borrow().clone());
        format!(
            "pipeline library: made {label} in {:.1} ms (the library did not load it: {refused})",
            began.elapsed().as_secs_f64() * 1e3
        )
    });
    let _all = one_call();
    let _stores = library.saving.read().unwrap_or_else(|e| e.into_inner());
    let _call = call("store", name);
    // SAFETY: a live pipeline; a name already stored (a description that
    // differs) is refused, which leaves the library as it was.
    if unsafe { library.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }.is_ok() {
        library.added.fetch_add(1, Ordering::Relaxed);
    }
    Ok((pso, false))
}

/// Writes the library and the early list if pipelines were added since the
/// last save and `SAVE_EVERY` has passed (or `now`).
pub(crate) fn save(now: bool) {
    static LAST: Mutex<Option<std::time::Instant>> = Mutex::new(None);
    const SAVE_EVERY: std::time::Duration = std::time::Duration::from_secs(20);
    let (Some(dir), Some(library)) = (DIR.get(), library()) else {
        return;
    };
    {
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        if !now && last.is_some_and(|t| t.elapsed() < SAVE_EVERY) {
            return;
        }
        *last = Some(std::time::Instant::now());
    }
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(error = %e, "host pipeline library not saved");
        return;
    }
    let tmp = dir.join(&library.files.tmp);
    // The bytes of the library written to `tmp`; 0 when no pipeline was
    // added: the library on disk is then what it was, and only the early
    // list is written (a title pipeline first drawn with wrote all 1,352 MB
    // of the library again: run 60's second save).
    let (bytes, write_early) = {
        let _all = one_call();
        // No store between asking the size and writing; loads go on.
        let _alone = library.saving.write().unwrap_or_else(|e| e.into_inner());
        let added = library.added.load(Ordering::Relaxed);
        let used_changed = USED_CHANGED.swap(false, Ordering::Relaxed);
        if !used_changed && added == 0 {
            return;
        }
        let bytes = if added == 0 {
            0
        } else {
            // SAFETY: a plain query.
            let size = unsafe { library.library.GetSerializedSize() };
            let _call = call("serialise", "the whole library");
            if let Err(e) = serialise(&library.library, &tmp, size) {
                if used_changed {
                    USED_CHANGED.store(true, Ordering::Relaxed);
                }
                tracing::warn!(error = %e, "host pipeline library not serialised");
                let _ = std::fs::remove_file(&tmp);
                return;
            }
            library.added.store(0, Ordering::Relaxed);
            SAVED.store(size as u64, Ordering::Relaxed);
            size
        };
        (bytes, used_changed)
    };
    let library_written = if bytes > 0 {
        replace_library_files(dir, &tmp, &library.files)
    } else {
        Ok(false)
    };
    if bytes > 0
        && let Ok(in_place) = library_written
    {
        if in_place && !library.from_legacy && VIEW[0].load(Ordering::Acquire) != 0 {
            REPLACED[1].store(crate::host::now_ms(), Ordering::Relaxed);
            REPLACED[0].fetch_add(1, Ordering::Relaxed);
        }
        crate::host::trail(|| {
            format!(
                "pipeline library ({}): saved {bytes} bytes, {}",
                library.files.bin,
                if in_place {
                    "renamed over the library's file"
                } else {
                    "left beside the library's file for the next start"
                }
            )
        });
    }
    let library_written = library_written.map(|_| ());
    // Only rewrite `early-pipelines.txt` when this run actually recorded new
    // used title pipelines; offline precompilation leaves `USED` untouched and
    // must preserve any existing early list on disk.
    let (early_count, written) = if write_early {
        let used: Vec<String> = used().iter().flatten().cloned().collect();
        let len = used.len();
        let res = library_written
            .and_then(|()| write_atomically(&dir.join(EARLY_FILE), used.join("\n").as_bytes()));
        (len, res)
    } else {
        let len = USED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map_or(0, HashSet::len);
        (len, library_written)
    };
    match written {
        Ok(()) => tracing::info!(
            bytes,
            file = %library.files.bin,
            early = early_count,
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipeline library saved"
        ),
        Err(e) => tracing::warn!(error = %e, file = %library.files.bin, "host pipeline library not saved"),
    }
}

/// Writes the library's `size` serialised bytes to the file `tmp`: through
/// a mapping of that file, so that the bytes are the file's pages and not
/// 1.3 GB more of the process's own memory for the seconds a save takes;
/// through a buffer where the file cannot be mapped (and with
/// `VICTORY_LIBRARY_MAP=0`).
fn serialise(
    library: &ID3D12PipelineLibrary1,
    tmp: &std::path::Path,
    size: usize,
) -> std::io::Result<()> {
    let failed = |e: windows::core::Error| std::io::Error::other(e.to_string());
    if mapped()
        && let Some(mut file) = Mapped::create(tmp, size)
    {
        // SAFETY: a buffer of the size the library asks for.
        return unsafe { library.Serialize(file.bytes_mut()) }.map_err(failed);
    }
    let mut bytes = vec![0u8; size];
    // SAFETY: a buffer of the size the library asks for.
    unsafe { library.Serialize(&mut bytes) }.map_err(failed)?;
    std::fs::write(tmp, &bytes)
}

/// Renames the written library `tmp` over the library's file in `dir`. This
/// run may have that file mapped: Windows 11 replaces a mapped file, and
/// the mapping goes on reading the old one for as long as a handle on it
/// is open ([`holds`]; with none, only the pages in memory at that moment
/// stay: the rest read as zeros). Where a system refuses, the new library
/// is left beside it as `.next` for the next start.
///
/// Returns `Ok(true)` if renamed in place over `files.bin`, `Ok(false)` if
/// left beside it as `files.next`.
fn replace_library_files(
    dir: &std::path::Path,
    tmp: &std::path::Path,
    files: &LibraryFiles,
) -> std::io::Result<bool> {
    if std::fs::rename(tmp, dir.join(&files.bin)).is_ok() {
        Ok(true)
    } else {
        std::fs::rename(tmp, dir.join(&files.next)).map(|()| false)
    }
}

#[cfg(test)]
fn replace_library(dir: &std::path::Path, tmp: &std::path::Path) -> std::io::Result<()> {
    replace_library_files(dir, tmp, &LibraryFiles::legacy()).map(|_| ())
}

/// Writes `bytes` to a temporary file beside `path`, then renames it over
/// `path`: a reader never sees a part.
fn write_atomically(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

/// Title pipeline keys requested by draws or dispatches (this run and the last).
static USED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static USED_CHANGED: AtomicBool = AtomicBool::new(false);

fn used() -> std::sync::MutexGuard<'static, Option<HashSet<String>>> {
    let mut used = USED.lock().unwrap_or_else(|e| e.into_inner());
    if used.is_none() {
        let listed = DIR
            .get()
            .and_then(|d| std::fs::read_to_string(d.join(EARLY_FILE)).ok())
            .unwrap_or_default();
        let set: HashSet<String> = listed
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_owned)
            .collect();
        tracing::info!(pipelines = set.len(), "title pipelines to start early");
        *used = Some(set);
    }
    used
}

/// Whether the title pipeline with `key` was requested in this or an earlier run.
pub(crate) fn wanted_early(key: &str) -> bool {
    used().as_ref().is_some_and(|s| s.contains(key))
}

/// A draw or dispatch requested the title pipeline with `key`.
pub(crate) fn note_used(key: &str) {
    let mut used = used();
    let set = used.get_or_insert_with(HashSet::new);
    if set.len() < EARLY_MAX && set.insert(key.to_owned()) {
        USED_CHANGED.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_change_with_what_goes_into_the_pipeline() {
        let a = name("graphics", b"root", &[b"vs", b"ps"], "state");
        assert_eq!(a, name("graphics", b"root", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps2"], "state"));
        assert_ne!(a, name("graphics", b"root", &[b"vs", b"ps"], "state2"));
        assert_ne!(a, name("graphics", b"root2", &[b"vs", b"ps"], "state"));
        assert_ne!(a, name("mesh", b"root", &[b"vs", b"ps"], "state"));
        // Parts do not run into each other.
        assert_ne!(a, name("graphics", b"root", &[b"vsp", b"s"], "state"));
    }

    #[test]
    fn a_rejected_blob_gives_an_empty_library() {
        let Some(device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let library = open(&device, Blob::Held(b"not a pipeline library".repeat(16)))
            .expect("an empty library");
        // SAFETY: a plain query on a live library.
        assert!(unsafe { library.library.GetSerializedSize() } > 0);
        assert!(matches!(library._blob, Blob::None));
    }

    /// A mapped file reads as the file's bytes, a file written through a
    /// mapping holds what was written, and the library's file can be
    /// replaced while it is mapped (or the new one waits as the next).
    #[test]
    fn the_library_file_is_mapped_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        assert!(matches!(Blob::of(dir.path()), Blob::None), "no file yet");
        std::fs::write(&path, b"").unwrap();
        assert!(matches!(Blob::of(dir.path()), Blob::None), "an empty file");
        let old: Vec<u8> = (0..200_000u32).map(|i| i as u8).collect();
        std::fs::write(&path, &old).unwrap();
        let blob = Blob::of(dir.path());
        assert!(matches!(blob, Blob::Mapped(_)) || !mapped());
        assert_eq!(blob.bytes(), &old[..]);
        // A save while the old file is mapped.
        let tmp = path.with_extension("tmp");
        let new: Vec<u8> = (0..300_000u32).map(|i| (i * 7) as u8).collect();
        {
            let mut file = Mapped::create(&tmp, new.len()).expect("a new file, mapped");
            file.bytes_mut().copy_from_slice(&new);
        }
        assert_eq!(std::fs::read(&tmp).unwrap(), new);
        replace_library(dir.path(), &tmp).unwrap();
        assert!(!tmp.exists());
        // The mapping still reads the old library.
        assert_eq!(blob.bytes(), &old[..]);
        drop(blob);
        // The next start reads the new one, wherever the save left it.
        assert_eq!(Blob::of(dir.path()).bytes(), &new[..]);
        assert!(!dir.path().join(NEXT_FILE).exists());
    }

    /// The bytes at `at` of the pattern the test below writes.
    fn pattern(at: usize, out: &mut [u8]) {
        for (i, word) in out.chunks_exact_mut(4).enumerate() {
            let v = ((at / 4 + i) as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(1);
            word.copy_from_slice(&v.to_le_bytes());
        }
    }

    /// A file of `len` bytes (whole MiB) of [`pattern`], written past the
    /// system's cache: none of its pages is in memory afterwards. False
    /// where the folder's file system does not take such writes.
    fn write_uncached(path: &std::path::Path, len: usize) -> bool {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        // FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH
        const UNCACHED: u32 = 0x2000_0000 | 0x8000_0000;
        const CHUNK: usize = 1 << 20;
        let Ok(mut file) = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .custom_flags(UNCACHED)
            .open(path)
        else {
            return false;
        };
        // Such writes want a buffer on a sector boundary.
        let mut store = vec![0u8; CHUNK + 4096];
        let start = store.as_ptr().align_offset(4096);
        for at in (0..len).step_by(CHUNK) {
            let chunk = &mut store[start..start + CHUNK];
            pattern(at, chunk);
            if file.write_all(chunk).is_err() {
                return false;
            }
        }
        true
    }

    /// The device loss of runs 72, 73 and 82: the library's file is
    /// replaced while it is mapped, and pages of the mapping that were not
    /// in memory then are read afterwards (in the runs: pages the trim let
    /// go and Windows gave to something else). With the file's handle kept
    /// open they still read as the old file. (With it closed, as before,
    /// all but the first few read as zeros on the PC this was found on.)
    #[test]
    fn a_replaced_library_file_still_reads_from_the_disk() {
        const LEN: usize = 8 << 20;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(LIBRARY_FILE);
        if !write_uncached(&path, LEN) {
            return; // no uncached writes here: nothing to show
        }
        let mapped = Mapped::open_holding(&path, true).expect("the file, mapped");
        // A save: the new library, renamed over the mapped one.
        let tmp = path.with_extension("tmp");
        {
            let mut file = Mapped::create(&tmp, LEN).expect("a new file, mapped");
            file.bytes_mut().fill(0xEE);
        }
        replace_library(dir.path(), &tmp).unwrap();
        // Every page of the old mapping, read for the first time now.
        let mut expected = vec![0u8; 4096];
        for (page, bytes) in mapped.bytes().chunks(4096).enumerate() {
            pattern(page * 4096, &mut expected);
            assert!(
                bytes == &expected[..],
                "page {page} of the replaced library no longer reads as the old file"
            );
        }
    }

    #[test]
    fn adapter_keys_produce_distinct_filenames_for_hardware_and_software() {
        let nv = AdapterKey {
            vendor: 0x10de,
            device: 0x2783,
            revision: 0xa1,
            software: false,
        };
        let warp = AdapterKey {
            vendor: 0x1414,
            device: 0x008c,
            revision: 0x00,
            software: true,
        };
        let nv_files = LibraryFiles::for_adapter(Some(nv));
        let warp_files = LibraryFiles::for_adapter(Some(warp));
        let fallback_files = LibraryFiles::for_adapter(None);

        assert_eq!(nv_files.bin, "host-pipelines-10de-2783-ra1-hw.bin");
        assert_eq!(nv_files.next, "host-pipelines-10de-2783-ra1-hw.next");
        assert_eq!(nv_files.tmp, "host-pipelines-10de-2783-ra1-hw.tmp");
        assert!(nv_files.legacy_fallback);

        assert_eq!(warp_files.bin, "host-pipelines-1414-008c-r00-sw.bin");
        assert_eq!(warp_files.next, "host-pipelines-1414-008c-r00-sw.next");
        assert_eq!(warp_files.tmp, "host-pipelines-1414-008c-r00-sw.tmp");
        assert!(!warp_files.legacy_fallback);

        assert_ne!(nv_files.bin, warp_files.bin);
        assert_ne!(nv_files.next, warp_files.next);
        assert_ne!(nv_files.tmp, warp_files.tmp);

        assert_eq!(fallback_files.bin, LIBRARY_FILE);
        assert_eq!(fallback_files.next, NEXT_FILE);
        assert!(!fallback_files.legacy_fallback);

        // Same vendor/device with software flag toggled never collides.
        let spoofed_sw = AdapterKey {
            software: true,
            ..nv
        };
        assert_ne!(
            LibraryFiles::for_adapter(Some(nv)).bin,
            LibraryFiles::for_adapter(Some(spoofed_sw)).bin
        );
    }

    #[test]
    fn legacy_fallback_reads_hardware_cache_and_never_touches_it_on_warp() {
        let dir = tempfile::tempdir().unwrap();
        let hw = LibraryFiles::for_adapter(Some(AdapterKey {
            vendor: 0x10de,
            device: 0x2783,
            revision: 0xa1,
            software: false,
        }));
        let sw = LibraryFiles::for_adapter(Some(AdapterKey {
            vendor: 0x1414,
            device: 0x008c,
            revision: 0x00,
            software: true,
        }));

        // Existing legacy unsuffixed cache from before adapter keying.
        let legacy_bytes = b"legacy-hardware-pso-blob".repeat(128);
        std::fs::write(dir.path().join(LIBRARY_FILE), &legacy_bytes).unwrap();

        // Software adapter ignores legacy hardware cache completely.
        let (sw_blob, sw_legacy) = Blob::of_files(dir.path(), &sw);
        assert!(matches!(sw_blob, Blob::None));
        assert!(!sw_legacy);

        // Software adapter saving writes only its own `-sw.bin` file.
        let sw_tmp = dir.path().join(&sw.tmp);
        let sw_saved = b"warp-pso-blob".repeat(64);
        std::fs::write(&sw_tmp, &sw_saved).unwrap();
        assert!(replace_library_files(dir.path(), &sw_tmp, &sw).unwrap());
        assert_eq!(std::fs::read(dir.path().join(LIBRARY_FILE)).unwrap(), legacy_bytes);
        assert_eq!(std::fs::read(dir.path().join(&sw.bin)).unwrap(), sw_saved);

        // Hardware adapter seamlessly picks up the legacy file read-only until
        // it writes its own keyed file.
        let (hw_blob, hw_legacy) = Blob::of_files(dir.path(), &hw);
        assert!(hw_legacy);
        assert_eq!(hw_blob.bytes(), &legacy_bytes[..]);
        drop(hw_blob);

        // Saving hardware library creates the keyed file without clobbering
        // either legacy or software files.
        let hw_tmp = dir.path().join(&hw.tmp);
        let hw_saved = b"keyed-hardware-pso-blob".repeat(96);
        std::fs::write(&hw_tmp, &hw_saved).unwrap();
        assert!(replace_library_files(dir.path(), &hw_tmp, &hw).unwrap());

        let (hw_blob2, hw_legacy2) = Blob::of_files(dir.path(), &hw);
        assert!(!hw_legacy2);
        assert_eq!(hw_blob2.bytes(), &hw_saved[..]);
        assert_eq!(std::fs::read(dir.path().join(&sw.bin)).unwrap(), sw_saved);
        assert_eq!(std::fs::read(dir.path().join(LIBRARY_FILE)).unwrap(), legacy_bytes);
    }

    #[test]
    fn warp_save_does_not_discard_hardware_pipeline_library() {
        use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, ID3DBlob};
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter,
            IDXGIAdapter1, IDXGIFactory4,
        };

        const WARP_CHILD_DIR: &str = "VICTORY_TEST_LIB_WARP_DIR";
        const WARP_CHILD_LEGACY_DIR: &str = "VICTORY_TEST_LIB_WARP_LEGACY_DIR";
        const PSO_NAME: &str = "adapter-key-isolation-test";

        let cs = crate::host::compile(
            "[numthreads(1, 1, 1)] void main() {}",
            windows::core::s!("main"),
            windows::core::s!("cs_5_0"),
        )
        .expect("compile test compute shader");

        let make_desc = |device: &ID3D12Device1| -> (ID3D12RootSignature, D3D12_COMPUTE_PIPELINE_STATE_DESC) {
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC::default();
            let mut blob: Option<ID3DBlob> = None;
            // SAFETY: empty root signature serialized and created on `device`.
            let root: ID3D12RootSignature = unsafe {
                D3D12SerializeRootSignature(
                    &rs_desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut blob,
                    None,
                )
                .expect("serialize empty root signature");
                let blob = blob.expect("root signature blob");
                let bytes = std::slice::from_raw_parts(
                    blob.GetBufferPointer().cast::<u8>(),
                    blob.GetBufferSize(),
                );
                device
                    .CreateRootSignature(0, bytes)
                    .expect("create root signature")
            };
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: borrowed pointer; caller keeps `root` alive while `desc` is used.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: cs.as_ptr().cast(),
                    BytecodeLength: cs.len(),
                },
                ..Default::default()
            };
            (root, desc)
        };

        let store_and_save = |dir: &std::path::Path, device: &ID3D12Device1, pso_name: &str| -> LibraryFiles {
            let files = LibraryFiles::for_adapter(adapter_key(device));
            let (blob, from_legacy) = Blob::of_files(dir, &files);
            let lib = open_with(device, blob, files.clone(), from_legacy).expect("open library");
            let (_root, desc) = make_desc(device);
            let wide = HSTRING::from(pso_name);
            // SAFETY: valid compute pipeline description on `device`.
            let pso: ID3D12PipelineState =
                unsafe { device.CreateComputePipelineState(&desc) }.expect("create compute PSO");
            // SAFETY: live PSO stored under unique name.
            unsafe { lib.library.StorePipeline(PCWSTR(wide.as_ptr()), &pso) }
                .expect("store compute PSO");
            let size = unsafe { lib.library.GetSerializedSize() };
            let tmp = dir.join(&files.tmp);
            serialise(&lib.library, &tmp, size).expect("serialise library");
            replace_library_files(dir, &tmp, &files).expect("replace library file");
            files
        };

        // Child process runs ONLY the WARP device so it never shares the
        // process-wide D3D12 WARP singleton with parallel tests in the parent.
        if let Some(warp_dir) = std::env::var_os(WARP_CHILD_DIR) {
            let warp_dir = PathBuf::from(warp_dir);
            let legacy_dir = PathBuf::from(
                std::env::var_os(WARP_CHILD_LEGACY_DIR).expect("legacy dir env var"),
            );
            // SAFETY: plain DXGI and D3D12 device creation for an isolated child WARP device.
            let warp_device: ID3D12Device1 = unsafe {
                let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                else {
                    return;
                };
                let Ok(adapter) = factory.EnumWarpAdapter::<IDXGIAdapter>() else {
                    return;
                };
                let Ok(desc) = adapter.cast::<IDXGIAdapter1>().and_then(|a| a.GetDesc1()) else {
                    return;
                };
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return;
                }
                let mut dev: Option<ID3D12Device> = None;
                if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut dev).is_err() {
                    return;
                }
                let Some(dev) = dev else { return };
                let Ok(dev1) = dev.cast::<ID3D12Device1>() else {
                    return;
                };
                dev1
            };
            let warp_key = adapter_key(&warp_device).expect("WARP adapter key");
            assert!(warp_key.software);

            // Store & save WARP library in the shared cache directory and verify
            // reopening hits the cached WARP PSO.
            let warp_files = store_and_save(&warp_dir, &warp_device, PSO_NAME);
            assert!(warp_dir.join(&warp_files.bin).exists());
            let (blob, from_legacy) = Blob::of_files(&warp_dir, &warp_files);
            assert!(!from_legacy);
            assert!(!blob.bytes().is_empty());
            let lib = open_with(&warp_device, blob, warp_files.clone(), false).expect("reopen WARP");
            let (_root, desc) = make_desc(&warp_device);
            let wide = HSTRING::from(PSO_NAME);
            // SAFETY: valid description matching the stored WARP PSO.
            let loaded: windows::core::Result<ID3D12PipelineState> =
                unsafe { lib.library.LoadComputePipeline(PCWSTR(wide.as_ptr()), &desc) };
            assert!(loaded.is_ok(), "expected cached WARP PSO hit: {:?}", loaded.err());

            // Also save WARP library into `legacy_dir` containing a legacy hardware file.
            let _ = store_and_save(&legacy_dir, &warp_device, PSO_NAME);
            println!("WARP_FILE={}", warp_files.bin);
            return;
        }

        let Some(hw_device) = crate::host::host().and_then(|h| h.device.cast::<ID3D12Device1>().ok())
        else {
            return; // no host GPU here
        };
        let Some(hw_key) = adapter_key(&hw_device) else {
            return;
        };
        if hw_key.software {
            return; // test requires a distinct hardware adapter alongside WARP
        }

        let dir = tempfile::tempdir().unwrap();
        let legacy_dir = tempfile::tempdir().unwrap();

        // 1. Hardware adapter stores and saves its adapter-keyed pipeline library.
        let hw_files = store_and_save(dir.path(), &hw_device, PSO_NAME);
        let hw_path = dir.path().join(&hw_files.bin);
        let hw_bytes_before = std::fs::read(&hw_path).expect("hardware library saved");
        assert!(!hw_bytes_before.is_empty());

        // Seed `legacy_dir` with a legacy unsuffixed hardware library file.
        let legacy_path = legacy_dir.path().join(LIBRARY_FILE);
        std::fs::write(&legacy_path, &hw_bytes_before).unwrap();

        // 2. Run WARP in a separate child process against the EXACT SAME cache directories.
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "pipeline::library::tests::warp_save_does_not_discard_hardware_pipeline_library",
                "--nocapture",
            ])
            .env(WARP_CHILD_DIR, dir.path())
            .env(WARP_CHILD_LEGACY_DIR, legacy_dir.path())
            .output()
            .expect("run WARP library isolation child process");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "WARP child failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        let Some(warp_bin) = stdout.lines().find_map(|l| l.strip_prefix("WARP_FILE=")) else {
            return; // WARP adapter unavailable on this host
        };
        let warp_path = dir.path().join(warp_bin);
        assert_ne!(hw_path, warp_path);
        assert!(warp_path.exists());

        // 3. Hardware library on disk is byte-for-byte untouched after WARP save.
        let hw_bytes_after = std::fs::read(&hw_path).expect("hardware library still present");
        assert_eq!(hw_bytes_before, hw_bytes_after);

        // 4. Re-opening hardware library from `dir` hits the cached hardware PSO.
        let (blob, from_legacy) = Blob::of_files(dir.path(), &hw_files);
        assert!(!from_legacy);
        assert!(!blob.bytes().is_empty());
        let lib = open_with(&hw_device, blob, hw_files.clone(), false).expect("reopen hardware");
        let (_root, desc) = make_desc(&hw_device);
        let wide = HSTRING::from(PSO_NAME);
        // SAFETY: valid description matching the stored hardware PSO.
        let loaded: windows::core::Result<ID3D12PipelineState> =
            unsafe { lib.library.LoadComputePipeline(PCWSTR(wide.as_ptr()), &desc) };
        assert!(
            loaded.is_ok(),
            "expected cached hardware PSO hit for {}: {:?}",
            hw_files.bin,
            loaded.err()
        );

        // 5. Legacy `host-pipelines.bin` compatibility: untouched by WARP, and
        //    loaded read-only seamlessly by hardware.
        assert_eq!(
            std::fs::read(&legacy_path).unwrap(),
            hw_bytes_before,
            "WARP must never overwrite legacy host-pipelines.bin"
        );
        let (hw_legacy_blob, used_legacy) = Blob::of_files(legacy_dir.path(), &hw_files);
        assert!(used_legacy);
        let hw_legacy_lib =
            open_with(&hw_device, hw_legacy_blob, hw_files.clone(), true).expect("open legacy");
        let (_root, desc) = make_desc(&hw_device);
        // SAFETY: valid description matching the stored hardware PSO.
        let loaded: windows::core::Result<ID3D12PipelineState> =
            unsafe { hw_legacy_lib.library.LoadComputePipeline(PCWSTR(wide.as_ptr()), &desc) };
        assert!(
            loaded.is_ok(),
            "hardware should load existing legacy host-pipelines.bin seamlessly"
        );
    }
}
