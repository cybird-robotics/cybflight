//! Persistent firmware configuration in STM32H743 internal flash.
//!
//! Storage is the key-value override log in `cybflight_core::param_store`,
//! spanning bank-2 flash sectors 6 and 7 (the last 256 KB of the 2 MB
//! part; `memory.x` shrinks the app FLASH region to 1792K to reserve
//! them). Code executes from bank 1, so KV programs/erases never stall
//! instruction fetches (H7 read-while-write across banks). At boot,
//! [`init_from_flash`] starts from the compile-time defaults and replays
//! the override records; `param set` mutates the in-memory copy and
//! `param save` appends only the changed records — one 32-byte flash word
//! per parameter, no erase. Appending cannot *remove* an override (the
//! log has no tombstone record), so `param save --prune`
//! ([`prune_to_flash`]) rewrites the store to exactly the set differing
//! from the baked defaults — the only way a re-flashed YAML edit stops
//! being shadowed by a stale record.
//!
//! # Flash erase and the watchdog
//!
//! A cross-bank erase does not stall the CPU, but `blocking_erase` spins
//! in the calling task for ~1–2 s — starving the thread executor (ESKF,
//! MPC, USB, and the IWDG feeder) while the IWDG keeps counting, so the
//! normal 500 ms timeout would fire mid-erase. Unlike the old whole-blob
//! scheme, the KV log only erases on *first-time formatting*, on
//! *compaction* (sector full or torn tail), and on an explicit
//! `param save --prune`; the flash adapter extends the
//! IWDG to ~4 s around every erase and restores it afterwards.

use core::cell::RefCell;

use critical_section::Mutex;
use cybflight_core::param_store::{self, KvFlash, StoreError, WORD};
use cybflight_core::params::FirmwareConfig;

use crate::hal;

/// Byte offsets (from flash base) of the two KV sectors: bank-2 sectors
/// 6 and 7 (0x081C_0000 / 0x081E_0000) — the region `memory.x` excludes
/// from the app FLASH length. Must be in the opposite bank from code so
/// programs/erases never stall instruction fetches.
const KV_SECTOR_OFFSET: [u32; 2] = [0x1C_0000, 0x1E_0000];
const KV_SECTOR_SIZE: usize = 128 * 1024;
const FLASH_BASE: u32 = 0x0800_0000;

static VEHICLE_PARAMS: Mutex<RefCell<Option<FirmwareConfig>>> = Mutex::new(RefCell::new(None));

/// Flash peripheral stored for later save operations (blocking mode).
static FLASH_PERI: Mutex<RefCell<Option<hal::flash::Flash<'static, hal::flash::Blocking>>>> =
    Mutex::new(RefCell::new(None));

/// Monotonic version counter — incremented every time params change
/// (`set()`, `save_to_flash()`, `init_from_flash()`). Consumers cache
/// a local copy and re-read params when the version changes.
///
/// Check this only when idle (disarmed) — never in the 8kHz control loop.
pub static PARAM_VERSION: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// [`KvFlash`] over the embassy blocking flash driver. Reads are
/// memory-mapped (same as the pre-KV code); writes/erases go through the
/// driver. Erases bracket the IWDG extension.
struct H7KvFlash<'a> {
    flash: &'a mut hal::flash::Flash<'static, hal::flash::Blocking>,
}

impl KvFlash for H7KvFlash<'_> {
    fn sector_size(&self) -> usize {
        KV_SECTOR_SIZE
    }

    fn read(&mut self, sector: u8, offset: usize, buf: &mut [u8]) {
        let addr = FLASH_BASE + KV_SECTOR_OFFSET[sector as usize] + offset as u32;
        let src = unsafe { core::slice::from_raw_parts(addr as *const u8, buf.len()) };
        buf.copy_from_slice(src);
    }

    fn write_word(&mut self, sector: u8, offset: usize, word: &[u8; WORD]) -> Result<(), StoreError> {
        let off = KV_SECTOR_OFFSET[sector as usize] + offset as u32;
        self.flash
            .blocking_write(off, word)
            .map_err(|_| StoreError::Flash)
    }

    fn erase_sector(&mut self, sector: u8) -> Result<(), StoreError> {
        // Re-check the armed state at the last possible moment: the shell's
        // armed guard is sampled at dispatch, but arming can race the
        // save. The cross-bank erase leaves DShot (interrupt executor)
        // running, but blocks the thread executor — ESKF, MPC, the IWDG
        // feeder — for ~1–2 s, guaranteeing a controller-silence failsafe
        // mid-flight. Never acceptable while armed.
        if crate::motors::IS_ARMED.load(core::sync::atomic::Ordering::Acquire) {
            return Err(StoreError::Armed);
        }
        let start = KV_SECTOR_OFFSET[sector as usize];
        // The ~1–2 s CPU stall would outlive the normal 500 ms IWDG window.
        crate::watchdog::extend_timeout();
        let result = self
            .flash
            .blocking_erase(start, start + KV_SECTOR_SIZE as u32)
            .map_err(|_| StoreError::Flash);
        crate::watchdog::restore_timeout();
        result
    }
}

/// BKPSRAM breadcrumb around the override replay: if the previous boot
/// died *inside* the replay (the H7 ECC hazard — a torn flash word can
/// bus-fault a memory-mapped read), the next boot detects the still-set
/// marker, erases the param sectors, and continues on baked defaults —
/// a loud warning instead of a permanent boot loop. Requires the
/// `postmortem` feature for the BKPSRAM clock/section infrastructure;
/// without it the breadcrumb is compiled out (no recovery, pre-existing
/// behavior).
#[cfg(feature = "postmortem")]
mod load_guard {
    use core::mem::MaybeUninit;

    const MAGIC: u32 = u32::from_le_bytes(*b"PRML");

    #[unsafe(link_section = ".bkpsram")]
    #[unsafe(no_mangle)]
    static mut PARAM_LOAD_GUARD: MaybeUninit<u32> = MaybeUninit::uninit();

    /// Mark "entering param load". Returns true if the previous boot
    /// never reached [`exit`] — i.e. it died inside the replay.
    pub fn enter() -> bool {
        // Idempotent; makes the BKPSRAM clock/domain writable this early
        // (main's postmortem init runs later and becomes a no-op).
        crate::postmortem::bkpsram::enable();
        // SAFETY: single-threaded boot path, before any task spawns; the
        // postmortem record is a different static in the same section.
        unsafe {
            let p = (&raw mut PARAM_LOAD_GUARD).cast::<u32>();
            let prev = core::ptr::read_volatile(p);
            core::ptr::write_volatile(p, MAGIC);
            cortex_m::asm::dsb();
            prev == MAGIC
        }
    }

    /// Clear the marker — param load completed without faulting.
    pub fn exit() {
        unsafe {
            let p = (&raw mut PARAM_LOAD_GUARD).cast::<u32>();
            core::ptr::write_volatile(p, 0);
            cortex_m::asm::dsb();
        }
    }
}

/// Initialize parameters from flash at boot: baked (YAML) defaults +
/// KV override replay.
///
/// Must be called once, early in `board_init`, before any task reads params.
pub fn init_from_flash(flash_peri: hal::Peri<'static, hal::peripherals::FLASH>) {
    defmt::info!("params: baked vehicle {}", crate::vehicle::BAKED_VEHICLE);
    let mut params = crate::vehicle::default_params();
    let mut flash = hal::flash::Flash::new_blocking(flash_peri);

    #[cfg(feature = "postmortem")]
    let prev_boot_died_in_load = load_guard::enter();
    #[cfg(not(feature = "postmortem"))]
    let prev_boot_died_in_load = false;

    if prev_boot_died_in_load {
        // The replay (or the fault it triggers) never completed last
        // boot — most plausibly an uncorrectable-ECC word from a
        // power-cut flash program. Reading it again would fault again,
        // so erase the store and fly on baked defaults.
        defmt::error!(
            "params: previous boot died during override load — erasing param store, using baked defaults (re-apply tuning via param set)"
        );
        let mut kv = H7KvFlash { flash: &mut flash };
        let _ = kv.erase_sector(0);
        let _ = kv.erase_sector(1);
    } else {
        let report = {
            let mut kv = H7KvFlash { flash: &mut flash };
            param_store::load(&mut kv, &mut params)
        };
        if report.rejected > 0 {
            defmt::warn!(
                "params: {} flash record(s) failed validation and were ignored",
                report.rejected,
            );
        }
        if report.applied > 0 || report.skipped > 0 {
            defmt::info!(
                "params: {} overrides applied, {} unknown skipped",
                report.applied,
                report.skipped,
            );
        } else {
            defmt::info!("params: no overrides in flash, using baked defaults");
        }
    }
    #[cfg(feature = "postmortem")]
    load_guard::exit();

    let mission_profile_idx = params.trajectory.mission_profile;
    let blackbox_record_set_byte = params.system.blackbox_record_set;
    let blackbox_rate_div = params.system.blackbox_rate_div as u32;
    critical_section::with(|cs| {
        VEHICLE_PARAMS.borrow_ref_mut(cs).replace(params);
        FLASH_PERI.borrow_ref_mut(cs).replace(flash);
    });
    PARAM_VERSION.fetch_add(1, core::sync::atomic::Ordering::Release);

    // Mirror the persisted offline-mission profile index into the
    // `offline_mission::ACTIVE_PROFILE` atomic so the planner sees the
    // user's last selection from the very first PLAN_REQUEST. Out-of-range
    // values (e.g., flash from a firmware build that knew of more profiles)
    // fall back to the default inside `init_active_from_index`.
    #[cfg(feature = "outer_mpc")]
    crate::control::offline_mission::init_active_from_index(mission_profile_idx);
    #[cfg(not(feature = "outer_mpc"))]
    let _ = mission_profile_idx;

    // Mirror the persisted blackbox record-set tier into the
    // BLACKBOX_RECORD_SET atomic. Boards without storage compile out
    // the recorder entirely, so this branch dead-code-eliminates.
    if crate::bsp::HAS_BLACKBOX_STORAGE {
        let rs = crate::blackbox::record_set::RecordSet::from_u8(blackbox_record_set_byte);
        crate::blackbox::record_set::set(rs);
        // Same mirroring for `blackbox_rate_div`: the IMU reader divides
        // `IMU_1_RAW` at the publisher, so it needs the value outside the
        // recorder's session snapshot. See `record_set::rate_div`.
        crate::blackbox::record_set::set_rate_div(blackbox_rate_div);
    }
}

/// Get a copy of the current firmware configuration.
pub fn get() -> FirmwareConfig {
    critical_section::with(|cs| {
        VEHICLE_PARAMS
            .borrow_ref(cs)
            .as_ref()
            .expect("params::get called before init")
            .clone()
    })
}

/// Update the in-memory configuration.
///
/// Increments `PARAM_VERSION` so consumers know to re-read.
pub fn set(params: FirmwareConfig) {
    // Mirror `blackbox_rate_div` out to its atomic on every runtime write,
    // not just at boot. The IMU reader thins `IMU_1_RAW` at the publisher
    // (see `blackbox::record_set::rate_div`), so it reads the atomic and
    // never sees `VEHICLE_PARAMS` — without this, `param set
    // blackbox_rate_div N` would apply only after a reboot while `param
    // get` cheerfully reported the new value. `set` is the single write
    // path for the live store, so this covers `set`, `reset` and
    // `defaults` alike.
    //
    // Safe to do here rather than at session start: `param set` is refused
    // while armed, and `blackbox_*` keys are refused while a bench session
    // is recording, so the divider cannot move under a session that has
    // already stamped `rate_div` into its file metadata.
    if crate::bsp::HAS_BLACKBOX_STORAGE {
        crate::blackbox::record_set::set_rate_div(params.system.blackbox_rate_div as u32);
    }
    critical_section::with(|cs| {
        VEHICLE_PARAMS.borrow_ref_mut(cs).replace(params);
    });
    PARAM_VERSION.fetch_add(1, core::sync::atomic::Ordering::Release);
}

/// Persist the current configuration: append override records for every
/// parameter whose value changed since the last save. Erases only on
/// first-time formatting or when the log compacts (both bracket the IWDG
/// extension inside the flash adapter).
///
/// Returns `Ok(())` on success, or an error string on failure.
pub fn save_to_flash() -> Result<(), &'static str> {
    save_with(false).map(|_| ())
}

/// Persist the current configuration as the *whole* override set: rewrite
/// the store to hold exactly the params differing from the baked (vehicle
/// YAML) defaults, dropping every record the append log accumulated on the
/// way there.
///
/// Needed because the KV log has no tombstone record, so a plain
/// [`save_to_flash`] after `param reset` records the baked value instead
/// of removing the key — which pins it against the *next* YAML edit. See
/// [`param_store::save_pruned`]. Two workflows want it:
///
/// - `param reset <name>` + prune — a YAML edit stops being shadowed.
/// - `param reset all` + prune — a clean store after flashing a different
///   vehicle onto a board (records carry no vehicle identity).
///
/// Always erases, so it is refused while armed by the adapter's
/// last-moment armed re-check. Returns the number of surviving override
/// records.
pub fn prune_to_flash() -> Result<u32, &'static str> {
    save_with(true)
}

/// Shared body of [`save_to_flash`] / [`prune_to_flash`]. Returns the
/// number of records written.
fn save_with(prune: bool) -> Result<u32, &'static str> {
    let params = get();
    let baked = crate::vehicle::default_params();

    // Take flash peripheral out of the static (short critical section).
    let mut flash = critical_section::with(|cs| FLASH_PERI.borrow_ref_mut(cs).take())
        .ok_or("flash not initialized")?;

    let result = {
        let mut kv = H7KvFlash { flash: &mut flash };
        if prune {
            param_store::save_pruned(&mut kv, &params, &baked)
        } else {
            param_store::save(&mut kv, &params, &baked)
        }
    };

    // Return flash peripheral to the static (short critical section).
    critical_section::with(|cs| {
        FLASH_PERI.borrow_ref_mut(cs).replace(flash);
    });

    match result {
        Ok(r) => {
            if prune {
                defmt::info!("params: pruned store to {} override record(s)", r.appended);
            } else {
                defmt::info!(
                    "params: saved {} record(s){}",
                    r.appended,
                    if r.compacted { " (compacted)" } else { "" },
                );
            }
            PARAM_VERSION.fetch_add(1, core::sync::atomic::Ordering::Release);
            Ok(r.appended)
        }
        Err(StoreError::Flash) => Err("flash write/erase failed"),
        Err(StoreError::Full) => Err("param store full"),
        Err(StoreError::Corrupt) => Err("param registry inconsistency"),
        Err(StoreError::Armed) => Err("refused: armed (erase would stall the CPU)"),
    }
}
