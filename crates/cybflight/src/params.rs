//! Persistent vehicle parameter storage in STM32H743 internal flash.
//!
//! Parameters are stored in flash sector 7 (last 128KB of bank 1, offset 0x0E_0000).
//! At boot, `init_from_flash` reads the sector and populates a global static.
//! Shell commands can modify in-memory params and save back to flash.
//!
//! # Flash erase and the watchdog
//!
//! On STM32H7, erasing a same-bank flash sector stalls the CPU bus for the
//! duration of the erase (~1-2 s for 128KB). The IWDG counter (clocked by LSI
//! at 32 kHz) keeps counting independently — so the normal 500 ms timeout
//! would fire before the erase completes.
//!
//! [`save_to_flash`] extends the IWDG timeout to ~4 s before starting the
//! operation and restores the normal 500 ms timeout afterwards. The CPU stall
//! during erase is unavoidable (same-bank limitation), but the watchdog stays
//! alive.

use core::cell::RefCell;

use critical_section::Mutex;
use cybflight_core::params::{VehicleParams, PADDED_SIZE};

use crate::hal;

/// Absolute address of the parameter sector in flash (sector 7, 0x0E0000).
const PARAM_FLASH_ADDR: u32 = 0x0800_0000 + 0x0E_0000;

static VEHICLE_PARAMS: Mutex<RefCell<Option<VehicleParams>>> = Mutex::new(RefCell::new(None));

/// Flash peripheral stored for later save operations (blocking mode).
static FLASH_PERI: Mutex<RefCell<Option<hal::flash::Flash<'static, hal::flash::Blocking>>>> =
    Mutex::new(RefCell::new(None));

/// Monotonic version counter — incremented every time params change
/// (`set()`, `save_to_flash()`, `init_from_flash()`). Consumers cache
/// a local copy and re-read params when the version changes.
///
/// Check this only when idle (disarmed) — never in the 8kHz control loop.
pub static PARAM_VERSION: core::sync::atomic::AtomicU32 =
    core::sync::atomic::AtomicU32::new(0);

/// Initialize parameters from flash at boot. Falls back to compile-time defaults
/// if flash is blank or corrupt.
///
/// Must be called once, early in `board_init`, before any task reads params.
pub fn init_from_flash(flash_peri: hal::Peri<'static, hal::peripherals::FLASH>) {
    // Read the parameter sector via raw pointer (flash is memory-mapped).
    let buf: &[u8; PADDED_SIZE] = unsafe { &*(PARAM_FLASH_ADDR as *const [u8; PADDED_SIZE]) };

    let params = match VehicleParams::from_bytes(buf) {
        Some(p) => {
            defmt::info!("params: loaded from flash");
            p
        }
        None => {
            defmt::info!("params: flash blank/corrupt, using defaults");
            crate::vehicle::default_params()
        }
    };

    let flash = hal::flash::Flash::new_blocking(flash_peri);
    let mission_profile_idx = params.mission_profile;
    let blackbox_record_set_byte = params.blackbox_record_set;
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
    }
}

/// Get a copy of the current vehicle parameters.
pub fn get() -> VehicleParams {
    critical_section::with(|cs| {
        VEHICLE_PARAMS
            .borrow_ref(cs)
            .as_ref()
            .expect("params::get called before init")
            .clone()
    })
}

/// Update the in-memory vehicle parameters.
///
/// Increments `PARAM_VERSION` so consumers know to re-read.
pub fn set(params: VehicleParams) {
    critical_section::with(|cs| {
        VEHICLE_PARAMS.borrow_ref_mut(cs).replace(params);
    });
    PARAM_VERSION.fetch_add(1, core::sync::atomic::Ordering::Release);
}

/// Erase sector 7 and write the current params to flash.
///
/// Extends the IWDG timeout to ~4 s before the operation (the erase stalls
/// the CPU for ~1-2 s due to same-bank flash access) and restores the normal
/// 500 ms timeout afterwards.
///
/// Returns `Ok(())` on success, or an error string on failure.
pub fn save_to_flash() -> Result<(), &'static str> {
    let params = get();
    let data = params.to_bytes();

    // Take flash peripheral out of the static (short critical section).
    let mut flash = critical_section::with(|cs| FLASH_PERI.borrow_ref_mut(cs).take())
        .ok_or("flash not initialized")?;

    let sector_offset = PARAM_FLASH_ADDR - 0x0800_0000;
    let sector_end = sector_offset + 128 * 1024; // 128KB sector

    // Extend watchdog to ~4 s so the CPU stall during erase doesn't trigger a reset.
    crate::watchdog::extend_timeout();

    let result = flash
        .blocking_erase(sector_offset, sector_end)
        .map_err(|_| "flash erase failed")
        .and_then(|()| {
            flash
                .blocking_write(sector_offset, &data)
                .map_err(|_| "flash write failed")
        });

    // Restore normal 500 ms watchdog timeout.
    crate::watchdog::restore_timeout();

    // Return flash peripheral to the static (short critical section).
    critical_section::with(|cs| {
        FLASH_PERI.borrow_ref_mut(cs).replace(flash);
    });

    if result.is_ok() {
        PARAM_VERSION.fetch_add(1, core::sync::atomic::Ordering::Release);
    }

    result
}
