// ---------------------------------------------------------------------------
// Platform init helpers
// ---------------------------------------------------------------------------

/// Enable Cortex-M7 instruction cache for flash acceleration.
pub fn enable_icache() {
    let mut cp = cortex_m::Peripherals::take().unwrap();
    cp.SCB.enable_icache();
}

// ---------------------------------------------------------------------------
// Platform reset helpers
// ---------------------------------------------------------------------------

/// Trigger an immediate software system reset via the Cortex-M SCB AIRCR.
pub fn sys_reboot() -> ! {
    cortex_m::peripheral::SCB::sys_reset()
}

#[cortex_m_rt::pre_init]
unsafe fn pre_init() {
    // ── Reset-cause capture ────────────────────────────────────────────
    //
    // Read RCC.RSR + PWR.CSR1 into a `.uninit` static **before** any
    // other code runs. The flags are sticky, so doing this in `main`
    // is fine — but `bsp::init()` may touch RCC for clock setup, and
    // a future change that writes RMVF anywhere upstream would
    // silently lose the prior boot's cause. Capturing here guarantees
    // we own the cause regardless.
    //
    // The capture itself is ~10 cycles and uses no peripherals beyond
    // raw RCC/PWR reads (always-on domain). Unconditional — NOT gated
    // on `postmortem`: a board that dies in the field on a build
    // without the postmortem subsystem must still be able to answer
    // "was that an IWDG reset or a power dip?" via the `resetcause`
    // shell verb on the next boot.
    crate::reset_cause::capture();

    const DFU_MAGIC: u32 = 0xDEAD_D00D;
    // RTC backup register 0 on STM32H743
    let bkpr0 = 0x58004050 as *mut u32;
    unsafe {
        if bkpr0.read_volatile() == DFU_MAGIC {
            bkpr0.write_volatile(0);
            // Jump to system memory DFU bootloader
            let sp = (0x1FF0_9800usize as *const u32).read();
            let entry = (0x1FF0_9800usize as *const u32).add(1).read();
            cortex_m::asm::bootstrap(sp as *const u32, entry as *const u32);
        }
    }
}

/// Jump directly to the STM32H743 system-memory USB DFU bootloader.
///
/// Disables all maskable interrupts and SysTick, clears the NVIC enable and
/// pending state, loads the bootloader's initial stack pointer and entry point
/// from its vector table at `0x1FF09800`, then branches there.  The USB
/// peripheral re-enumerates on the host as a DFU target without a full reset
/// cycle.
///
/// This function never returns.
pub fn enter_dfu() -> ! {
    const DFU_MAGIC: u32 = 0xDEAD_D00D;
    unsafe { (0x58004050 as *mut u32).write_volatile(DFU_MAGIC) };
    sys_reboot()
}
