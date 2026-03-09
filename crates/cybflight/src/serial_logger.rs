//! UART-based defmt logger.
//!
//! When the `defmt_uart` feature is enabled (default), frames are encoded and
//! written to a blocking `UartTx` on USART2. When disabled, a no-op logger is
//! compiled instead — combine with `DEFMT_LOG=off` in `.cargo/config.toml` to
//! eliminate all logging overhead in production builds.

use crate::hal::mode::Blocking;
use crate::hal::usart::UartTx;

// ---- Feature-gated UART logger -------------------------------------------

#[cfg(feature = "defmt_uart")]
mod inner {
    use core::cell::UnsafeCell;
    use core::sync::atomic::{AtomicBool, Ordering};

    use embedded_io::Write as _;

    use crate::hal::mode::Blocking;
    use crate::hal::usart::UartTx;

    pub(super) struct SerialLogger {
        taken: AtomicBool,
        cs_restore: UnsafeCell<critical_section::RestoreState>,
        encoder: UnsafeCell<defmt::Encoder>,
        uart: UnsafeCell<Option<UartTx<'static, Blocking>>>,
    }

    unsafe impl Sync for SerialLogger {}

    pub(super) static LOGGER: SerialLogger = SerialLogger {
        taken: AtomicBool::new(false),
        cs_restore: UnsafeCell::new(critical_section::RestoreState::invalid()),
        encoder: UnsafeCell::new(defmt::Encoder::new()),
        uart: UnsafeCell::new(None),
    };

    pub(super) fn store_uart(uart: UartTx<'static, Blocking>) {
        critical_section::with(|_| {
            // Safety: we are inside a critical section.
            unsafe {
                (*LOGGER.uart.get()) = Some(uart);
            }
        });
    }

    /// Best-effort blocking write to the UART (no-op if not yet initialised).
    unsafe fn do_write(bytes: &[u8]) {
        unsafe {
            if let Some(uart) = (*LOGGER.uart.get()).as_mut() {
                let _ = uart.write_all(bytes);
            }
        }
    }

    #[defmt::global_logger]
    struct Logger;

    unsafe impl defmt::Logger for Logger {
        fn acquire() {
            // Safety: must be paired with exactly one release().
            let restore = unsafe { critical_section::acquire() };

            if LOGGER.taken.load(Ordering::Relaxed) {
                panic!("defmt logger taken reentrantly");
            }
            LOGGER.taken.store(true, Ordering::Relaxed);

            // Safety: inside critical section, exclusive access guaranteed.
            unsafe {
                LOGGER.cs_restore.get().write(restore);
                (*LOGGER.encoder.get()).start_frame(|bytes| {
                    do_write(bytes);
                });
            }
        }

        unsafe fn write(bytes: &[u8]) {
            unsafe {
                (*LOGGER.encoder.get()).write(bytes, |encoded| {
                    do_write(encoded);
                });
            }
        }

        unsafe fn flush() {
            unsafe {
                if let Some(uart) = (*LOGGER.uart.get()).as_mut() {
                    let _ = uart.flush();
                }
            }
        }

        unsafe fn release() {
            unsafe {
                (*LOGGER.encoder.get()).end_frame(|bytes| {
                    do_write(bytes);
                });

                LOGGER.taken.store(false, Ordering::Relaxed);
                let restore = LOGGER.cs_restore.get().read();
                // Safety: paired with the acquire() call above.
                critical_section::release(restore);
            }
        }
    }
}

// ---- No-op logger (production) -------------------------------------------

#[cfg(not(feature = "defmt_uart"))]
mod inner {
    #[defmt::global_logger]
    struct Logger;

    unsafe impl defmt::Logger for Logger {
        fn acquire() {}
        unsafe fn flush() {}
        unsafe fn release() {}
        unsafe fn write(_bytes: &[u8]) {}
    }
}

// ---- Public API (cfg-free) -----------------------------------------------

/// Initialise the defmt serial transport.
///
/// When `defmt_uart` is enabled the UART is stored for logging. When disabled
/// the peripheral is simply dropped (no bytes are ever transmitted).
pub fn init(uart: UartTx<'static, Blocking>) {
    #[cfg(feature = "defmt_uart")]
    inner::store_uart(uart);

    #[cfg(not(feature = "defmt_uart"))]
    drop(uart);
}
