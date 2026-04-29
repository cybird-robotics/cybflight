//! Battery voltage / current monitoring via ADC3.
//!
//! Embassy-stm32 v0.5.0 does not map PC2/PC3 to ADC3 channels (dual-pad pins
//! on STM32H743).  We use `Adc::new_with_config()` for RCC/calibration init
//! and PAC registers for the actual channel reads — same hybrid pattern as
//! DShot's PAC-level timer access.

use crate::hal;
use crate::msgs;
use crate::sensors::POWER_STATUS;
use bsp_types::PowerCalibration;
use embassy_time::{Duration, Instant, Ticker};
use hal::adc::{Adc, AdcConfig, Resolution};
use hal::pac::adc::vals::Pcsel;

/// ADC3 channel for VBAT (PC3 = ADC3_INP1 per STM32H743 datasheet / BF `adc_stm32h7xx.c`).
const VBAT_CHANNEL: u8 = 1;
/// ADC3 channel for CURR (PC2 = ADC3_INP0).
const CURR_CHANNEL: u8 = 0;

/// Sample time matching Betaflight (~387.5 ADC clock cycles).
const ADC_SAMPLE_TIME: hal::pac::adc::vals::SampleTime =
    hal::pac::adc::vals::SampleTime::CYCLES387_5;

pub struct PowerMonitor {
    /// Kept alive to prevent ADC3 RCC clock from being disabled.
    _adc: Adc<'static, hal::peripherals::ADC3>,
    cal: PowerCalibration,
    mah_drawn_f: f32,
    cell_count: u8,
    cells_detected: bool,
    settle_count: u8,
}

impl PowerMonitor {
    pub fn new(
        adc3: hal::Peri<'static, hal::peripherals::ADC3>,
        vbat_pin: hal::Peri<'static, hal::peripherals::PC3>,
        curr_pin: hal::Peri<'static, hal::peripherals::PC2>,
        cal: PowerCalibration,
    ) -> Self {
        // Ensure GPIO pins are in analog mode (MODER=0b11, which is also the
        // reset default).  We consume the Peri tokens to prevent reuse.
        hal::pac::GPIOC.moder().modify(|w| {
            w.set_moder(2, hal::pac::gpio::vals::Moder::ANALOG); // PC2
            w.set_moder(3, hal::pac::gpio::vals::Moder::ANALOG); // PC3
        });
        core::mem::forget(vbat_pin);
        core::mem::forget(curr_pin);

        // Embassy init: RCC enable, prescaler, voltage regulator, calibration.
        let adc = Adc::new_with_config(
            adc3,
            AdcConfig {
                resolution: Some(Resolution::BITS12),
                ..Default::default()
            },
        );

        // Pre-select both channels so PCSEL bits are set once.
        let regs = hal::pac::ADC3;
        regs.pcsel().modify(|w| {
            w.set_pcsel(VBAT_CHANNEL as _, Pcsel::PRESELECTED);
            w.set_pcsel(CURR_CHANNEL as _, Pcsel::PRESELECTED);
        });

        Self {
            _adc: adc,
            cal,
            mah_drawn_f: 0.0,
            cell_count: 0,
            cells_detected: false,
            settle_count: 0,
        }
    }

    /// Read a single ADC3 channel via PAC registers (12-bit, blocking poll).
    /// Returns 0 on timeout (should never happen with correct clock config).
    fn read_channel(channel: u8) -> u16 {
        let regs = hal::pac::ADC3;

        // Sample time (channels 0-9 → SMPR1 register).
        regs.smpr(0)
            .modify(|w| w.set_smp(channel as _, ADC_SAMPLE_TIME));

        // Sequence: 1 conversion (L=0 means length 1), SQ1 = channel.
        regs.sqr1().modify(|w| {
            w.set_l(0);
            w.set_sq(0, channel);
        });

        // Clear EOS/EOC, start conversion, poll for end-of-sequence.
        regs.isr().modify(|w| {
            w.set_eos(true);
            w.set_eoc(true);
        });
        regs.cr().modify(|w| w.set_adstart(true));

        // Safety timeout: ~200 µs at 480 MHz. Conversion at 32 MHz ADC clock
        // with 387.5-cycle sample ≈ 12 µs, so this is 16× margin.
        let mut timeout = 100_000u32;
        while !regs.isr().read().eos() {
            timeout -= 1;
            if timeout == 0 {
                defmt::warn!("ADC3 read timeout ch{}", channel);
                return 0;
            }
        }

        regs.dr().read().rdata()
    }

    /// BF `voltage.c:165` — convert 12-bit ADC reading to centivolts.
    fn raw_to_voltage_cv(&self, raw: u16) -> u16 {
        let src = raw as u32;
        let scale = self.cal.voltage_scale as u32;
        let div = self.cal.voltage_divider as u32;
        let mul = self.cal.voltage_multiplier as u32;
        // Rounding term: 0xFFF * div / 2 (half-LSB rounding).
        ((src * scale * 3300 / 10 + 0xFFF * div / 2) / (0xFFF * div) / mul) as u16
    }

    /// BF `current.c:112-119` — convert 12-bit ADC reading to centiamps.
    fn raw_to_current_ca(&self, raw: u16) -> i32 {
        let millivolts = raw as i32 * 3300 / 4096;
        millivolts * 10000 / self.cal.current_scale as i32 + self.cal.current_offset as i32
    }

    /// BF `battery.c:214` — auto-detect cell count from voltage.
    fn detect_cells(&mut self, voltage_cv: u16) {
        if self.cells_detected {
            return;
        }

        // Discard first ~10 readings (~100 ms at 100 Hz) for ADC settling.
        if self.settle_count < 10 {
            self.settle_count += 1;
            return;
        }

        if voltage_cv < 300 {
            self.cell_count = 0; // < 3.0 V → no battery
        } else {
            let cells = voltage_cv / 430 + 1;
            self.cell_count = if cells > 8 { 8 } else { cells as u8 };
        }
        self.cells_detected = true;
        defmt::info!(
            "Battery: {}S ({}.{:02}V)",
            self.cell_count,
            voltage_cv / 100,
            voltage_cv % 100
        );
    }
}

#[embassy_executor::task]
pub async fn power_task(mut mon: PowerMonitor) -> ! {
    let mut ticker = Ticker::every(Duration::from_hz(100));
    let pub_ = POWER_STATUS.immediate_publisher();
    let mut last_time = Instant::now();

    loop {
        ticker.next().await;

        let now = Instant::now();
        let dt_us = now.duration_since(last_time).as_micros() as f32;
        last_time = now;

        let vbat_raw = PowerMonitor::read_channel(VBAT_CHANNEL);
        let curr_raw = PowerMonitor::read_channel(CURR_CHANNEL);

        let voltage_cv = mon.raw_to_voltage_cv(vbat_raw);
        let current_ca = mon.raw_to_current_ca(curr_raw);

        mon.detect_cells(voltage_cv);

        // Integrate mAh drawn (only positive current contributes).
        if current_ca > 0 {
            // centiamps → mAh:  cA * µs / (100 * 1e6 µs/s * 3600 s/h) = cA * µs / 3.6e11
            mon.mah_drawn_f += current_ca as f32 * dt_us / (100.0 * 1_000_000.0 * 3600.0);
        }

        pub_.publish_immediate(msgs::PowerStatus {
            timestamp: now,
            voltage_cv,
            current_ca,
            mah_drawn: mon.mah_drawn_f as u32,
            cell_count: mon.cell_count,
        });
    }
}
