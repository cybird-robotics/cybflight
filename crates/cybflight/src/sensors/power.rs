//! Battery voltage / current monitoring via ADC3.
//!
//! Embassy-stm32 v0.5.0 does not map the GPIOC dual-pad pins to ADC3 channels
//! on STM32H743.  We use `Adc::new_with_config()` for RCC/calibration init and
//! PAC registers for the actual channel reads — same hybrid pattern as DShot's
//! PAC-level timer access.
//!
//! The VBAT/CURR pins and their ADC3 channels are **board-specific** (e.g.
//! SAKURAH743: PC3/PC2 = INP1/INP0; MICOAIR743V2: PC0/PC1 = INP10/INP11), so
//! they are passed in via [`AdcInput`] rather than hardcoded here.

use crate::hal;
use crate::msgs;
use crate::sensors::{POWER_STATUS, POWER_TELEM};
use bsp_types::PowerCalibration;
use embassy_time::{Duration, Instant, Ticker};
use hal::adc::{Adc, AdcConfig, Averaging, Resolution};
use hal::pac::adc::vals::Pcsel;

/// One `power_task` tick as the blackbox sees it (`/power` topic).
///
/// Distinct from [`msgs::PowerStatus`] (an external wire type we can't
/// extend) in carrying both the raw and the `batt_lpf_hz`-filtered pack
/// voltage. `voltage_raw_cv` is the hardware-oversampled ADC reading
/// converted straight to centivolts, no software filter — it is what
/// the pack actually did this tick, ripple included; `voltage_cv` is
/// what every other consumer (INDI thrust table, shell, GCS) sees.
#[derive(Clone, Copy, Debug, defmt::Format)]
pub struct PowerTelemetry {
    pub timestamp: Instant,
    /// Filtered pack voltage, centivolts (same value as `POWER_STATUS`).
    pub voltage_cv: u16,
    /// Unfiltered pack voltage, centivolts.
    pub voltage_raw_cv: u16,
    /// Unfiltered current, centiamps.
    pub current_ca: i32,
    /// Milliamp-hours consumed since power-on.
    pub mah_drawn: u32,
    /// Detected cell count (0 = no battery).
    pub cell_count: u8,
}

/// Sample time matching Betaflight (~387.5 ADC clock cycles).
const ADC_SAMPLE_TIME: hal::pac::adc::vals::SampleTime =
    hal::pac::adc::vals::SampleTime::CYCLES387_5;

/// Hardware oversampling depth. The H7's ADC accumulates this many
/// conversions and right-shifts back to 12 bits itself, so the noise floor
/// drops ~√N for zero CPU cost — the sequencer just takes ~16× longer
/// (~190 µs, against a 10 ms tick). Nothing downstream changes: the result
/// register still reads 0..4095.
const ADC_OVERSAMPLING: Averaging = Averaging::Samples16;

/// Single-pole (PT1) low-pass, the same shape Betaflight runs on its vbat
/// ADC reading. Applied to the raw *count* rather than the converted
/// centivolts: the conversion is linear, so filtering first costs nothing,
/// and it keeps the averaged value at sub-count precision — worth having,
/// since one ADC count is 27 mV of pack on SAKURAH743.
struct Pt1 {
    /// `None` until the first sample. Seeding with the first reading rather
    /// than 0 matters: cell detection runs 100 ms after boot, long before a
    /// filter starting at zero would have converged, and would latch a wrong
    /// cell count off the ramp.
    state: Option<f32>,
    /// RC time constant, from the configured cutoff.
    rc: f32,
}

impl Pt1 {
    fn new(cutoff_hz: f32) -> Self {
        Self {
            state: None,
            rc: 1.0 / (2.0 * core::f32::consts::PI * cutoff_hz),
        }
    }

    fn apply(&mut self, sample: f32, dt_s: f32) -> f32 {
        let out = match self.state {
            None => sample,
            // dt is measured, not assumed, so a late tick weights correctly
            // instead of silently shifting the cutoff.
            Some(prev) => prev + (dt_s / (self.rc + dt_s)) * (sample - prev),
        };
        self.state = Some(out);
        out
    }
}

/// One ADC analog input: its GPIOC pin number (for analog-mode config) and its
/// ADC3 channel number. Both are board-specific; see the BSP pinout.
#[derive(Copy, Clone)]
pub struct AdcInput {
    /// GPIOC pin number (0-15) to switch into analog mode.
    pub gpioc_pin: u8,
    /// ADC3 channel (INPx) this pin maps to.
    pub channel: u8,
}

pub struct PowerMonitor {
    /// Kept alive to prevent ADC3 RCC clock from being disabled.
    _adc: Adc<'static, hal::peripherals::ADC3>,
    cal: PowerCalibration,
    vbat_channel: u8,
    curr_channel: u8,
    mah_drawn_f: f32,
    cell_count: u8,
    cells_detected: bool,
    settle_count: u16,
}

impl PowerMonitor {
    /// Construct the monitor. `vbat`/`curr` carry each input's GPIOC pin and
    /// ADC3 channel; the pin `Peri` tokens are consumed to prevent reuse. All
    /// inputs are on GPIOC and ADC3 across our boards.
    pub fn new<VBAT: hal::PeripheralType, CURR: hal::PeripheralType>(
        adc3: hal::Peri<'static, hal::peripherals::ADC3>,
        vbat_pin: hal::Peri<'static, VBAT>,
        curr_pin: hal::Peri<'static, CURR>,
        vbat: AdcInput,
        curr: AdcInput,
        cal: PowerCalibration,
    ) -> Self {
        // Ensure GPIO pins are in analog mode (MODER=0b11, which is also the
        // reset default).  We consume the Peri tokens to prevent reuse.
        hal::pac::GPIOC.moder().modify(|w| {
            w.set_moder(vbat.gpioc_pin as _, hal::pac::gpio::vals::Moder::ANALOG);
            w.set_moder(curr.gpioc_pin as _, hal::pac::gpio::vals::Moder::ANALOG);
        });
        core::mem::forget(vbat_pin);
        core::mem::forget(curr_pin);

        // Embassy init: RCC enable, prescaler, voltage regulator, calibration.
        let adc = Adc::new_with_config(
            adc3,
            AdcConfig {
                resolution: Some(Resolution::BITS12),
                averaging: Some(ADC_OVERSAMPLING),
            },
        );

        // Pre-select both channels so PCSEL bits are set once.
        let regs = hal::pac::ADC3;
        regs.pcsel().modify(|w| {
            w.set_pcsel(vbat.channel as _, Pcsel::PRESELECTED);
            w.set_pcsel(curr.channel as _, Pcsel::PRESELECTED);
        });

        Self {
            _adc: adc,
            cal,
            vbat_channel: vbat.channel,
            curr_channel: curr.channel,
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

    /// BF `voltage.c:165` — convert a 12-bit ADC reading to centivolts.
    ///
    /// Same ratio as Betaflight's integer form (`src·scale·33·10 / (0xFFF·
    /// divider·multiplier)`), evaluated in f32 because the input is a
    /// filtered count with a fractional part. Rounding to the nearest
    /// centivolt is what caps the output resolution at 10 mV — finer than
    /// the 27 mV one raw count is worth on SAKURAH743, so the filter's
    /// sub-count precision survives into the reported value instead of
    /// being quantized away.
    fn raw_to_voltage_cv(&self, raw: f32) -> u16 {
        let scale = self.cal.voltage_scale as f32;
        let div = self.cal.voltage_divider as f32;
        let mul = self.cal.voltage_multiplier as f32;
        let cv = libm::roundf(raw * scale * 330.0 / (4095.0 * div * mul));
        // A wild ADC read must not wrap into a plausible-looking voltage.
        if cv <= 0.0 {
            0
        } else if cv >= u16::MAX as f32 {
            u16::MAX
        } else {
            cv as u16
        }
    }

    /// BF `current.c:112-119` — convert 12-bit ADC reading to centiamps.
    fn raw_to_current_ca(&self, raw: u16) -> i32 {
        let millivolts = raw as i32 * 3300 / 4096;
        millivolts * 10000 / self.cal.current_scale as i32 + self.cal.current_offset as i32
    }

    /// BF `battery.c:214` — auto-detect cell count from voltage.
    ///
    /// Thresholds are chemistry-coupled (`batt_cell_detect_v`,
    /// `batt_no_battery_v`), so they come in from the caller's param
    /// snapshot rather than being welded in here.
    fn detect_cells(
        &mut self,
        voltage_cv: u16,
        no_battery_cv: u16,
        cell_detect_cv: u16,
        settle_ticks: u16,
        max_cells: u8,
    ) {
        if self.cells_detected {
            return;
        }

        // Discard the first `batt_settle_ticks` readings (~100 ms at the
        // task's 100 Hz tick) so the divider and its low-pass settle
        // before the count is taken from them.
        if self.settle_count < settle_ticks {
            self.settle_count += 1;
            return;
        }

        if voltage_cv < no_battery_cv {
            self.cell_count = 0; // below batt_no_battery_v → no pack
        } else {
            let cells = voltage_cv / cell_detect_cv + 1;
            let cap = u16::from(max_cells);
            self.cell_count = if cells > cap { max_cells } else { cells as u8 };
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
    // Chemistry-coupled cell-detection thresholds, in centivolts.
    // Reboot-flagged group, so read once.
    let batt = crate::params::get().battery;
    let no_battery_cv = (batt.no_battery_v * 100.0) as u16;
    let cell_detect_cv = (batt.cell_detect_v * 100.0) as u16;
    // Cell-detection shape from the same reboot-flagged group. A zero
    // ceiling would report every pack as 0S, so it degrades to 1.
    let settle_ticks = batt.settle_ticks;
    let max_cells = batt.max_cells.max(1);

    // Hardware oversampling (see ADC_OVERSAMPLING) takes out the per-
    // conversion noise; this takes out what's left of the rail ripple the
    // motors put on the pack, which oversampling over ~190 µs can't see.
    let mut vbat_lpf = Pt1::new(batt.lpf_hz);

    let mut ticker = Ticker::every(Duration::from_hz(100));
    let pub_ = POWER_STATUS.immediate_publisher();
    let telem_pub = POWER_TELEM.immediate_publisher();
    // Blackbox-only stream, thinned here rather than in the recorder:
    // a divider behind the PubSub still spends a channel slot and a
    // drain-budget iteration on every sample it discards. See
    // `rates::BLACKBOX_POWER_DECIM`. `POWER_STATUS` above is
    // untouched — its consumers still get the full 100 Hz tick.
    let mut telem_ctr: u32 = 0;
    let mut last_time = Instant::now();

    loop {
        ticker.next().await;

        let now = Instant::now();
        let dt_us = now.duration_since(last_time).as_micros() as f32;
        last_time = now;

        let vbat_raw = PowerMonitor::read_channel(mon.vbat_channel);
        let curr_raw = PowerMonitor::read_channel(mon.curr_channel);

        let vbat_filtered = vbat_lpf.apply(vbat_raw as f32, dt_us / 1_000_000.0);
        let voltage_cv = mon.raw_to_voltage_cv(vbat_filtered);
        // Unfiltered twin for the blackbox only — lets a log quantify
        // the filter's lag against real sag instead of guessing.
        let voltage_raw_cv = mon.raw_to_voltage_cv(vbat_raw as f32);
        // Current is left unfiltered: `mah_drawn` integrates it, and
        // integration is already a low-pass — filtering first would only add
        // lag to the running total.
        let current_ca = mon.raw_to_current_ca(curr_raw);

        mon.detect_cells(
            voltage_cv,
            no_battery_cv,
            cell_detect_cv,
            settle_ticks,
            max_cells,
        );

        // Integrate mAh drawn (only positive current contributes).
        if current_ca > 0 {
            // centiamps → mAh:  cA * µs / (100 * 1e6 µs/s * 3600 s/h) = cA * µs / 3.6e11
            mon.mah_drawn_f += current_ca as f32 * dt_us / (100.0 * 1_000_000.0 * 3600.0);
        }

        let mah_drawn = mon.mah_drawn_f as u32;
        pub_.publish_immediate(msgs::PowerStatus {
            timestamp: now,
            voltage_cv,
            current_ca,
            mah_drawn,
            cell_count: mon.cell_count,
        });
        telem_ctr += 1;
        if telem_ctr >= crate::rates::BLACKBOX_POWER_DECIM {
            telem_ctr = 0;
            telem_pub.publish_immediate(PowerTelemetry {
                timestamp: now,
                voltage_cv,
                voltage_raw_cv,
                current_ca,
                mah_drawn,
                cell_count: mon.cell_count,
            });
        }
    }
}
