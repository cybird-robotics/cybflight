use crate::hal::dma::{Transfer, TransferOptions};
use crate::hal::pac::gpio::vals as gpio_vals;
use crate::hal::pac::timer::regs::{CcerGp16, CcmrInput2ch, CcmrOutputGp16};
use crate::hal::pac::timer::vals;
use crate::hal::peripherals::{DMA1_CH0, DMA1_CH1, DMA1_CH2, DMA1_CH3};
use crate::hal::Peri;
use crate::motors::{ACTUATOR_MOTORS, ARM_STATE};
use crate::sensors::DSHOT_TELEMETRY;
use cybflight_drivers::dshot::{
    gcr, telemetry, DSHOT600_GCR_TICKS_PER_BIT, DSHOT_CMD_MOTOR_STOP, DSHOT_DMA_BUFFER_SIZE,
    DSHOT_MAX_THROTTLE, DSHOT_MIN_THROTTLE, MAX_GCR_EDGES, MIN_GCR_EDGES,
};

/// If no motor command arrives for this long while armed, the controller
/// has gone silent (stale odometry, NaN, a starved task). Drop to idle
/// throttle so the airframe does not hold stale thrust while waiting for
/// the failsafe watchdog to disarm.
///
/// This is the *first* stage of a two-stage degradation, and it only
/// exists if the second stage is meaningfully later: `fs_ctrl_timeout_s`
/// must stay well above it or the vehicle disarms before it ever idles.
/// That ordering used to be asserted only in prose; it is now checked at
/// two levels — [`MIN_CTRL_TIMEOUT_RATIO`] against the live parameter at
/// boot, and a compile-time assertion against the schema's own minimum
/// in `control::failsafe`.
pub const MOTOR_CMD_STALE: embassy_time::Duration = embassy_time::Duration::from_millis(10);

/// [`MOTOR_CMD_STALE`] in seconds, for the compile-time comparison
/// against the `fs_ctrl_timeout_s` schema minimum.
pub const MOTOR_CMD_STALE_S: f32 = 0.010;

/// How many times [`MOTOR_CMD_STALE`] the failsafe control timeout must
/// be for the idle stage to be worth having.
///
/// At 3× the airframe spends at least two stale windows at idle before
/// the disarm lands, which is enough for the stage to be observable in a
/// log. Below that the two stages collapse into one and the graceful
/// ramp is theatre.
pub const MIN_CTRL_TIMEOUT_RATIO: f32 = 3.0;

/// Pin-mode stabilization delay before handing the line back to the
/// timer, in CPU cycles.
///
/// `cortex_m::asm::delay` counts cycles, so the ~0.42 µs this used to be
/// was a property of the 480 MHz core rather than a stated requirement.
/// Derived from SYSCLK so it stays the same wall-clock time on a board
/// clocked differently.
const PIN_SETTLE_NS: u32 = 420;
const PIN_SETTLE_CYCLES: u32 = (crate::bsp::SYSCLK_HZ / 1_000_000) * PIN_SETTLE_NS / 1_000;

const DSHOT_THROTTLE_RANGE: u16 = DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE;
/// Minimum DShot throttle sent when motors are armed and controller commands
/// a non-zero output. Prevents motor stall at very low throttle.
/// 5% of throttle range (≈100 DShot steps above DSHOT_MIN_THROTTLE).
const DSHOT_IDLE_THROTTLE: u16 = DSHOT_MIN_THROTTLE + (DSHOT_THROTTLE_RANGE / 20);

use cybflight_msgs::{ActuatorMotors, DshotMotorTelemetry, DshotTelemetry};
use embassy_futures::join::join4;
use embassy_futures::select::{select, Either};
use embassy_time::{Instant, Timer};

use super::{DshotQuadConfig, DSHOT600_ARR, DSHOT600_BIT_0, DSHOT600_BIT_1, DSHOT600_PSC};

/// GCR IC capture: ARR set to max for free-running timestamp counter.
const IC_ARR: u16 = 0xFFFF;

/// CCMR value for IC mode: CC_S=01 (direct mapping), ICF=0x2 (4-sample filter)
/// for both channel halves.
const CCMR_IC: u32 = 0x2121;

/// CCER value for IC mode: CCxE=1, CCxP=1, CCxNP=1 (both edges) for all 4 channels.
const CCER_IC: u32 = 0xBBBB;

#[embassy_executor::task]
pub async fn dshot_task(
    config: DshotQuadConfig,
    mut m1_dma: Peri<'static, DMA1_CH0>,
    mut m2_dma: Peri<'static, DMA1_CH1>,
    mut m3_dma: Peri<'static, DMA1_CH2>,
    mut m4_dma: Peri<'static, DMA1_CH3>,
) {
    let telem_pub = DSHOT_TELEMETRY.immediate_publisher();

    // --- One-time timer configuration ---
    for i in 0..config.timer_count as usize {
        let regs = config.timers[i];
        regs.psc().write_value(DSHOT600_PSC);
        regs.arr().write(|w| w.set_arr(DSHOT600_ARR));
        regs.cr1().modify(|w| {
            w.set_arpe(true);
            w.set_urs(vals::Urs::COUNTER_ONLY);
        });
    }

    // Configure each motor channel for output compare — inverted polarity (bidir DShot)
    for m in 0..4 {
        let regs = config.motors[m].timer_regs;
        let ch = config.motors[m].channel_index as usize;

        regs.ccmr_output(ch / 2).modify(|w| {
            w.set_ocm(ch % 2, vals::Ocm::PWM_MODE1);
            w.set_ocpe(ch % 2, true);
        });

        regs.ccer().modify(|w| {
            w.set_cce(ch, true);
            w.set_ccp(ch, true); // inverted polarity: idle HIGH, pulse LOW
            w.set_ccnp(ch, false);
        });

        regs.ccr(ch).write(|w| w.set_ccr(0));
    }

    for i in 0..config.timer_count as usize {
        config.timers[i].egr().write(|w| w.set_ug(true));
    }

    // Save OC register values (per-timer) for restore after IC mode.
    let mut saved_ccmr = [[CcmrOutputGp16(0); 2]; 2];
    let mut saved_ccer = [CcerGp16(0); 2];
    let mut ccmr_no_preload = [[CcmrOutputGp16(0); 2]; 2];
    for i in 0..config.timer_count as usize {
        saved_ccmr[i][0] = config.timers[i].ccmr_output(0).read();
        saved_ccmr[i][1] = config.timers[i].ccmr_output(1).read();
        saved_ccer[i] = config.timers[i].ccer().read();
        // CCMR values with OCPE disabled — for writing CCR directly to shadow register
        ccmr_no_preload[i][0] = CcmrOutputGp16(saved_ccmr[i][0].0 & !0x0808);
        ccmr_no_preload[i][1] = CcmrOutputGp16(saved_ccmr[i][1].0 & !0x0808);
    }

    let mut telem_motor: usize = 0;
    let mut dshot_throttle: [u16; 4] = [DSHOT_CMD_MOTOR_STOP; 4];
    let mut armed = false;
    let mut last_motor_cmd_time: Option<Instant> = None;

    // --- Bidirectional DShot frame loop ---
    loop {
        // Check for arm state changes (non-blocking)
        if let Some(arm_msg) = ARM_STATE.try_take() {
            if arm_msg.armed != armed {
                armed = arm_msg.armed;
                // Update IS_ARMED atomic for other tasks (INDI, etc.)
                crate::motors::IS_ARMED.store(armed, core::sync::atomic::Ordering::Release);
                if armed {
                    defmt::info!("DShot: ARMED — motors enabled");
                    last_motor_cmd_time = None;
                } else {
                    defmt::info!("DShot: DISARMED — motors stopped");
                    dshot_throttle = [DSHOT_CMD_MOTOR_STOP; 4];
                    last_motor_cmd_time = None;
                }
            }
        }

        if armed {
            if let Either::First(ActuatorMotors { motor_commands, .. }) =
                select(ACTUATOR_MOTORS.wait(), Timer::after_micros(1)).await
            {
                // When armed, 0.0 maps to DSHOT_IDLE_THROTTLE (not MOTOR_STOP).
                // Only the disarm path sends MOTOR_STOP.
                dshot_throttle = motor_commands.map(|nrm| {
                    let raw =
                        (nrm.value() * DSHOT_THROTTLE_RANGE as f32) as u16 + DSHOT_MIN_THROTTLE;
                    raw.max(DSHOT_IDLE_THROTTLE)
                });
                last_motor_cmd_time = Some(Instant::now());
            }

            // Controller gone silent — drop to idle. The failsafe watchdog
            // will disarm after CTRL_TIMEOUT (500 ms). This just ensures we
            // don't hold stale throttle during the gap.
            if let Some(t) = last_motor_cmd_time {
                if Instant::now().duration_since(t) > MOTOR_CMD_STALE {
                    dshot_throttle = [DSHOT_IDLE_THROTTLE; 4];
                }
            }
        } else {
            // Send MOTOR_STOP while disarmed. ESC firmware (BLHeli_32, AM32)
            // requires seeing zero-throttle after power-on before it will
            // accept throttle commands — skipping this blocks ESC arming.
            dshot_throttle = [DSHOT_CMD_MOTOR_STOP; 4];
        }

        // ======================== A: Output DShot frame ========================
        let mut bufs = [[0u32; DSHOT_DMA_BUFFER_SIZE]; 4];
        for (i, (buf, throttle)) in bufs.iter_mut().zip(dshot_throttle).enumerate() {
            let telem_req = i == telem_motor;
            let frame = cybflight_drivers::dshot::frame::encode_packet(throttle, telem_req, true);
            cybflight_drivers::dshot::frame::packet_to_dma_buffer(
                frame,
                buf,
                DSHOT600_BIT_0,
                DSHOT600_BIT_1,
            );
        }

        let dma_opts = TransferOptions::default();

        let xfer_m1 = unsafe {
            Transfer::new_write(
                m1_dma.reborrow(),
                config.motors[0].dma_request,
                &bufs[0],
                config.motors[0]
                    .timer_regs
                    .ccr(config.motors[0].channel_index as usize)
                    .as_ptr() as *mut u32,
                dma_opts,
            )
        };
        let xfer_m2 = unsafe {
            Transfer::new_write(
                m2_dma.reborrow(),
                config.motors[1].dma_request,
                &bufs[1],
                config.motors[1]
                    .timer_regs
                    .ccr(config.motors[1].channel_index as usize)
                    .as_ptr() as *mut u32,
                dma_opts,
            )
        };
        let xfer_m3 = unsafe {
            Transfer::new_write(
                m3_dma.reborrow(),
                config.motors[2].dma_request,
                &bufs[2],
                config.motors[2]
                    .timer_regs
                    .ccr(config.motors[2].channel_index as usize)
                    .as_ptr() as *mut u32,
                dma_opts,
            )
        };
        let xfer_m4 = unsafe {
            Transfer::new_write(
                m4_dma.reborrow(),
                config.motors[3].dma_request,
                &bufs[3],
                config.motors[3]
                    .timer_regs
                    .ccr(config.motors[3].channel_index as usize)
                    .as_ptr() as *mut u32,
                dma_opts,
            )
        };

        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .dier()
                .modify(|w| w.set_ccde(config.motors[m].channel_index as usize, true));
        }

        for i in 0..config.timer_count as usize {
            config.timers[i].cnt().write(|w| w.set_cnt(0));
            config.timers[i].cr1().modify(|w| w.set_cen(true));
        }

        join4(xfer_m1, xfer_m2, xfer_m3, xfer_m4).await;

        // Zero CCRs (output goes to idle HIGH with inverted polarity), disable CC DMA, stop timers
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.ccr(ch).write(|w| w.set_ccr(0));
        }
        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .dier()
                .modify(|w| w.set_ccde(config.motors[m].channel_index as usize, false));
        }
        for i in 0..config.timer_count as usize {
            config.timers[i].cr1().modify(|w| w.set_cen(false));
        }

        // ======================== B: Switch to Input Capture ========================
        // Tight GPIO→OUTPUT → IC config → GPIO→AF sequence matching BF
        // (pwm_output_dshot_hal.c:112-131). Minimise time FC drives the line
        // while ESC may be starting its GCR response.

        // Step 1: GPIO → OUTPUT (drive HIGH, LOW speed) while reconfiguring timer
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.bsrr().write(|w| w.set_bs(pin, true));
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::OUTPUT));
        }

        // Step 2: Reconfigure timer channels for IC mode
        for i in 0..config.timer_count as usize {
            config.timers[i].ccer().write_value(CcerGp16(0));
            config.timers[i]
                .ccmr_input(0)
                .write_value(CcmrInput2ch(CCMR_IC));
            config.timers[i]
                .ccmr_input(1)
                .write_value(CcmrInput2ch(CCMR_IC));
            config.timers[i].ccer().write_value(CcerGp16(CCER_IC));
        }

        // Step 3: GPIO → AF immediately — release line to ESC BEFORE DMA setup.
        // This matches BF's tight ISR sequence where GPIO→AF happens right after
        // IC_Init, before DMA is configured. Any IC captures before DMA is ready
        // are lost, but this prevents bus contention with the ESC.
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::ALTERNATE));
        }

        // Step 4: Now configure timer free-running + DMA (line is already released)
        for i in 0..config.timer_count as usize {
            config.timers[i].arr().write(|w| w.set_arr(IC_ARR));
            config.timers[i].egr().write(|w| w.set_ug(true));
            config.timers[i].cnt().write(|w| w.set_cnt(0));
            config.timers[i].cr1().modify(|w| w.set_cen(true));
        }

        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .dier()
                .modify(|w| w.set_ccde(config.motors[m].channel_index as usize, true));
        }

        let mut edge_buf0 = [0u32; MAX_GCR_EDGES];
        let mut edge_buf1 = [0u32; MAX_GCR_EDGES];
        let mut edge_buf2 = [0u32; MAX_GCR_EDGES];
        let mut edge_buf3 = [0u32; MAX_GCR_EDGES];

        let ic_xfer_m1 = unsafe {
            Transfer::new_read(
                m1_dma.reborrow(),
                config.motors[0].dma_request,
                config.motors[0]
                    .timer_regs
                    .ccr(config.motors[0].channel_index as usize)
                    .as_ptr() as *mut u32,
                &mut edge_buf0,
                dma_opts,
            )
        };
        let ic_xfer_m2 = unsafe {
            Transfer::new_read(
                m2_dma.reborrow(),
                config.motors[1].dma_request,
                config.motors[1]
                    .timer_regs
                    .ccr(config.motors[1].channel_index as usize)
                    .as_ptr() as *mut u32,
                &mut edge_buf1,
                dma_opts,
            )
        };
        let ic_xfer_m3 = unsafe {
            Transfer::new_read(
                m3_dma.reborrow(),
                config.motors[2].dma_request,
                config.motors[2]
                    .timer_regs
                    .ccr(config.motors[2].channel_index as usize)
                    .as_ptr() as *mut u32,
                &mut edge_buf2,
                dma_opts,
            )
        };
        let ic_xfer_m4 = unsafe {
            Transfer::new_read(
                m4_dma.reborrow(),
                config.motors[3].dma_request,
                config.motors[3]
                    .timer_regs
                    .ccr(config.motors[3].channel_index as usize)
                    .as_ptr() as *mut u32,
                &mut edge_buf3,
                dma_opts,
            )
        };

        // ======================== C: Wait for ESC response ========================
        embassy_time::Timer::after_micros(80).await;

        // Read how many edges each DMA captured
        let edge_counts = [
            MAX_GCR_EDGES - ic_xfer_m1.get_remaining_transfers() as usize,
            MAX_GCR_EDGES - ic_xfer_m2.get_remaining_transfers() as usize,
            MAX_GCR_EDGES - ic_xfer_m3.get_remaining_transfers() as usize,
            MAX_GCR_EDGES - ic_xfer_m4.get_remaining_transfers() as usize,
        ];

        // Drop IC transfers (stops DMA streams)
        drop(ic_xfer_m1);
        drop(ic_xfer_m2);
        drop(ic_xfer_m3);
        drop(ic_xfer_m4);

        // ======================== D: Restore Output Compare mode ========================
        // Stop timer, disable CC DMA
        for i in 0..config.timer_count as usize {
            config.timers[i].cr1().modify(|w| w.set_cen(false));
        }
        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .dier()
                .modify(|w| w.set_ccde(config.motors[m].channel_index as usize, false));
        }

        // H7 GPIO trick: drive HIGH (inverted idle) during CCMR reconfiguration.
        // Speed stays LOW (set in IC phase) — matches BF which never restores high speed.
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.bsrr().write(|w| w.set_bs(pin, true));
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::OUTPUT));
        }

        // Restore OC CCMR (channels must be off for CC1S write)
        for i in 0..config.timer_count as usize {
            config.timers[i].ccer().write_value(CcerGp16(0));
            config.timers[i]
                .ccmr_output(0)
                .write_value(saved_ccmr[i][0]);
            config.timers[i]
                .ccmr_output(1)
                .write_value(saved_ccmr[i][1]);
        }

        // IC captures corrupt CCR shadow registers. Fix: temporarily disable
        // output preload (OCPE=0) so CCR writes go directly to shadow, then
        // re-enable preload and restore CCER.
        for i in 0..config.timer_count as usize {
            config.timers[i]
                .ccmr_output(0)
                .write_value(ccmr_no_preload[i][0]);
            config.timers[i]
                .ccmr_output(1)
                .write_value(ccmr_no_preload[i][1]);
        }
        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .ccr(config.motors[m].channel_index as usize)
                .write(|w| w.set_ccr(0));
        }
        for i in 0..config.timer_count as usize {
            config.timers[i]
                .ccmr_output(0)
                .write_value(saved_ccmr[i][0]);
            config.timers[i]
                .ccmr_output(1)
                .write_value(saved_ccmr[i][1]);
            config.timers[i].ccer().write_value(saved_ccer[i]);
        }

        // Restore DShot600 ARR
        for i in 0..config.timer_count as usize {
            config.timers[i].arr().write(|w| w.set_arr(DSHOT600_ARR));
            config.timers[i].egr().write(|w| w.set_ug(true));
        }

        // Brief stabilization before switching the pin back to timer AF.
        // `asm::delay` counts CPU cycles, so the wall-clock duration
        // depends on SYSCLK; expressed here as a time and converted, it
        // no longer silently shortens if the core clock changes.
        cortex_m::asm::delay(PIN_SETTLE_CYCLES);

        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::ALTERNATE));
        }

        // ======================== E: Decode telemetry ========================
        let invalid_telem = DshotMotorTelemetry {
            value: telemetry::TelemetryValue::Invalid,
            raw: None,
        };
        let mut telem = DshotTelemetry {
            timestamp: Instant::now(),
            motors: [
                invalid_telem.clone(),
                invalid_telem.clone(),
                invalid_telem.clone(),
                invalid_telem,
            ],
        };

        let edge_bufs = [
            &edge_buf0[..],
            &edge_buf1[..],
            &edge_buf2[..],
            &edge_buf3[..],
        ];
        for m in 0..4 {
            if edge_counts[m] >= MIN_GCR_EDGES {
                if let Some(raw) = gcr::decode_telemetry_packet(
                    &edge_bufs[m][..edge_counts[m]],
                    edge_counts[m],
                    DSHOT600_GCR_TICKS_PER_BIT,
                ) {
                    telem.motors[m].raw = Some(raw);
                    telem.motors[m].value = telemetry::interpret(raw, false);
                }
            }
        }

        telem_pub.publish_immediate(telem);

        telem_motor = (telem_motor + 1) % 4;

        // ======================== F: Wait for next frame ========================
        embassy_time::Timer::after_micros(30).await;
    }
}
