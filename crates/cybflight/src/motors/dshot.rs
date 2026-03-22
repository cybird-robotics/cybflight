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

const DSHOT_THROTTLE_RANGE: u16 = DSHOT_MAX_THROTTLE - DSHOT_MIN_THROTTLE;
/// Minimum DShot throttle sent when motors are armed and controller commands
/// a non-zero output. Prevents motor stall at very low throttle.
/// 0.5% of throttle range ≈ 10 DShot steps above DSHOT_MIN_THROTTLE.
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
    let mut dshot_throttle: [u16; 4] = [0; 4];
    let mut armed = false;
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
                } else {
                    defmt::info!("DShot: DISARMED — motors stopped");
                    dshot_throttle = [DSHOT_CMD_MOTOR_STOP; 4];
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
            }
        } else {
            dshot_throttle = [DSHOT_CMD_MOTOR_STOP; 4];
        }

        defmt::debug!(
            "DShot throttles: M1={} M2={} M3={} M4={}",
            dshot_throttle[0],
            dshot_throttle[1],
            dshot_throttle[2],
            dshot_throttle[3]
        );

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
        // H7 GPIO trick: OUTPUT during CCMR reconfiguration
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.bsrr().write(|w| w.set_bs(pin, true)); // drive HIGH (inverted idle)
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::OUTPUT));
        }

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

        // Reconfigure timer for free-running IC timestamps
        for i in 0..config.timer_count as usize {
            config.timers[i].arr().write(|w| w.set_arr(IC_ARR));
            config.timers[i].egr().write(|w| w.set_ug(true));
            config.timers[i].cnt().write(|w| w.set_cnt(0));
            config.timers[i].cr1().modify(|w| w.set_cen(true));
        }

        // Enable CC DMA for IC capture
        for m in 0..4 {
            config.motors[m]
                .timer_regs
                .dier()
                .modify(|w| w.set_ccde(config.motors[m].channel_index as usize, true));
        }

        // Start DMA reads: timer CCR → edge timestamp buffers
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

        // Switch GPIO to AF — ESC can now drive the line with GCR telemetry
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::ALTERNATE));
        }

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

        // H7 GPIO trick: drive HIGH (inverted idle) during CCMR reconfiguration
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

        // Brief stabilization before switching pin back to timer AF
        cortex_m::asm::delay(200);

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
                // Skip leading glitch edge(s) from GPIO→AF transition.
                // The glitch-to-GCR gap is much larger than any gap within GCR data
                // (GCR guarantees transitions every ~3-4 bits max = ~64 ticks).
                let mut gcr_start = 0;
                for i in 1..edge_counts[m] {
                    let gap = edge_bufs[m][i].wrapping_sub(edge_bufs[m][i - 1]);
                    if gap > DSHOT600_GCR_TICKS_PER_BIT * 5 {
                        gcr_start = i;
                    }
                }
                let gcr_edges = &edge_bufs[m][gcr_start..];
                let gcr_count = edge_counts[m] - gcr_start;

                if gcr_count >= MIN_GCR_EDGES {
                    let ticks = gcr::detect_ticks_per_bit(
                        gcr_edges,
                        gcr_count,
                        DSHOT600_GCR_TICKS_PER_BIT,
                    )
                    .unwrap_or(DSHOT600_GCR_TICKS_PER_BIT);
                    if let Some(raw) = gcr::decode_telemetry_packet(
                        gcr_edges,
                        gcr_count,
                        ticks,
                    ) {
                        telem.motors[m].raw = Some(raw);
                        telem.motors[m].value = telemetry::interpret(raw, false);
                    }
                }
            }
        }

        telem_pub.publish_immediate(telem);

        telem_motor = (telem_motor + 1) % 4;

        // ======================== F: Wait for next frame ========================
        embassy_time::Timer::after_micros(30).await;
    }
}
