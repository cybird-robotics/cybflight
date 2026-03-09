use core::sync::atomic::Ordering;

use crate::hal::dma::{Transfer, TransferOptions};
use crate::hal::pac::gpio::vals as gpio_vals;
use crate::hal::pac::timer::{regs, vals};
use crate::hal::peripherals::{DMA1_CH0, DMA1_CH1, DMA1_CH2, DMA1_CH3};
use crate::hal::Peri;
use crate::msgs::{DshotMotorTelemetry, DshotTelemetry};
use crate::sensors::DSHOT_TELEMETRY;
use cybflight_drivers::dshot::{
    gcr, telemetry, DSHOT600_GCR_TICKS_PER_BIT, DSHOT_DMA_BUFFER_SIZE, MAX_GCR_EDGES,
    MIN_GCR_EDGES,
};
use embassy_futures::join::join4;
use embassy_time::Instant;

use super::{DshotQuadConfig, DSHOT600_ARR, DSHOT600_BIT_0, DSHOT600_BIT_1, DSHOT600_PSC};

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
        defmt::info!(
            "DShot timer {}: PSC={} ARR={}",
            i,
            regs.psc().read(),
            regs.arr().read().arr()
        );
    }

    // Configure each motor channel for output compare
    for m in 0..4 {
        let regs = config.motors[m].timer_regs;
        let ch = config.motors[m].channel_index as usize;

        regs.ccmr_output(ch / 2).modify(|w| {
            w.set_ocm(ch % 2, vals::Ocm::PWM_MODE1);
            w.set_ocpe(ch % 2, true);
        });

        regs.ccer().modify(|w| {
            w.set_cce(ch, true);
            w.set_ccp(ch, true); // inverted polarity: idle HIGH, pulse LOW (bidir DShot)
            w.set_ccnp(ch, false);
        });

        regs.ccr(ch).write(|w| w.set_ccr(0));

        defmt::info!(
            "DShot M{}: ch={} dma_req={} af={}",
            m,
            config.motors[m].channel_index,
            config.motors[m].dma_request,
            config.motors[m].af_number
        );
    }

    // Generate update event to load preloaded values
    for i in 0..config.timer_count as usize {
        config.timers[i].egr().write(|w| w.set_ug(true));
    }

    // Save raw CCMR/CCER register addresses and values for OC restore.
    // We use raw pointer writes to completely bypass PAC typed accessors.
    let tim3_base = config.timers[0].ccmr_output(0).as_ptr() as usize;
    let ccmr1_ptr = tim3_base as *mut u32;
    let ccmr2_ptr = (tim3_base + 4) as *mut u32; // CCMR2 is CCMR1 + 4
    let ccer_ptr = config.timers[0].ccer().as_ptr() as *mut u32;

    let saved_ccmr1: u32;
    let saved_ccmr2: u32;
    let saved_ccer: u32;
    unsafe {
        saved_ccmr1 = core::ptr::read_volatile(ccmr1_ptr);
        saved_ccmr2 = core::ptr::read_volatile(ccmr2_ptr);
        saved_ccer = core::ptr::read_volatile(ccer_ptr);
    }
    defmt::info!(
        "DShot saved: CCMR1={:#010x} CCMR2={:#010x} CCER={:#010x}",
        saved_ccmr1,
        saved_ccmr2,
        saved_ccer
    );

    // Telemetry request rotation: one motor per frame
    let mut telem_motor: usize = 0;
    let mut frame_count: u32 = 0;

    // --- Bidirectional frame output loop ---
    loop {
        // ======================== A: Output DShot frame ========================
        let mut bufs = [[0u32; DSHOT_DMA_BUFFER_SIZE]; 4];
        for (i, buf) in bufs.iter_mut().enumerate() {
            let throttle = super::MOTOR_THROTTLE[i].load(Ordering::Relaxed);
            let telem_req = i == telem_motor;
            let frame =
                cybflight_drivers::dshot::frame::encode_packet(throttle, telem_req, true);
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

        // Enable CC DMA for each motor channel
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.dier().modify(|w| w.set_ccde(ch, true));
        }

        // Reset + start timers
        for i in 0..config.timer_count as usize {
            config.timers[i].cnt().write(|w| w.set_cnt(0));
            config.timers[i].cr1().modify(|w| w.set_cen(true));
        }

        // Await all 4 DMA write transfers
        join4(xfer_m1, xfer_m2, xfer_m3, xfer_m4).await;

        // Zero CCRs, disable CC DMA, stop timers
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.ccr(ch).write(|w| w.set_ccr(0));
        }
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.dier().modify(|w| w.set_ccde(ch, false));
        }
        for i in 0..config.timer_count as usize {
            config.timers[i].cr1().modify(|w| w.set_cen(false));
        }

        // ======================== B: Switch to IC mode ========================
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.bsrr().write(|w| w.set_bs(pin, true)); // drive HIGH (inverted idle)
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::OUTPUT));
        }
        unsafe {
            core::ptr::write_volatile(ccer_ptr, 0);
            core::ptr::write_volatile(ccmr1_ptr, 0x2121);
            core::ptr::write_volatile(ccmr2_ptr, 0x2121);
            core::ptr::write_volatile(ccer_ptr, 0xBBBB);
        }
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::ALTERNATE));
        }

        embassy_time::Timer::after_micros(80).await;

        // ======================== D: Restore OC mode ========================
        // Drive HIGH immediately to take control back from ESC
        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.bsrr().write(|w| w.set_bs(pin, true));
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::OUTPUT));
        }

        // Restore timer OC config
        unsafe {
            core::ptr::write_volatile(ccer_ptr, 0);
            core::ptr::write_volatile(ccmr1_ptr, saved_ccmr1);
            core::ptr::write_volatile(ccmr2_ptr, saved_ccmr2);
        }

        // Zero CCR shadow: disable preload, write 0 directly, re-enable preload
        let ccmr_no_preload1 = saved_ccmr1 & !0x0808;
        let ccmr_no_preload2 = saved_ccmr2 & !0x0808;
        unsafe {
            core::ptr::write_volatile(ccmr1_ptr, ccmr_no_preload1);
            core::ptr::write_volatile(ccmr2_ptr, ccmr_no_preload2);
        }
        for ch in 0..4usize {
            config.timers[0].ccr(ch).write(|w| w.set_ccr(0));
        }
        unsafe {
            core::ptr::write_volatile(ccmr1_ptr, saved_ccmr1);
            core::ptr::write_volatile(ccmr2_ptr, saved_ccmr2);
            core::ptr::write_volatile(ccer_ptr, saved_ccer);
        }

        // Stay in GPIO OUTPUT HIGH for a bit to stabilize before switching to AF
        cortex_m::asm::delay(200); // ~1µs at 200MHz

        for m in 0..4 {
            let gpio = config.motors[m].gpio_port;
            let pin = config.motors[m].gpio_pin as usize;
            gpio.moder()
                .modify(|w| w.set_moder(pin, gpio_vals::Moder::ALTERNATE));
        }

        // ======================== E: Decode telemetry (stub) ========================
        let invalid_telem = DshotMotorTelemetry {
            value: telemetry::TelemetryValue::Invalid,
            raw: None,
        };
        let telem = DshotTelemetry {
            timestamp: Instant::now(),
            motors: [
                invalid_telem.clone(),
                invalid_telem.clone(),
                invalid_telem.clone(),
                invalid_telem,
            ],
        };

        // telem_pub.publish_immediate(telem); // disabled to reduce log noise

        // Rotate telemetry request to next motor
        telem_motor = (telem_motor + 1) % 4;
        frame_count = frame_count.wrapping_add(1);

        // ======================== F: Wait for next frame ========================
        embassy_time::Timer::after_micros(30).await;
    }
}
