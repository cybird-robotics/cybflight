use core::sync::atomic::Ordering;

use crate::hal::dma::{Transfer, TransferOptions};
use crate::hal::pac::timer::vals;
use crate::hal::peripherals::{DMA1_CH0, DMA1_CH1, DMA1_CH2, DMA1_CH3};
use crate::hal::Peri;
use cybflight_drivers::dshot::DSHOT_DMA_BUFFER_SIZE;
use embassy_futures::join::join4;

use super::{DshotQuadConfig, DSHOT600_ARR, DSHOT600_BIT_0, DSHOT600_BIT_1, DSHOT600_PSC};

#[embassy_executor::task]
pub async fn dshot_task(
    config: DshotQuadConfig,
    mut m1_dma: Peri<'static, DMA1_CH0>,
    mut m2_dma: Peri<'static, DMA1_CH1>,
    mut m3_dma: Peri<'static, DMA1_CH2>,
    mut m4_dma: Peri<'static, DMA1_CH3>,
) {
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

    // Configure each motor channel
    for m in 0..4 {
        let regs = config.motors[m].timer_regs;
        let ch = config.motors[m].channel_index as usize;

        // CCMR output: PWM mode 1, preload enabled
        // ccmr_output(0) covers Ch1+Ch2, ccmr_output(1) covers Ch3+Ch4
        regs.ccmr_output(ch / 2).modify(|w| {
            w.set_ocm(ch % 2, vals::Ocm::PWM_MODE1);
            w.set_ocpe(ch % 2, true);
        });

        // CCER: enable channel output, active high polarity
        regs.ccer().modify(|w| {
            w.set_cce(ch, true);
            w.set_ccp(ch, false);
            w.set_ccnp(ch, false);
        });

        // CCR: start at 0 (output low)
        regs.ccr(ch).write(|w| w.set_ccr(0));

        defmt::info!(
            "DShot M{}: ch={} dma_req={}",
            m,
            config.motors[m].channel_index,
            config.motors[m].dma_request
        );
    }

    // Generate update event to load preloaded values
    for i in 0..config.timer_count as usize {
        config.timers[i].egr().write(|w| w.set_ug(true));
    }

    // --- Frame output loop ---
    loop {
        // 1. Encode per-motor frames from shared throttle state
        // NOTE: bidirectional=false (unidirectional CRC). Switch to true when
        // Phase 4 pin-switching + input-capture telemetry is implemented.
        let mut bufs = [[0u32; DSHOT_DMA_BUFFER_SIZE]; 4];
        for (i, buf) in bufs.iter_mut().enumerate() {
            let throttle = super::MOTOR_THROTTLE[i].load(Ordering::Relaxed);
            let frame =
                cybflight_drivers::dshot::frame::encode_packet(throttle, false, false);
            cybflight_drivers::dshot::frame::packet_to_dma_buffer(
                frame,
                buf,
                DSHOT600_BIT_0,
                DSHOT600_BIT_1,
            );
        }

        // 2. Create DMA transfers (armed, waiting for CC events — timers still stopped)
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

        // 3. Enable CC DMA for each motor channel (DIER.CCxDE)
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.dier().modify(|w| w.set_ccde(ch, true));
        }

        // 4. Reset + start timers → first CC event triggers first DMA word
        for i in 0..config.timer_count as usize {
            config.timers[i].cnt().write(|w| w.set_cnt(0));
            config.timers[i].cr1().modify(|w| w.set_cen(true));
        }

        // 5. Await all 4 DMA transfers
        join4(xfer_m1, xfer_m2, xfer_m3, xfer_m4).await;

        // 6. Zero CCRs (safety belt — last DMA word is 0, but ensure pins stay low)
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.ccr(ch).write(|w| w.set_ccr(0));
        }

        // 7. Disable CC DMA, stop timers
        for m in 0..4 {
            let regs = config.motors[m].timer_regs;
            let ch = config.motors[m].channel_index as usize;
            regs.dier().modify(|w| w.set_ccde(ch, false));
        }
        for i in 0..config.timer_count as usize {
            config.timers[i].cr1().modify(|w| w.set_cen(false));
        }

        // 8. Wait for next frame period (~8 kHz = 125us)
        embassy_time::Timer::after_micros(95).await;
    }
}
