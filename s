[1mdiff --git a/crates/bsp/sakurah743/src/lib.rs b/crates/bsp/sakurah743/src/lib.rs[m
[1mindex a584cf9..23799fc 100644[m
[1m--- a/crates/bsp/sakurah743/src/lib.rs[m
[1m+++ b/crates/bsp/sakurah743/src/lib.rs[m
[36m@@ -229,7 +229,6 @@[m [mpub struct SerialPins {[m
     pub uart7_rx: hal::Peri<'static, hal::peripherals::PE7>,[m
     pub uart7_cts: hal::Peri<'static, hal::peripherals::PE10>,[m
     pub uart7_rts: hal::Peri<'static, hal::peripherals::PE9>,[m
[31m-[m
 }[m
 [m
 pub struct AdcPins {[m
[36m@@ -361,21 +360,23 @@[m [mfn board_config() -> Config {[m
         use hal::rcc::*;[m
         config.rcc.pll1 = Some(Pll {[m
             source: PllSource::HSI,[m
[31m-            prediv: PllPreDiv::DIV4,   // 64 / 4 = 16 MHz ref[m
[31m-            mul:    PllMul::MUL60,     // 16 * 60 = 960 MHz VCO[m
[31m-            fracn:  None,[m
[31m-            divp:   Some(PllDiv::DIV2), // 480 MHz SYSCLK[m
[31m-            divq:   Some(PllDiv::DIV4), // 240 MHz for SPI123[m
[31m-            divr:   None,[m
[32m+[m[32m            prediv: PllPreDiv::DIV4, // 64 / 4 = 16 MHz ref[m
[32m+[m[32m            mul: PllMul::MUL60,      // 16 * 60 = 960 MHz VCO[m
[32m+[m[32m            fracn: None,[m
[32m+[m[32m            divp: Some(PllDiv::DIV2), // 480 MHz SYSCLK[m
[32m+[m[32m            divq: Some(PllDiv::DIV4), // 240 MHz for SPI123[m
[32m+[m[32m            divr: None,[m
         });[m
[31m-        config.rcc.sys      = Sysclk::PLL1_P;[m
[31m-        config.rcc.ahb_pre  = AHBPrescaler::DIV2;  // 240 MHz AHB[m
[31m-        config.rcc.apb1_pre = APBPrescaler::DIV2;  // 120 MHz[m
[32m+[m[32m        config.rcc.sys = Sysclk::PLL1_P;[m
[32m+[m[32m        config.rcc.ahb_pre = AHBPrescaler::DIV2; // 240 MHz AHB[m
[32m+[m[32m        config.rcc.apb1_pre = APBPrescaler::DIV2; // 120 MHz[m
         config.rcc.apb2_pre = APBPrescaler::DIV2;[m
         config.rcc.apb3_pre = APBPrescaler::DIV2;[m
         config.rcc.apb4_pre = APBPrescaler::DIV2;[m
[31m-        config.rcc.hsi48 = Some(Hsi48Config { sync_from_usb: true }); // USB clock[m
[31m-        config.rcc.mux.usbsel    = mux::Usbsel::HSI48;[m
[32m+[m[32m        config.rcc.hsi48 = Some(Hsi48Config {[m
[32m+[m[32m            sync_from_usb: true,[m
[32m+[m[32m        }); // USB clock[m
[32m+[m[32m        config.rcc.mux.usbsel = mux::Usbsel::HSI48;[m
         config.rcc.mux.spi123sel = mux::Saisel::PLL1_Q;[m
     }[m
     config[m
[36m@@ -487,7 +488,6 @@[m [mpub fn init() -> (Board, hal::usart::UartTx<'static, hal::mode::Blocking>) {[m
         uart7_rx: p.PE7,[m
         uart7_cts: p.PE10,[m
         uart7_rts: p.PE9,[m
[31m-[m
     };[m
 [m
     let adc = AdcPins {[m
[36m@@ -543,11 +543,11 @@[m [mpub fn init() -> (Board, hal::usart::UartTx<'static, hal::mode::Blocking>) {[m
     let sensors = SensorPins {[m
         gyro1_cs,[m
         gyro1_drdy,[m
[31m-        gyro1_align: SensorAlign::Cw0DegFlip,[m
[32m+[m[32m        gyro1_align: SensorAlign::Cw0Deg,[m
 [m
         gyro2_cs,[m
         gyro2_drdy,[m
[31m-        gyro2_align: SensorAlign::Cw0DegFlip,[m
[32m+[m[32m        gyro2_align: SensorAlign::Cw0Deg,[m
 [m
         baro2_cs,[m
 [m
[1mdiff --git a/crates/cybflight/Cargo.toml b/crates/cybflight/Cargo.toml[m
[1mindex bb4181b..e1c6e10 100644[m
[1m--- a/crates/cybflight/Cargo.toml[m
[1m+++ b/crates/cybflight/Cargo.toml[m
[36m@@ -22,7 +22,7 @@[m [mbsp-sakurah743 = { path = "../bsp/sakurah743", optional = true }[m
 bsp-foxeerh743 = { path = "../bsp/foxeerh743", optional = true }[m
 bsp-types = { path = "../bsp/types" }[m
 cybflight-drivers = { path = "../drivers" }[m
[31m-cybflight-msgs = { version = "0.1.2", registry = "utadr" }[m
[32m+[m[32mcybflight-msgs = { path = "../../../cybflight-msgs" }[m
 cybflight-core = { path = "../cybflight_core/" }[m
 [m
 # Embassy runtime pieces (board-agnostic)[m
[1mdiff --git a/crates/cybflight/src/control/attitude_control.rs b/crates/cybflight/src/control/attitude_control.rs[m
[1mindex e55a40c..a96700c 100644[m
[1m--- a/crates/cybflight/src/control/attitude_control.rs[m
[1m+++ b/crates/cybflight/src/control/attitude_control.rs[m
[36m@@ -1,5 +1,5 @@[m
 use cybflight_core::{[m
[31m-    attitude_control::{self, geometric_controller, AttitudeControlOutput},[m
[32m+[m[32m    attitude_control::{self, AttitudeControlOutput, geometric_controller},[m
     mixer::LinearAllocator,[m
 };[m
 use embassy_time::Instant;[m
[36m@@ -13,7 +13,7 @@[m [muse crate::{[m
     motors::ACTUATOR_MOTORS,[m
     msgs,[m
     sensors::{self, MANUAL_CONTROL},[m
[31m-    vehicle::{self, quadrotor_allocator, QUADROTOR_BODY},[m
[32m+[m[32m    vehicle::{self, QUADROTOR_BODY, quadrotor_allocator},[m
 };[m
 [m
 pub struct AttitudeControl<const N: usize> {[m
[1mdiff --git a/crates/cybflight/src/main.rs b/crates/cybflight/src/main.rs[m
[1mindex bb3ae50..b5959a3 100644[m
[1m--- a/crates/cybflight/src/main.rs[m
[1m+++ b/crates/cybflight/src/main.rs[m
[36m@@ -51,9 +51,9 @@[m [masync fn main(spawner: embassy_executor::Spawner) {[m
     spawner[m
         .spawn(cybflight::watchdog::iwdg_feed_task())[m
         .unwrap_or_else(|_| defmt::panic!("failed to spawn IWDG feed task"));[m
[31m-    // spawner[m
[31m-    //     .spawn(cybflight::sensors::attitude::mahony_task())[m
[31m-    //     .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude task"));[m
[32m+[m[32m    spawner[m
[32m+[m[32m        .spawn(cybflight::sensors::attitude::mahony_task())[m
[32m+[m[32m        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude task"));[m
     //[m
     // spawner[m
     //     .spawn(cybflight::control::nmpc_driver::nmpc_task())[m
[1mdiff --git a/crates/cybflight/src/sensors/attitude.rs b/crates/cybflight/src/sensors/attitude.rs[m
[1mindex 8795500..9f5734d 100644[m
[1m--- a/crates/cybflight/src/sensors/attitude.rs[m
[1m+++ b/crates/cybflight/src/sensors/attitude.rs[m
[36m@@ -11,7 +11,7 @@[m [muse cybflight_msgs as msgs;[m
 pub async fn mahony_task() {[m
     let mut sub = IMU_1.subscriber().unwrap();[m
     let publisher = VEHICLE_ATTITUDE.immediate_publisher();[m
[31m-    let mut mahony = Mahony::<f32>::new();[m
[32m+[m[32m    let mut mahony: Mahony<f32> = Mahony::<f32>::new();[m
     let mut prev_timestamp: Option<Instant> = None;[m
 [m
     loop {[m
[1mdiff --git a/crates/cybflight/src/sensors/imu.rs b/crates/cybflight/src/sensors/imu.rs[m
[1mindex 71e282c..e6aaa27 100644[m
[1m--- a/crates/cybflight/src/sensors/imu.rs[m
[1m+++ b/crates/cybflight/src/sensors/imu.rs[m
[36m@@ -1,21 +1,21 @@[m
 use core::sync::atomic::{AtomicBool, Ordering};[m
 [m
[32m+[m[32muse crate::apply_alignment;[m
[32m+[m[32muse crate::hal;[m
 use bsp_types::SensorAlign;[m
 use cybflight_core::butterworth::ButterworthFilter;[m
 use cybflight_drivers::imu::ReadImu;[m
 use cybflight_drivers::imu::icm426xx::Icm426xx;[m
 use cybflight_drivers::imu::mpu6x00::Mpu6x00;[m
[32m+[m[32muse cybflight_msgs as msgs;[m
 use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;[m
 use embassy_sync::blocking_mutex::raw::{CriticalSectionRawMutex, NoopRawMutex};[m
 use embassy_sync::mutex::Mutex;[m
 use embassy_sync::pubsub::PubSubChannel;[m
 use embassy_time::{Instant, Timer};[m
[31m-use nalgebra::Vector3;[m
[31m-use crate::hal;[m
[31m-use cybflight_msgs as msgs;[m
[31m-use crate::apply_alignment;[m
 use hal::gpio::Output;[m
 use hal::spi::{self, Spi};[m
[32m+[m[32muse nalgebra::Vector3;[m
 [m
 pub type SpiBus = Spi<'static, hal::mode::Async, spi::mode::Master>;[m
 pub type SpiBusMtx = Mutex<NoopRawMutex, SpiBus>;[m
[36m@@ -129,7 +129,7 @@[m [mimpl<D: ReadImu> ImuReader<D> {[m
 [m
     pub async fn run([m
         &mut self,[m
[31m-        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1>,[m
[32m+[m[32m        channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,[m
     ) -> ! {[m
         let publisher = channel.immediate_publisher();[m
         let sample_rate = self.imu.sample_rate_hz();[m
[36m@@ -225,7 +225,7 @@[m [mimpl<D: ReadImu> ImuReader<D> {[m
 #[embassy_executor::task(pool_size = 2)][m
 pub async fn icm_reader_task([m
     mut reader: ImuReader<IcmDev>,[m
[31m-    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1>,[m
[32m+[m[32m    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,[m
 ) {[m
     reader.run(channel).await;[m
 }[m
[36m@@ -233,7 +233,7 @@[m [mpub async fn icm_reader_task([m
 #[embassy_executor::task(pool_size = 2)][m
 pub async fn mpu_reader_task([m
     mut reader: ImuReader<MpuDev>,[m
[31m-    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1>,[m
[32m+[m[32m    channel: &'static PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1>,[m
 ) {[m
     reader.run(channel).await;[m
 }[m
[1mdiff --git a/crates/cybflight/src/sensors/mod.rs b/crates/cybflight/src/sensors/mod.rs[m
[1mindex 835a196..4ac555a 100644[m
[1m--- a/crates/cybflight/src/sensors/mod.rs[m
[1m+++ b/crates/cybflight/src/sensors/mod.rs[m
[36m@@ -14,12 +14,10 @@[m [muse embassy_sync::{blocking_mutex::raw::CriticalSectionRawMutex, pubsub::PubSubC[m
 pub static GYRO_CALIBRATED: AtomicBool = AtomicBool::new(false);[m
 [m
 // IMU 1: CAP=4 (small queue, fresh data preferred), SUBS=4 (attitude + telemetry + shell + spare), PUBS=1.[m
[31m-pub static IMU_1: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1> =[m
[31m-    PubSubChannel::new();[m
[32m+[m[32mpub static IMU_1: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1> = PubSubChannel::new();[m
 [m
 // IMU 2: same sizing. Empty on single-IMU boards (shell prints "no data").[m
[31m-pub static IMU_2: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 4, 1> =[m
[31m-    PubSubChannel::new();[m
[32m+[m[32mpub static IMU_2: PubSubChannel<CriticalSectionRawMutex, msgs::Imu, 4, 6, 1> = PubSubChannel::new();[m
 [m
 // CAP=4, SUBS=6 (attitude_control + nmpc + CRSF telem + esp_bridge + shell stream + spare),[m
 // PUBS=1 (single attitude estimator).[m
[1mdiff --git a/crates/drivers/Cargo.toml b/crates/drivers/Cargo.toml[m
[1mindex a389480..eb5846f 100644[m
[1m--- a/crates/drivers/Cargo.toml[m
[1m+++ b/crates/drivers/Cargo.toml[m
[36m@@ -5,7 +5,7 @@[m [medition = "2024"[m
 publish = false[m
 [m
 [dependencies][m
[31m-cybflight-msgs = { version = "0.1.0", registry = "utadr" }[m
[32m+[m[32mcybflight-msgs = { path = "../../../cybflight-msgs" }[m
 embedded-hal = "1.0"[m
 embedded-hal-async = "1.0"[m
 defmt = "1.0.1"[m
