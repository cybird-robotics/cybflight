#![no_std]
#![no_main]

use bsp::hal;
use bsp_sakurah743 as bsp;
use cybflight::status;
use cybflight::{IMU_CHANNEL, ImuMessage, ImuSample, apply_alignment};
use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::led::Led;
use defmt_rtt as _;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::Spawner;
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Instant, Timer};
use hal::gpio::Output;
use hal::spi::{self, Spi};
use hal::time::Hertz;
use panic_probe as _;
use static_cell::StaticCell;

type SpiBus = Spi<'static, hal::mode::Async, spi::mode::Master>;
type SpiBusMtx = Mutex<NoopRawMutex, SpiBus>;
type StatusLed = Led<Output<'static>>;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = bsp::init();

    defmt::info!("cybflight: {} starting", bsp::BOARD_NAME);

    // --- Status LEDs ---
    status::STATUS.sender().send(status::SystemStatus::Alive);
    let led0 = Led::new(p.leds.led0, false);
    let led1 = Led::new(p.leds.led1, false);
    let led2 = Led::new(p.leds.led2, false);
    spawner.spawn(status_led_task(led0, led1, led2)).unwrap();

    // --- USB CDC serial ---
    spawner
        .spawn(usb_serial_task(p.usb.usb_otg_fs, p.usb.dp, p.usb.dm))
        .unwrap();

    // Wait for power to stabilize before touching SPI devices.
    Timer::after_millis(100).await;

    let mut spi_config = spi::Config::default();
    spi_config.frequency = Hertz(1_000_000);
    spi_config.mode = spi::MODE_3;

    // --- IMU1: ICM42688P on SPI4 (PE12/13/14, CS=PE11, DRDY=PB2) ---
    static SPI4_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi4 = Spi::new(
        p.spi.spi4,
        p.spi.spi4_sck,
        p.spi.spi4_mosi,
        p.spi.spi4_miso,
        p.spi.spi4_tx_dma,
        p.spi.spi4_rx_dma,
        spi_config,
    );
    let spi4_bus = SPI4_BUS.init(Mutex::new(spi4));
    let dev4 = SpiDevice::new(spi4_bus, p.sensors.gyro1_cs);

    let mut delay = embassy_time::Delay;
    match Icm426xx::new(dev4, p.sensors.gyro1_drdy, &mut delay).await {
        Ok(imu1) => {
            defmt::info!("IMU1 init OK");
            spawner
                .spawn(imu_task(1, imu1, p.sensors.gyro1_align))
                .unwrap();
        }
        Err(e) => defmt::error!("IMU1 init failed: {}", e),
    }

    // --- IMU2: IIM42652 on SPI1 (PA5/6/7, CS=PA4, DRDY=PC4) ---
    static SPI1_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi1 = Spi::new(
        p.spi.spi1,
        p.spi.spi1_sck,
        p.spi.spi1_mosi,
        p.spi.spi1_miso,
        p.spi.spi1_tx_dma,
        p.spi.spi1_rx_dma,
        spi_config,
    );
    let spi1_bus = SPI1_BUS.init(Mutex::new(spi1));
    let dev1 = SpiDevice::new(spi1_bus, p.sensors.gyro2_cs);

    match Icm426xx::new(dev1, p.sensors.gyro2_drdy, &mut delay).await {
        Ok(imu2) => {
            defmt::info!("IMU2 init OK");
            spawner
                .spawn(imu_task(2, imu2, p.sensors.gyro2_align))
                .unwrap();
        }
        Err(e) => defmt::error!("IMU2 init failed: {}", e),
    }

    defmt::info!("all init done");
}

type ImuDev = Icm426xx<
    SpiDevice<'static, NoopRawMutex, SpiBus, Output<'static>>,
    hal::exti::ExtiInput<'static>,
>;

#[embassy_executor::task(pool_size = 2)]
async fn imu_task(source: u8, mut imu: ImuDev, align: bsp_types::SensorAlign) {
    loop {
        match imu.read().await {
            Ok(reading) => {
                let sample = ImuSample {
                    accel: apply_alignment(align, reading.accel),
                    gyro: apply_alignment(align, reading.gyro),
                    temp_c: reading.temp_c,
                    timestamp: Instant::now(),
                };
                let _ = IMU_CHANNEL.try_send(ImuMessage { source, sample });
            }
            Err(e) => {
                defmt::warn!("IMU{} read error: {}", source, e);
                Timer::after_millis(10).await;
            }
        }
    }
}

#[embassy_executor::task]
async fn status_led_task(led0: StatusLed, led1: StatusLed, led2: StatusLed) {
    status::run(led0, led1, led2).await
}

#[embassy_executor::task]
async fn usb_serial_task(
    usb_otg: hal::Peri<'static, hal::peripherals::USB_OTG_FS>,
    dp: hal::Peri<'static, hal::peripherals::PA12>,
    dm: hal::Peri<'static, hal::peripherals::PA11>,
) {
    cybflight::usb_serial::run(usb_otg, dp, dm).await
}
