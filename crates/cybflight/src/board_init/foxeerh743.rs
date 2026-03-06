use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::led::Led;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::Spawner;
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::sensors::imu::{SpiBusMtx, imu_reader_task};
use crate::status;
use crate::usb_serial;
use hal::spi::{self, Spi};
use hal::time::Hertz;

pub async fn init(spawner: &Spawner, board: bsp::Board) {
    // --- LED: only 1 LED, use it for status ---
    let led0 = Led::new(board.leds.led0, false);
    spawner.spawn(status::task(led0)).unwrap();

    // --- USB CDC serial ---
    spawner
        .spawn(usb_serial::task(
            board.usb.usb_otg_fs,
            board.usb.dp,
            board.usb.dm,
        ))
        .unwrap();

    // Wait for power to stabilize before touching SPI devices.
    Timer::after_millis(100).await;

    let mut spi_config = spi::Config::default();
    spi_config.frequency = Hertz(1_000_000);
    spi_config.mode = spi::MODE_3;

    // --- IMU1: ICM42688P on SPI2 (PB13/14/15, CS=PB12, DRDY=PD0) ---
    static SPI2_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi2 = Spi::new(
        board.spi.spi2,
        board.spi.spi2_sck,
        board.spi.spi2_mosi,
        board.spi.spi2_miso,
        board.spi.spi2_tx_dma,
        board.spi.spi2_rx_dma,
        spi_config,
    );
    let spi2_bus = SPI2_BUS.init(Mutex::new(spi2));
    let dev2 = SpiDevice::new(spi2_bus, board.sensors.gyro1_cs);

    let mut delay = embassy_time::Delay;
    match Icm426xx::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
        Ok(imu1) => {
            defmt::info!("IMU1 init OK");
            spawner
                .spawn(imu_reader_task(imu1, board.sensors.gyro1_align))
                .unwrap();
        }
        Err(e) => defmt::error!("IMU1 init failed: {}", e),
    }
}
