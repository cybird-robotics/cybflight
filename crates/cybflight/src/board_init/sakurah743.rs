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
    // --- LEDs: turn off led1 & led2, use led0 for status ---
    let mut led1 = Led::new(board.leds.led1, false);
    let mut led2 = Led::new(board.leds.led2, false);
    led1.off();
    led2.off();
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

    // --- IMU1: ICM42688P on SPI4 (PE12/13/14, CS=PE11, DRDY=PB2) ---
    static SPI4_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi4 = Spi::new(
        board.spi.spi4,
        board.spi.spi4_sck,
        board.spi.spi4_mosi,
        board.spi.spi4_miso,
        board.spi.spi4_tx_dma,
        board.spi.spi4_rx_dma,
        spi_config,
    );
    let spi4_bus = SPI4_BUS.init(Mutex::new(spi4));
    let dev4 = SpiDevice::new(spi4_bus, board.sensors.gyro1_cs);

    let mut delay = embassy_time::Delay;
    match Icm426xx::new(dev4, board.sensors.gyro1_drdy, &mut delay).await {
        Ok(imu1) => {
            defmt::info!("IMU1 init OK");
            spawner
                .spawn(imu_reader_task(imu1, board.sensors.gyro1_align))
                .unwrap();
        }
        Err(e) => defmt::error!("IMU1 init failed: {}", e),
    }

    // TODO: spawn IMU2 when fusion task exists
    // IMU2: IIM42652 on SPI1 (PA5/6/7, CS=PA4, DRDY=PC4)
}
