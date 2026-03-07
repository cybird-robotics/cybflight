use cybflight_drivers::imu::icm426xx::Icm426xx;
use cybflight_drivers::imu::mpu6x00::Mpu6x00;
use cybflight_drivers::imu::{DetectedImu, probe_imu_raw};
use cybflight_drivers::led::Led;
use embassy_embedded_hal::shared_bus::asynch::spi::SpiDevice;
use embassy_executor::Spawner;
use embassy_sync::mutex::Mutex;
use embassy_time::Timer;
use static_cell::StaticCell;

use crate::bsp;
use crate::hal;
use crate::sensors::imu::{ImuReader, SpiBusMtx, icm_reader_task, mpu_reader_task};
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

    // --- IMU1 on SPI2 (PB13/14/15, CS=PB12, DRDY=PD0) ---
    // Probe WHO_AM_I on raw bus before wrapping in Mutex/SpiDevice.
    let mut spi2 = Spi::new(
        board.spi.spi2,
        board.spi.spi2_sck,
        board.spi.spi2_mosi,
        board.spi.spi2_miso,
        board.spi.spi2_tx_dma,
        board.spi.spi2_rx_dma,
        spi_config,
    );
    let mut cs = board.sensors.gyro1_cs;
    let detected = probe_imu_raw(&mut spi2, &mut cs).await;
    defmt::info!("IMU probe: {:?}", detected);

    // Wrap bus in shared mutex now that probe is done.
    static SPI2_BUS: StaticCell<SpiBusMtx> = StaticCell::new();
    let spi2_bus = SPI2_BUS.init(Mutex::new(spi2));
    let dev2 = SpiDevice::new(spi2_bus, cs);

    let mut delay = embassy_time::Delay;
    const ACCEL_CUTOFF_HZ: f32 = 20.0;
    const GYRO_CUTOFF_HZ: f32 = 150.0;

    match detected {
        Ok(DetectedImu::Icm42605)
        | Ok(DetectedImu::Icm42622P)
        | Ok(DetectedImu::Icm42688P)
        | Ok(DetectedImu::Iim42652)
        | Ok(DetectedImu::Iim42653) => {
            match Icm426xx::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
                Ok(imu1) => {
                    defmt::info!("IMU1 init OK (ICM)");
                    spawner
                        .spawn(icm_reader_task(ImuReader::new(
                            imu1,
                            board.sensors.gyro1_align,
                            ACCEL_CUTOFF_HZ,
                            GYRO_CUTOFF_HZ,
                        )))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 ICM init failed: {}", e),
            }
        }
        Ok(DetectedImu::Mpu6000) | Ok(DetectedImu::Mpu6500) => {
            match Mpu6x00::new(dev2, board.sensors.gyro1_drdy, &mut delay).await {
                Ok(imu1) => {
                    defmt::info!("IMU1 init OK (MPU)");
                    spawner
                        .spawn(mpu_reader_task(ImuReader::new(
                            imu1,
                            board.sensors.gyro1_align,
                            ACCEL_CUTOFF_HZ,
                            GYRO_CUTOFF_HZ,
                        )))
                        .unwrap();
                }
                Err(e) => defmt::error!("IMU1 MPU init failed: {}", e),
            }
        }
        Err(id) => {
            defmt::error!("Unknown IMU: WHO_AM_I={:#x}", id);
        }
    }
}
