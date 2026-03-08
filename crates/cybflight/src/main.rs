#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;

#[embassy_executor::main]
async fn main(spawner: embassy_executor::Spawner) {
    cybflight::platform::enable_icache();
    let board = cybflight::bsp::init();
    defmt::info!("cybflight: {} starting", cybflight::bsp::BOARD_NAME);
    cybflight::status::STATUS.sender().send(cybflight::status::SystemStatus::Alive);
    cybflight::board_init::init(&spawner, board).await;
    spawner
        .spawn(cybflight::sensors::attitude::mahony_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude task"));

    spawner
        .spawn(cybflight::control::nmpc_driver::nmpc_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn NMPC driver task"));

    spawner
        .spawn(cybflight::usb_serial::imu_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn IMU stream task"));
    spawner
        .spawn(cybflight::usb_serial::att_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn attitude stream task"));
    spawner
        .spawn(cybflight::usb_serial::ocp_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn OCP stream task"));
    spawner
        .spawn(cybflight::usb_serial::rc_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC stream task"));
    spawner
        .spawn(cybflight::usb_serial::rc_link_stream_task())
        .unwrap_or_else(|_| defmt::panic!("failed to spawn RC link stream task"));

    defmt::info!("all init done");
}
