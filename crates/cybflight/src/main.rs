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
    defmt::info!("all init done");
}
