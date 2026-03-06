#![no_std]
#![no_main]

use defmt_rtt as _;
use panic_probe as _;

#[embassy_executor::main]
async fn main(spawner: embassy_executor::Spawner) {
    let board = cybflight::bsp::init();
    defmt::info!("cybflight: {} starting", cybflight::bsp::BOARD_NAME);
    cybflight::status::STATUS.sender().send(cybflight::status::SystemStatus::Alive);
    cybflight::board_init::init(&spawner, board).await;
    defmt::info!("all init done");
}
