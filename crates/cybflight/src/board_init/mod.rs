#[cfg(feature = "board_sakurah743")]
mod sakurah743;
#[cfg(feature = "board_sakurah743")]
pub use sakurah743::init;

#[cfg(feature = "board_foxeerh743")]
mod foxeerh743;
#[cfg(feature = "board_foxeerh743")]
pub use foxeerh743::init;

#[cfg(feature = "board_micoair743v2")]
mod micoair743v2;
#[cfg(feature = "board_micoair743v2")]
pub use micoair743v2::init;
