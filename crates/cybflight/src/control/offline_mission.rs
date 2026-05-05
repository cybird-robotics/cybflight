//! Offline-prebaked mission trajectories (multi-profile).
//!
//! Each [`MissionProfile`] holds a self-contained schedule that the
//! `mission_planner` task can feed directly into a [`MincoSnap`] solver,
//! bypassing the on-device BFGS optimizer. Profiles were solved offline by
//! a heavier planner and copied verbatim from the corresponding
//! `tmp/planning_results/*.yaml` file.
//!
//! `timestamps[i]` is the absolute time of arrival at `waypoints[i]`
//! measured from the trajectory start. Segment durations are recovered
//! by consecutive differences, with the first segment running from
//! `t=0` to `timestamps[0]`.
//!
//! ## MINCO contract (n = `profile.num_pieces()`)
//!
//! For an `n`-piece trajectory, `MincoSnap` consumes `n−1` intermediate
//! waypoints plus a tail. The schedule provides `n` waypoints —
//! `wp[0..n−1]` are the intermediates and `wp[n−1]` is the tail. None of
//! the waypoint slots holds the head pose; that comes from the live
//! `ACTIVE_POSITION_SETPOINT` (or [`MissionProfile::start_pos`] when
//! `OFFLINE_USE_YAML_START` is set).
//!
//! ## Profile selection
//!
//! Selection is runtime, addressed by `(env, variant, speed)`. The active
//! index lives in [`ACTIVE_PROFILE`] (an `AtomicUsize`); the shell
//! `mission set <env> <variant> <speed>` writes it and the value is
//! mirrored into `VehicleParams.mission_profile` for flash persistence.
//! At boot, [`init_active_from_index`] re-applies the persisted choice.
//!
//! [`MincoSnap`]: cybflight_core::trajectory_planning::minco_snap::MincoSnap

#![cfg(feature = "outer_mpc")]

use core::sync::atomic::{AtomicUsize, Ordering};

/// Largest piece count across all profiles. The mission planner sizes its
/// scratch arrays to this bound so the offline path stays alloc-free
/// regardless of which profile is active. Bump this if a new profile
/// exceeds the current ceiling.
pub const OFFLINE_MAX_PIECES: usize = 64;

/// One offline trajectory schedule.
#[derive(Clone, Copy)]
pub struct MissionProfile {
    /// Internal name shown by `mission list` / `mission get`.
    pub name: &'static str,
    /// Selection axis: environment.
    pub env: &'static str,
    /// Selection axis: trajectory family (currently always `"splits"`).
    pub variant: &'static str,
    /// Selection axis: speed tier.
    pub speed: &'static str,
    /// Recorded mission start position [m]. Used as the head pose only when
    /// `OFFLINE_USE_YAML_START` is `true` in `mission_planner.rs`; otherwise
    /// the live `ACTIVE_POSITION_SETPOINT` is the head.
    pub start_pos: [f32; 3],
    /// Intermediate + tail waypoints (length == `timestamps.len()`).
    pub waypoints: &'static [[f32; 3]],
    /// Absolute time of arrival at each waypoint [s] (length == `waypoints.len()`).
    pub timestamps: &'static [f32],
}

impl MissionProfile {
    pub const fn num_pieces(&self) -> usize {
        self.waypoints.len()
    }

    /// Whether this profile's environment matches the firmware build.
    ///
    /// `est_pos_mocap` builds (indoor lab / HIL) only accept `env="indoor"`
    /// profiles; `est_pos_gps` builds (outdoor) only accept `env="outdoor"`.
    /// The `outer_mpc + est_pos_*` mutual-exclusion is enforced at compile
    /// time in `control/mod.rs`.
    pub fn is_compatible_with_build(&self) -> bool {
        self.env.eq_ignore_ascii_case(BUILD_ENV)
    }
}

// ─── Build-time environment ──────────────────────────────────────────
//
// Selected by the position-source feature:
// - `est_pos_mocap` → "indoor" (VICON / lab)
// - `est_pos_gps`   → "outdoor" (u-blox M10 NAV-PVT)
//
// The compile_error guards in `control/mod.rs` ensure exactly one is set
// whenever this module is compiled (`outer_mpc` requires it).

/// Indoor build (mocap / VICON position source).
#[cfg(feature = "est_pos_mocap")]
pub const BUILD_ENV: &str = "indoor";

/// Outdoor build (GPS position source).
#[cfg(feature = "est_pos_gps")]
pub const BUILD_ENV: &str = "outdoor";

// ─── Indoor, SplitS, Slow ────────────────────────────────────────────

static INDOOR_SPLITS_SLOW_WP: [[f32; 3]; 60] = [
    [-2.31, -3.483, 1.032],
    [-1.187, -3.222, 1.269],
    [-0.3347, -2.231, 1.6],
    [-0.6571, -1.146, 1.657],
    [-1.37, 0.2179, 1.514],
    [-1.837, 1.643, 1.199],
    [-1.14, 3.186, 0.7041],
    [0.7796, 3.265, 0.6496],
    [2.285, 1.635, 1.203],
    [2.585, 0.2585, 1.59],
    [2.604, -1.092, 1.824],
    [2.553, -2.107, 1.795],
    [2.555, -2.529, 1.581],
    [2.581, -2.515, 1.281],
    [2.543, -2.106, 1.007],
    [2.22, -1.198, 0.8225],
    [1.458, -0.2428, 0.8563],
    [0.3081, 0.3508, 0.9944],
    [-1.327, 0.2032, 1.102],
    [-2.4, -0.8644, 1.054],
    [-2.39, -2.211, 1.003],
    [-1.53, -3.067, 1.113],
    [-0.5896, -3.087, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7456, -1.084, 1.634],
    [-1.399, 0.2807, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.213, 3.315, 0.6677],
    [0.7386, 3.371, 0.6137],
    [2.285, 1.635, 1.203],
    [2.581, 0.246, 1.592],
    [2.602, -1.1, 1.825],
    [2.553, -2.107, 1.795],
    [2.554, -2.529, 1.581],
    [2.58, -2.515, 1.281],
    [2.543, -2.106, 1.007],
    [2.221, -1.197, 0.822],
    [1.459, -0.2409, 0.856],
    [0.3082, 0.3508, 0.9944],
    [-1.327, 0.1998, 1.102],
    [-2.4, -0.8675, 1.053],
    [-2.39, -2.211, 1.003],
    [-1.532, -3.063, 1.114],
    [-0.5915, -3.083, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7479, -1.081, 1.635],
    [-1.405, 0.2845, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.18, 3.299, 0.6677],
    [0.7865, 3.342, 0.6125],
    [2.285, 1.635, 1.202],
    [2.56, 0.1874, 1.626],
    [2.566, -1.17, 1.866],
    [2.554, -2.107, 1.797],
    [2.588, -2.404, 1.577],
    [2.61, -2.389, 1.285],
    [2.54, -2.108, 1.003],
    [1.593, -0.7033, 0.7181],
    [0.4993, 0.2339, 0.9388],
    [0.3099, 0.3554, 1.0],
];

static INDOOR_SPLITS_SLOW_TS: [f32; 60] = [
    0.7289, 1.458, 2.188, 2.633, 3.079, 3.524, 4.176, 4.828, 5.48, 5.848, 6.216, 6.584, 6.888,
    7.193, 7.498, 7.893, 8.289, 8.684, 9.197, 9.709, 10.22, 10.71, 11.2, 11.69, 12.08, 12.47,
    12.86, 13.52, 14.19, 14.85, 15.22, 15.58, 15.94, 16.25, 16.55, 16.86, 17.25, 17.65, 18.04,
    18.56, 19.07, 19.58, 20.07, 20.56, 21.05, 21.44, 21.84, 22.23, 22.9, 23.57, 24.24, 24.63,
    25.02, 25.42, 25.7, 25.98, 26.25, 27.02, 27.78, 28.55,
];

const _: () = assert!(INDOOR_SPLITS_SLOW_WP.len() == INDOOR_SPLITS_SLOW_TS.len());
const _: () = assert!(INDOOR_SPLITS_SLOW_WP.len() <= OFFLINE_MAX_PIECES);

pub static INDOOR_SPLITS_SLOW: MissionProfile = MissionProfile {
    name: "indoor_splits_slow",
    env: "indoor",
    variant: "splits",
    speed: "slow",
    start_pos: [-2.5, -3.5, 1.0],
    waypoints: &INDOOR_SPLITS_SLOW_WP,
    timestamps: &INDOOR_SPLITS_SLOW_TS,
};

// ─── Indoor, SplitS, Mid ─────────────────────────────────────────────

static INDOOR_SPLITS_MID_WP: [[f32; 3]; 60] = [
    [-2.31, -3.483, 1.032],
    [-1.188, -3.221, 1.269],
    [-0.3347, -2.231, 1.6],
    [-0.657, -1.145, 1.657],
    [-1.371, 0.2182, 1.515],
    [-1.837, 1.643, 1.199],
    [-1.138, 3.181, 0.7051],
    [0.7805, 3.26, 0.6511],
    [2.285, 1.635, 1.203],
    [2.585, 0.2588, 1.59],
    [2.604, -1.092, 1.825],
    [2.553, -2.107, 1.795],
    [2.555, -2.528, 1.581],
    [2.581, -2.514, 1.281],
    [2.543, -2.106, 1.007],
    [2.22, -1.198, 0.8223],
    [1.458, -0.2425, 0.8562],
    [0.3082, 0.3507, 0.9944],
    [-1.325, 0.2024, 1.102],
    [-2.398, -0.8647, 1.054],
    [-2.39, -2.211, 1.003],
    [-1.53, -3.069, 1.113],
    [-0.5886, -3.09, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7464, -1.083, 1.634],
    [-1.4, 0.2817, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.213, 3.313, 0.6677],
    [0.7383, 3.368, 0.6138],
    [2.285, 1.635, 1.203],
    [2.582, 0.2454, 1.593],
    [2.603, -1.101, 1.826],
    [2.553, -2.107, 1.795],
    [2.554, -2.526, 1.581],
    [2.58, -2.512, 1.281],
    [2.543, -2.106, 1.007],
    [2.221, -1.198, 0.821],
    [1.459, -0.2417, 0.8553],
    [0.3082, 0.3507, 0.9944],
    [-1.326, 0.2003, 1.102],
    [-2.399, -0.867, 1.054],
    [-2.39, -2.211, 1.003],
    [-1.531, -3.065, 1.113],
    [-0.5894, -3.085, 1.369],
    [-0.3345, -2.232, 1.599],
    [-0.7498, -1.08, 1.635],
    [-1.407, 0.285, 1.49],
    [-1.837, 1.643, 1.2],
    [-1.176, 3.295, 0.6691],
    [0.789, 3.339, 0.6142],
    [2.285, 1.635, 1.202],
    [2.56, 0.1884, 1.625],
    [2.566, -1.169, 1.865],
    [2.554, -2.107, 1.797],
    [2.588, -2.405, 1.577],
    [2.61, -2.39, 1.285],
    [2.54, -2.108, 1.003],
    [1.592, -0.7028, 0.7189],
    [0.4989, 0.2339, 0.9391],
    [0.3099, 0.3554, 1.0],
];

static INDOOR_SPLITS_MID_TS: [f32; 60] = [
    0.4098, 0.8196, 1.23, 1.481, 1.731, 1.982, 2.348, 2.715, 3.081, 3.288, 3.495, 3.702, 3.873,
    4.044, 4.215, 4.438, 4.66, 4.883, 5.171, 5.459, 5.747, 6.023, 6.299, 6.576, 6.794, 7.013,
    7.231, 7.604, 7.978, 8.351, 8.556, 8.761, 8.965, 9.136, 9.307, 9.477, 9.7, 9.923, 10.15, 10.43,
    10.72, 11.01, 11.29, 11.56, 11.84, 12.06, 12.28, 12.5, 12.88, 13.25, 13.63, 13.85, 14.07,
    14.29, 14.45, 14.61, 14.76, 15.19, 15.62, 16.05,
];

const _: () = assert!(INDOOR_SPLITS_MID_WP.len() == INDOOR_SPLITS_MID_TS.len());
const _: () = assert!(INDOOR_SPLITS_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static INDOOR_SPLITS_MID: MissionProfile = MissionProfile {
    name: "indoor_splits_mid",
    env: "indoor",
    variant: "splits",
    speed: "mid",
    start_pos: [-2.5, -3.5, 1.0],
    waypoints: &INDOOR_SPLITS_MID_WP,
    timestamps: &INDOOR_SPLITS_MID_TS,
};

// ─── Indoor, SplitS, Fast ────────────────────────────────────────────

// static INDOOR_SPLITS_FAST_WP: [[f32; 3]; 60] = [
//     [-2.385, -3.563, 1.077],
//     [-1.3, -3.436, 1.389],
//     [-0.335, -2.231, 1.6],
//     [-0.3995, -1.016, 1.223],
//     [-1.163, 0.3885, 0.9517],
//     [-1.837, 1.643, 1.199],
//     [-1.135, 3.086, 1.409],
//     [0.6928, 3.185, 1.306],
//     [2.285, 1.635, 1.203],
//     [2.734, 0.2063, 1.309],
//     [2.732, -1.178, 1.58],
//     [2.553, -2.107, 1.795],
//     [2.483, -2.476, 1.671],
//     [2.518, -2.482, 1.341],
//     [2.543, -2.106, 1.007],
//     [2.285, -1.191, 0.7673],
//     [1.519, -0.2076, 0.8247],
//     [0.308, 0.3502, 0.9944],
//     [-1.313, 0.1156, 1.002],
//     [-2.352, -0.9535, 0.9395],
//     [-2.389, -2.211, 1.003],
//     [-1.661, -2.893, 1.198],
//     [-0.77, -2.88, 1.524],
//     [-0.3348, -2.232, 1.599],
//     [-0.6452, -1.218, 1.178],
//     [-1.448, 0.1667, 0.9672],
//     [-1.837, 1.643, 1.2],
//     [-1.047, 3.242, 1.356],
//     [0.6791, 3.331, 1.271],
//     [2.285, 1.635, 1.203],
//     [2.71, 0.1738, 1.326],
//     [2.707, -1.195, 1.601],
//     [2.553, -2.107, 1.795],
//     [2.499, -2.484, 1.656],
//     [2.534, -2.487, 1.33],
//     [2.543, -2.106, 1.007],
//     [2.273, -1.185, 0.7762],
//     [1.502, -0.2005, 0.8267],
//     [0.3081, 0.3502, 0.9944],
//     [-1.307, 0.1094, 0.9973],
//     [-2.346, -0.9616, 0.9287],
//     [-2.389, -2.211, 1.003],
//     [-1.662, -2.901, 1.204],
//     [-0.7722, -2.885, 1.525],
//     [-0.3348, -2.232, 1.599],
//     [-0.6241, -1.207, 1.174],
//     [-1.422, 0.1791, 0.9569],
//     [-1.837, 1.643, 1.2],
//     [-1.061, 3.284, 1.357],
//     [0.689, 3.342, 1.247],
//     [2.284, 1.635, 1.202],
//     [2.722, 0.1099, 1.344],
//     [2.71, -1.246, 1.632],
//     [2.554, -2.107, 1.797],
//     [2.485, -2.478, 1.618],
//     [2.52, -2.473, 1.284],
//     [2.54, -2.108, 1.004],
//     [1.699, -0.4978, 0.7795],
//     [0.5322, 0.296, 1.037],
//     [0.3099, 0.3554, 1.0],
// ];

// static INDOOR_SPLITS_FAST_TS: [f32; 60] = [
//     0.1678, 0.3648, 0.5559, 0.6823, 0.8232, 0.9756, 1.186, 1.378, 1.567, 1.678, 1.791, 1.907,
//     1.999, 2.09, 2.181, 2.301, 2.421, 2.541, 2.689, 2.839, 2.993, 3.128, 3.259, 3.396, 3.538,
//     3.679, 3.813, 4.008, 4.203, 4.399, 4.508, 4.619, 4.732, 4.823, 4.914, 5.005, 5.124, 5.243,
//     5.362, 5.511, 5.66, 5.814, 5.948, 6.078, 6.214, 6.354, 6.494, 6.627, 6.827, 7.027, 7.221,
//     7.334, 7.445, 7.555, 7.647, 7.738, 7.828, 8.04, 8.246, 8.452,
// ];

static INDOOR_SPLITS_FAST_WP: [[f32; 3]; 60] = [
    [-2.423, -3.514, 1.048],
    [-1.617, -3.444, 1.308],
    [-0.331, -2.228, 1.595],
    [-0.5492, -0.8822, 1.745],
    [-1.214, 0.3425, 1.662],
    [-1.838, 1.645, 1.199],
    [-1.227, 2.155, 1.042],
    [0.4519, 2.393, 1.034],
    [2.289, 1.631, 1.205],
    [2.811, 0.7344, 1.549],
    [2.685, -0.3515, 1.883],
    [2.546, -2.105, 1.793],
    [2.529, -2.816, 1.477],
    [2.56, -2.899, 1.18],
    [2.548, -2.108, 1.008],
    [2.09, -1.016, 0.9776],
    [1.261, -0.1319, 0.9829],
    [0.3084, 0.3526, 0.9917],
    [-1.004, -0.03513, 1.074],
    [-1.925, -0.8095, 1.057],
    [-2.393, -2.211, 1.007],
    [-1.475, -2.942, 1.167],
    [-0.5922, -2.797, 1.363],
    [-0.3307, -2.232, 1.593],
    [-0.9028, -1.079, 1.784],
    [-1.641, 0.2301, 1.607],
    [-1.841, 1.642, 1.207],
    [-0.8296, 2.481, 1.018],
    [0.4009, 2.606, 1.009],
    [2.289, 1.637, 1.208],
    [2.758, 0.4622, 1.561],
    [2.675, -0.9192, 1.886],
    [2.548, -2.107, 1.791],
    [2.49, -2.682, 1.551],
    [2.485, -2.923, 1.253],
    [2.546, -2.107, 1.008],
    [2.085, -0.8593, 0.9624],
    [1.31, -0.03543, 0.9623],
    [0.309, 0.3534, 0.9917],
    [-0.9416, -0.03766, 1.051],
    [-2.02, -0.9767, 1.024],
    [-2.393, -2.215, 1.008],
    [-1.794, -2.925, 1.131],
    [-0.8325, -2.99, 1.327],
    [-0.3334, -2.226, 1.598],
    [-0.7713, -0.877, 1.797],
    [-1.528, 0.5085, 1.634],
    [-1.839, 1.64, 1.194],
    [-1.165, 2.088, 1.049],
    [0.5164, 2.334, 1.044],
    [2.288, 1.638, 1.208],
    [2.854, 0.6612, 1.558],
    [2.744, -0.4349, 1.887],
    [2.549, -2.106, 1.793],
    [2.552, -2.847, 1.495],
    [2.595, -2.992, 1.174],
    [2.544, -2.108, 1.007],
    [1.6, -0.4852, 0.96],
    [0.477, 0.2408, 0.9711],
    [0.3099, 0.3554, 1.0],
];

static INDOOR_SPLITS_FAST_TS: [f32; 60] = [
    0.1494, 0.3568, 0.6596, 0.8785, 1.078, 1.367, 1.51, 1.683, 1.878, 2.054, 2.261, 2.565, 2.729,
    2.878, 3.03, 3.161, 3.291, 3.433, 3.64, 3.794, 4.012, 4.206, 4.373, 4.556, 4.797, 5.018, 5.239,
    5.417, 5.538, 5.745, 5.916, 6.143, 6.369, 6.503, 6.643, 6.833, 6.983, 7.11, 7.26, 7.45, 7.626,
    7.808, 7.963, 8.137, 8.359, 8.588, 8.818, 9.066, 9.201, 9.37, 9.556, 9.734, 9.943, 10.23,
    10.39, 10.55, 10.72, 10.93, 11.16, 11.41,
];

const _: () = assert!(INDOOR_SPLITS_FAST_WP.len() == INDOOR_SPLITS_FAST_TS.len());
const _: () = assert!(INDOOR_SPLITS_FAST_WP.len() <= OFFLINE_MAX_PIECES);

pub static INDOOR_SPLITS_FAST: MissionProfile = MissionProfile {
    name: "indoor_splits_fast",
    env: "indoor",
    variant: "splits",
    speed: "fast",
    start_pos: [-2.5, -3.5, 1.0],
    waypoints: &INDOOR_SPLITS_FAST_WP,
    timestamps: &INDOOR_SPLITS_FAST_TS,
};

// ─── Outdoor, SplitS, Slow ───────────────────────────────────────────

static OUTDOOR_SPLITS_SLOW_WP: [[f32; 3]; 63] = [
    [-0.2231, -0.09659, 3.012],
    [-1.513, -0.8685, 3.14],
    [-2.386, -2.213, 3.5],
    [-1.821, -2.868, 3.857],
    [-0.8193, -2.922, 4.229],
    [-0.3367, -2.231, 4.501],
    [-0.6694, -1.135, 4.55],
    [-1.373, 0.2438, 4.375],
    [-1.835, 1.642, 3.998],
    [-1.166, 3.122, 3.368],
    [0.7439, 3.179, 3.271],
    [2.283, 1.636, 4.005],
    [2.598, 0.3473, 4.55],
    [2.623, -0.9818, 4.938],
    [2.554, -2.107, 4.993],
    [2.561, -2.932, 4.521],
    [2.657, -2.89, 3.703],
    [2.543, -2.105, 3.009],
    [2.062, -1.204, 2.759],
    [1.173, -0.4098, 2.799],
    [-0.002066, -0.005642, 2.992],
    [-1.438, -0.233, 3.228],
    [-2.348, -1.116, 3.372],
    [-2.388, -2.21, 3.504],
    [-1.582, -3.045, 3.773],
    [-0.6271, -3.075, 4.167],
    [-0.3366, -2.232, 4.499],
    [-0.7425, -1.073, 4.565],
    [-1.407, 0.2973, 4.385],
    [-1.835, 1.643, 3.999],
    [-1.187, 3.168, 3.312],
    [0.73, 3.215, 3.223],
    [2.283, 1.636, 4.005],
    [2.596, 0.3418, 4.56],
    [2.622, -0.9864, 4.946],
    [2.554, -2.107, 4.993],
    [2.561, -2.923, 4.515],
    [2.656, -2.881, 3.7],
    [2.543, -2.105, 3.009],
    [2.063, -1.204, 2.758],
    [1.174, -0.4096, 2.797],
    [-0.002056, -0.005637, 2.992],
    [-1.436, -0.2339, 3.229],
    [-2.345, -1.117, 3.372],
    [-2.388, -2.21, 3.504],
    [-1.582, -3.05, 3.773],
    [-0.625, -3.08, 4.168],
    [-0.3366, -2.232, 4.499],
    [-0.7457, -1.07, 4.563],
    [-1.411, 0.3011, 4.382],
    [-1.835, 1.643, 3.999],
    [-1.164, 3.159, 3.315],
    [0.7665, 3.189, 3.227],
    [2.283, 1.635, 4.004],
    [2.584, 0.2795, 4.609],
    [2.589, -1.062, 5.009],
    [2.555, -2.106, 4.995],
    [2.631, -2.679, 4.487],
    [2.714, -2.632, 3.703],
    [2.539, -2.108, 3.005],
    [1.332, -0.8102, 2.64],
    [0.1838, -0.08655, 2.926],
    [0.0, 0.0, 3.0],
];

static OUTDOOR_SPLITS_SLOW_TS: [f32; 63] = [
    0.5782, 1.156, 1.735, 2.123, 2.51, 2.897, 3.224, 3.55, 3.876, 4.361, 4.846, 5.332, 5.598,
    5.864, 6.13, 6.469, 6.809, 7.148, 7.426, 7.705, 7.983, 8.324, 8.664, 9.004, 9.378, 9.751,
    10.12, 10.43, 10.73, 11.03, 11.52, 12.0, 12.49, 12.76, 13.02, 13.29, 13.63, 13.96, 14.3, 14.58,
    14.86, 15.14, 15.48, 15.82, 16.16, 16.53, 16.91, 17.28, 17.58, 17.88, 18.18, 18.68, 19.17,
    19.66, 19.95, 20.24, 20.53, 20.84, 21.16, 21.47, 22.03, 22.59, 23.15,
];

const _: () = assert!(OUTDOOR_SPLITS_SLOW_WP.len() == OUTDOOR_SPLITS_SLOW_TS.len());
const _: () = assert!(OUTDOOR_SPLITS_SLOW_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_SPLITS_SLOW: MissionProfile = MissionProfile {
    name: "outdoor_splits_slow",
    env: "outdoor",
    variant: "splits",
    speed: "slow",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_SPLITS_SLOW_WP,
    timestamps: &OUTDOOR_SPLITS_SLOW_TS,
};

// ─── Outdoor, SplitS, Mid ────────────────────────────────────────────
//
// Source YAML included a leading `[0, 0, 3]` waypoint that duplicated
// `start_pos`. The other profiles (incl. outdoor_splits_slow) treat
// `wp[0]` as the first intermediate, not the head pose, so the
// duplicate is dropped here to keep the convention uniform — gives 63
// waypoints matching 63 timestamps.
static OUTDOOR_SPLITS_MID_WP: [[f32; 3]; 63] = [
    [-0.2244, -0.09681, 3.013],
    [-1.517, -0.8698, 3.144],
    [-2.386, -2.213, 3.5],
    [-1.823, -2.871, 3.852],
    [-0.8194, -2.928, 4.226],
    [-0.3367, -2.231, 4.501],
    [-0.6732, -1.13, 4.549],
    [-1.376, 0.2518, 4.372],
    [-1.835, 1.642, 3.998],
    [-1.159, 3.145, 3.361],
    [0.7815, 3.181, 3.274],
    [2.283, 1.636, 4.005],
    [2.596, 0.3469, 4.546],
    [2.623, -0.9803, 4.933],
    [2.554, -2.107, 4.993],
    [2.558, -2.946, 4.523],
    [2.655, -2.897, 3.699],
    [2.543, -2.105, 3.009],
    [2.066, -1.2, 2.759],
    [1.179, -0.4041, 2.798],
    [-0.00211, -0.005694, 2.992],
    [-1.423, -0.2426, 3.226],
    [-2.33, -1.119, 3.371],
    [-2.388, -2.21, 3.504],
    [-1.582, -3.06, 3.776],
    [-0.6276, -3.085, 4.168],
    [-0.3366, -2.232, 4.499],
    [-0.7568, -1.038, 4.561],
    [-1.421, 0.3379, 4.374],
    [-1.835, 1.643, 3.999],
    [-1.192, 3.183, 3.31],
    [0.7507, 3.219, 3.226],
    [2.283, 1.636, 4.005],
    [2.594, 0.3401, 4.558],
    [2.62, -0.9875, 4.943],
    [2.555, -2.107, 4.993],
    [2.563, -2.928, 4.517],
    [2.658, -2.883, 3.698],
    [2.543, -2.105, 3.009],
    [2.063, -1.206, 2.758],
    [1.175, -0.4117, 2.797],
    [-0.002138, -0.005762, 2.992],
    [-1.444, -0.2362, 3.23],
    [-2.348, -1.119, 3.373],
    [-2.388, -2.21, 3.503],
    [-1.583, -3.051, 3.772],
    [-0.6209, -3.081, 4.17],
    [-0.3366, -2.232, 4.499],
    [-0.7069, -1.153, 4.566],
    [-1.358, 0.189, 4.405],
    [-1.835, 1.643, 3.999],
    [-1.16, 3.172, 3.307],
    [0.7636, 3.197, 3.22],
    [2.283, 1.635, 4.004],
    [2.585, 0.2588, 4.619],
    [2.588, -1.078, 5.014],
    [2.555, -2.106, 4.995],
    [2.632, -2.676, 4.485],
    [2.713, -2.628, 3.701],
    [2.539, -2.108, 3.005],
    [1.35, -0.8243, 2.636],
    [0.1888, -0.08918, 2.924],
    [0.0, 0.0, 3.0],
];

static OUTDOOR_SPLITS_MID_TS: [f32; 63] = [
    0.4344, 0.8681, 1.301, 1.591, 1.882, 2.174, 2.419, 2.663, 2.905, 3.272, 3.642, 4.0, 4.198,
    4.396, 4.593, 4.85, 5.107, 5.361, 5.57, 5.779, 5.99, 6.243, 6.495, 6.748, 7.032, 7.312, 7.593,
    7.823, 8.048, 8.266, 8.635, 9.005, 9.368, 9.567, 9.765, 9.963, 10.22, 10.47, 10.72, 10.93,
    11.14, 11.35, 11.61, 11.86, 12.11, 12.4, 12.68, 12.96, 13.17, 13.39, 13.63, 14.01, 14.37,
    14.74, 14.96, 15.18, 15.4, 15.63, 15.86, 16.1, 16.51, 16.94, 17.36,
];

const _: () = assert!(OUTDOOR_SPLITS_MID_WP.len() == OUTDOOR_SPLITS_MID_TS.len());
const _: () = assert!(OUTDOOR_SPLITS_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_SPLITS_MID: MissionProfile = MissionProfile {
    name: "outdoor_splits_mid",
    env: "outdoor",
    variant: "splits",
    speed: "mid",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_SPLITS_MID_WP,
    timestamps: &OUTDOOR_SPLITS_MID_TS,
};

// ─── Outdoor, Drag, Slow ─────────────────────────────────────────────
// Source: tmp/planning_results/outdoor_drag_slow/race_0501_drag_outdoor_waypoints.yaml

static OUTDOOR_DRAG_SLOW_WP: [[f32; 3]; 6] = [
    [-0.0, 0.1058, 3.0],
    [-0.0, 1.04, 3.0],
    [-0.0, 3.0, 3.0],
    [-0.0, 4.96, 3.0],
    [-0.0, 5.894, 3.0],
    [0.0, 6.0, 3.0],
];

static OUTDOOR_DRAG_SLOW_TS: [f32; 6] = [0.4441, 0.8882, 1.332, 1.776, 2.22, 2.664];

const _: () = assert!(OUTDOOR_DRAG_SLOW_WP.len() == OUTDOOR_DRAG_SLOW_TS.len());
const _: () = assert!(OUTDOOR_DRAG_SLOW_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_DRAG_SLOW: MissionProfile = MissionProfile {
    name: "outdoor_drag_slow",
    env: "outdoor",
    variant: "drag",
    speed: "slow",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_DRAG_SLOW_WP,
    timestamps: &OUTDOOR_DRAG_SLOW_TS,
};

// ─── Outdoor, Drag, Mid ──────────────────────────────────────────────
// Source: tmp/planning_results/outdoor_drag_mid/race_0501_drag_outdoor_waypoints.yaml
// Same waypoints as drag_slow; faster (uniform 0.333s) per-segment timing.

static OUTDOOR_DRAG_MID_WP: [[f32; 3]; 6] = [
    [-0.0, 0.1058, 3.0],
    [-0.0, 1.04, 3.0],
    [-0.0, 3.0, 3.0],
    [-0.0, 4.96, 3.0],
    [-0.0, 5.894, 3.0],
    [0.0, 6.0, 3.0],
];

static OUTDOOR_DRAG_MID_TS: [f32; 6] = [0.333, 0.666, 0.999, 1.332, 1.665, 1.998];

const _: () = assert!(OUTDOOR_DRAG_MID_WP.len() == OUTDOOR_DRAG_MID_TS.len());
const _: () = assert!(OUTDOOR_DRAG_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_DRAG_MID: MissionProfile = MissionProfile {
    name: "outdoor_drag_mid",
    env: "outdoor",
    variant: "drag",
    speed: "mid",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_DRAG_MID_WP,
    timestamps: &OUTDOOR_DRAG_MID_TS,
};

// ─── Outdoor, Drag-Large, Mid ────────────────────────────────────────
// Source: tmp/planning_results/outdoor_drag-large_mid/race_0501_drag_outdoor_large_waypoints.yaml
// Same straight-line drag layout as the standard drag profiles, scaled
// 2× in length (12 m run vs 6 m).

static OUTDOOR_DRAG_LARGE_MID_WP: [[f32; 3]; 6] = [
    [-0.0, 0.2116, 3.0],
    [-0.0, 2.08, 3.0],
    [-0.0, 6.0, 3.0],
    [-0.0, 9.92, 3.0],
    [-0.0, 11.79, 3.0],
    [0.0, 12.0, 3.0],
];

static OUTDOOR_DRAG_LARGE_MID_TS: [f32; 6] = [0.396, 0.792, 1.188, 1.584, 1.98, 2.376];

const _: () = assert!(OUTDOOR_DRAG_LARGE_MID_WP.len() == OUTDOOR_DRAG_LARGE_MID_TS.len());
const _: () = assert!(OUTDOOR_DRAG_LARGE_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_DRAG_LARGE_MID: MissionProfile = MissionProfile {
    name: "outdoor_drag-large_mid",
    env: "outdoor",
    variant: "drag-large",
    speed: "mid",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_DRAG_LARGE_MID_WP,
    timestamps: &OUTDOOR_DRAG_LARGE_MID_TS,
};

// ─── Outdoor, Drag-Super, Mid ────────────────────────────────────────
// Source: tmp/planning_results/outdoor_drag-super_mid/race_0501_drag_outdoor_super_waypoints.yaml
// Same straight-line drag layout, scaled 4× in length (24 m run vs 6 m).

static OUTDOOR_DRAG_SUPER_MID_WP: [[f32; 3]; 6] = [
    [-0.0, 0.4232, 3.0],
    [-0.0, 4.159, 3.0],
    [-0.0, 12.0, 3.0],
    [-0.0, 19.84, 3.0],
    [-0.0, 23.58, 3.0],
    [0.0, 24.0, 3.0],
];

static OUTDOOR_DRAG_SUPER_MID_TS: [f32; 6] = [0.4709, 0.9419, 1.413, 1.884, 2.355, 2.826];

const _: () = assert!(OUTDOOR_DRAG_SUPER_MID_WP.len() == OUTDOOR_DRAG_SUPER_MID_TS.len());
const _: () = assert!(OUTDOOR_DRAG_SUPER_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_DRAG_SUPER_MID: MissionProfile = MissionProfile {
    name: "outdoor_drag-super_mid",
    env: "outdoor",
    variant: "drag-super",
    speed: "mid",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_DRAG_SUPER_MID_WP,
    timestamps: &OUTDOOR_DRAG_SUPER_MID_TS,
};

// ─── Outdoor, SplitS-Large, Slow ─────────────────────────────────────
// Source: tmp/planning_results/outdoor_splits-large_slow/race_0716_splits_outdoor_large_waypoints.yaml
// Tail waypoint in the YAML is `[-1.665e-16, -6.072e-16, 3]` — i.e. the
// origin to within float epsilon. Rounded to exact zero here.

static OUTDOOR_SPLITS_LARGE_SLOW_WP: [[f32; 3]; 63] = [
    [-0.6731, -0.2904, 3.02],
    [-4.562, -2.609, 3.176],
    [-7.18, -6.641, 3.5],
    [-5.43, -8.635, 3.801],
    [-2.374, -8.802, 4.161],
    [-0.988, -6.693, 4.501],
    [-2.037, -3.465, 4.623],
    [-4.128, 0.6447, 4.469],
    [-5.527, 4.927, 3.999],
    [-3.492, 9.883, 3.023],
    [2.379, 10.1, 2.834],
    [6.869, 4.909, 4.003],
    [7.699, 0.6301, 4.827],
    [7.735, -3.436, 5.251],
    [7.648, -6.323, 4.996],
    [7.706, -7.307, 4.418],
    [7.779, -7.265, 3.684],
    [7.635, -6.323, 3.006],
    [6.472, -3.88, 2.448],
    [3.816, -1.362, 2.541],
    [-0.001149, -0.004693, 2.995],
    [-4.435, -0.6737, 3.435],
    [-7.199, -3.345, 3.547],
    [-7.181, -6.639, 3.502],
    [-4.681, -9.022, 3.621],
    [-1.836, -9.113, 4.028],
    [-0.9879, -6.694, 4.5],
    [-2.194, -3.31, 4.671],
    [-4.182, 0.7834, 4.503],
    [-5.527, 4.927, 3.999],
    [-3.618, 10.08, 2.908],
    [2.309, 10.27, 2.737],
    [6.869, 4.909, 4.003],
    [7.695, 0.6089, 4.844],
    [7.733, -3.452, 5.262],
    [7.648, -6.323, 4.996],
    [7.705, -7.298, 4.415],
    [7.777, -7.257, 3.682],
    [7.635, -6.323, 3.006],
    [6.477, -3.88, 2.447],
    [3.821, -1.36, 2.541],
    [-0.001137, -0.004699, 2.995],
    [-4.432, -0.6789, 3.433],
    [-7.195, -3.35, 3.544],
    [-7.181, -6.639, 3.502],
    [-4.688, -9.013, 3.624],
    [-1.846, -9.102, 4.031],
    [-0.9879, -6.694, 4.5],
    [-2.2, -3.295, 4.671],
    [-4.2, 0.8036, 4.502],
    [-5.527, 4.927, 3.999],
    [-3.531, 9.956, 2.923],
    [2.41, 10.09, 2.747],
    [6.869, 4.909, 4.002],
    [7.66, 0.4328, 4.933],
    [7.647, -3.661, 5.371],
    [7.648, -6.322, 4.997],
    [7.762, -7.005, 4.408],
    [7.824, -6.964, 3.69],
    [7.634, -6.325, 3.003],
    [4.479, -2.661, 2.188],
    [0.67, -0.2935, 2.822],
    [0.0, 0.0, 3.0],
];

static OUTDOOR_SPLITS_LARGE_SLOW_TS: [f32; 63] = [
    0.762, 1.524, 2.286, 2.803, 3.319, 3.836, 4.255, 4.673, 5.092, 5.749, 6.404, 7.061, 7.431,
    7.802, 8.173, 8.443, 8.714, 8.984, 9.387, 9.79, 10.19, 10.65, 11.11, 11.57, 12.05, 12.53,
    13.01, 13.4, 13.79, 14.18, 14.84, 15.5, 16.17, 16.54, 16.9, 17.27, 17.54, 17.81, 18.08, 18.48,
    18.89, 19.29, 19.75, 20.21, 20.67, 21.15, 21.63, 22.11, 22.5, 22.89, 23.28, 23.95, 24.61,
    25.28, 25.68, 26.08, 26.48, 26.73, 26.98, 27.23, 28.0, 28.77, 29.54,
];

const _: () = assert!(OUTDOOR_SPLITS_LARGE_SLOW_WP.len() == OUTDOOR_SPLITS_LARGE_SLOW_TS.len());
const _: () = assert!(OUTDOOR_SPLITS_LARGE_SLOW_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_SPLITS_LARGE_SLOW: MissionProfile = MissionProfile {
    name: "outdoor_splits-large_slow",
    env: "outdoor",
    variant: "splits-large",
    speed: "slow",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_SPLITS_LARGE_SLOW_WP,
    timestamps: &OUTDOOR_SPLITS_LARGE_SLOW_TS,
};

// ─── Outdoor, SplitS-Large, Mid ──────────────────────────────────────
// Source: tmp/planning_results/outdoor_splits-large_mid/race_0716_splits_outdoor_large_waypoints.yaml
// Tail waypoint in the YAML is `[-1.235e-15, 4.163e-16, 3]` — i.e. the
// origin to within float epsilon. Rounded to exact zero here.

static OUTDOOR_SPLITS_LARGE_MID_WP: [[f32; 3]; 63] = [
    [-0.6744, -0.2898, 3.019],
    [-4.567, -2.608, 3.174],
    [-7.178, -6.641, 3.5],
    [-5.43, -8.634, 3.802],
    [-2.375, -8.8, 4.162],
    [-0.99, -6.693, 4.501],
    [-2.035, -3.464, 4.621],
    [-4.126, 0.6459, 4.467],
    [-5.525, 4.927, 3.999],
    [-3.496, 9.882, 3.029],
    [2.38, 10.1, 2.84],
    [6.867, 4.909, 4.003],
    [7.7, 0.6307, 4.825],
    [7.738, -3.436, 5.249],
    [7.649, -6.322, 4.995],
    [7.706, -7.305, 4.418],
    [7.777, -7.264, 3.685],
    [7.634, -6.323, 3.007],
    [6.475, -3.878, 2.449],
    [3.822, -1.358, 2.541],
    [-0.001546, -0.006387, 2.993],
    [-4.446, -0.6999, 3.43],
    [-7.185, -3.367, 3.541],
    [-7.18, -6.638, 3.503],
    [-4.689, -9.033, 3.629],
    [-1.83, -9.127, 4.038],
    [-0.9899, -6.695, 4.5],
    [-2.169, -3.381, 4.66],
    [-4.151, 0.6995, 4.499],
    [-5.525, 4.927, 3.999],
    [-3.555, 10.08, 2.927],
    [2.261, 10.25, 2.759],
    [6.867, 4.909, 4.003],
    [7.701, 0.6162, 4.837],
    [7.74, -3.442, 5.254],
    [7.649, -6.322, 4.995],
    [7.704, -7.31, 4.414],
    [7.777, -7.269, 3.686],
    [7.634, -6.323, 3.007],
    [6.467, -3.864, 2.452],
    [3.809, -1.35, 2.546],
    [-0.00158, -0.006451, 2.993],
    [-4.431, -0.685, 3.425],
    [-7.195, -3.353, 3.538],
    [-7.179, -6.638, 3.502],
    [-4.684, -9.0, 3.634],
    [-1.85, -9.091, 4.041],
    [-0.9899, -6.694, 4.5],
    [-2.202, -3.308, 4.66],
    [-4.199, 0.7854, 4.491],
    [-5.525, 4.927, 3.999],
    [-3.535, 9.974, 2.946],
    [2.412, 10.12, 2.763],
    [6.867, 4.908, 4.003],
    [7.658, 0.4329, 4.926],
    [7.649, -3.659, 5.365],
    [7.65, -6.322, 4.997],
    [7.763, -7.006, 4.409],
    [7.823, -6.965, 3.691],
    [7.632, -6.325, 3.004],
    [4.474, -2.659, 2.186],
    [0.67, -0.2935, 2.821],
    [0.0, 0.0, 3.0],
];

static OUTDOOR_SPLITS_LARGE_MID_TS: [f32; 63] = [
    0.5717, 1.143, 1.715, 2.101, 2.488, 2.875, 3.189, 3.503, 3.817, 4.309, 4.801, 5.293, 5.57,
    5.848, 6.126, 6.329, 6.531, 6.734, 7.036, 7.339, 7.642, 7.987, 8.329, 8.67, 9.032, 9.395,
    9.758, 10.04, 10.33, 10.63, 11.13, 11.62, 12.12, 12.39, 12.67, 12.95, 13.15, 13.35, 13.56,
    13.86, 14.16, 14.46, 14.81, 15.15, 15.49, 15.86, 16.21, 16.58, 16.87, 17.16, 17.46, 17.96,
    18.45, 18.95, 19.25, 19.55, 19.85, 20.04, 20.23, 20.41, 20.99, 21.57, 22.15,
];

const _: () = assert!(OUTDOOR_SPLITS_LARGE_MID_WP.len() == OUTDOOR_SPLITS_LARGE_MID_TS.len());
const _: () = assert!(OUTDOOR_SPLITS_LARGE_MID_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_SPLITS_LARGE_MID: MissionProfile = MissionProfile {
    name: "outdoor_splits-large_mid",
    env: "outdoor",
    variant: "splits-large",
    speed: "mid",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_SPLITS_LARGE_MID_WP,
    timestamps: &OUTDOOR_SPLITS_LARGE_MID_TS,
};

// ─── Outdoor, SplitS-Large, Fast ─────────────────────────────────────
// Source: tmp/planning_results/outdoor-splits-large-fast/race_0716_splits_outdoor_large_waypoints.yaml
// Tail waypoint in the YAML is `[4.996e-16, 1.943e-16, 3]` — i.e. the
// origin to within float epsilon. Rounded to exact zero here.

static OUTDOOR_SPLITS_LARGE_FAST_WP: [[f32; 3]; 63] = [
    [-0.6922, -0.211, 3.035],
    [-4.627, -2.299, 3.238],
    [-7.183, -6.642, 3.505],
    [-5.438, -8.625, 3.964],
    [-2.479, -8.525, 4.282],
    [-0.9852, -6.692, 4.495],
    [-2.316, -3.618, 4.751],
    [-4.648, 0.3268, 4.594],
    [-5.53, 4.926, 4.002],
    [-2.563, 7.932, 3.633],
    [1.386, 8.248, 3.587],
    [6.873, 4.91, 4.005],
    [8.191, 0.5318, 4.856],
    [7.916, -3.247, 5.259],
    [7.642, -6.322, 4.992],
    [7.508, -8.557, 4.298],
    [7.702, -8.734, 3.462],
    [7.641, -6.323, 3.009],
    [6.537, -3.668, 2.884],
    [3.948, -1.118, 2.866],
    [-0.001043, -0.002578, 2.994],
    [-3.802, -1.229, 3.203],
    [-6.36, -3.54, 3.32],
    [-7.185, -6.641, 3.503],
    [-4.084, -8.752, 4.014],
    [-2.207, -8.394, 4.225],
    [-0.9835, -6.693, 4.495],
    [-2.698, -3.215, 4.797],
    [-4.927, 0.6211, 4.585],
    [-5.531, 4.925, 4.002],
    [-2.484, 7.86, 3.63],
    [1.744, 8.135, 3.579],
    [6.875, 4.911, 4.005],
    [8.172, 0.6557, 4.847],
    [7.913, -3.137, 5.27],
    [7.642, -6.323, 4.992],
    [7.54, -8.048, 4.502],
    [7.714, -8.738, 3.508],
    [7.64, -6.323, 3.009],
    [6.482, -3.614, 2.882],
    [3.894, -1.116, 2.866],
    [-0.001138, -0.003241, 2.993],
    [-4.091, -1.319, 3.207],
    [-6.639, -3.783, 3.321],
    [-7.184, -6.641, 3.505],
    [-4.776, -8.59, 3.926],
    [-2.153, -8.305, 4.231],
    [-0.9839, -6.692, 4.495],
    [-2.342, -3.602, 4.8],
    [-4.739, 0.3995, 4.627],
    [-5.532, 4.926, 4.003],
    [-2.46, 7.903, 3.609],
    [1.819, 8.124, 3.57],
    [6.872, 4.911, 4.004],
    [8.168, 0.259, 4.937],
    [7.86, -3.561, 5.295],
    [7.642, -6.322, 4.992],
    [7.631, -8.53, 4.238],
    [7.978, -8.417, 3.354],
    [7.639, -6.324, 3.008],
    [4.656, -2.481, 2.95],
    [0.9551, -0.4156, 2.934],
    [0.0, 0.0, 3.0],
];

static OUTDOOR_SPLITS_LARGE_FAST_TS: [f32; 63] = [
    0.3219, 0.6573, 1.021, 1.324, 1.607, 1.886, 2.181, 2.475, 2.8, 3.108, 3.329, 3.667, 3.978,
    4.273, 4.539, 4.786, 5.062, 5.292, 5.457, 5.644, 5.882, 6.127, 6.344, 6.605, 6.973, 7.155,
    7.419, 7.763, 8.05, 8.359, 8.659, 8.892, 9.213, 9.516, 9.814, 10.09, 10.27, 10.59, 10.84,
    11.01, 11.19, 11.42, 11.69, 11.92, 12.16, 12.46, 12.71, 12.96, 13.26, 13.57, 13.9, 14.21,
    14.44, 14.76, 15.09, 15.4, 15.65, 15.91, 16.2, 16.39, 16.68, 16.97, 17.35,
];

const _: () = assert!(OUTDOOR_SPLITS_LARGE_FAST_WP.len() == OUTDOOR_SPLITS_LARGE_FAST_TS.len());
const _: () = assert!(OUTDOOR_SPLITS_LARGE_FAST_WP.len() <= OFFLINE_MAX_PIECES);

pub static OUTDOOR_SPLITS_LARGE_FAST: MissionProfile = MissionProfile {
    name: "outdoor_splits-large_fast",
    env: "outdoor",
    variant: "splits-large",
    speed: "fast",
    start_pos: [0.0, 0.0, 3.0],
    waypoints: &OUTDOOR_SPLITS_LARGE_FAST_WP,
    timestamps: &OUTDOOR_SPLITS_LARGE_FAST_TS,
};

// ─── Registry ────────────────────────────────────────────────────────

pub static PROFILES: &[&MissionProfile] = &[
    &INDOOR_SPLITS_SLOW,
    &INDOOR_SPLITS_MID,
    &INDOOR_SPLITS_FAST,
    &OUTDOOR_SPLITS_SLOW,
    &OUTDOOR_SPLITS_MID,
    &OUTDOOR_DRAG_SLOW,
    &OUTDOOR_DRAG_MID,
    &OUTDOOR_DRAG_LARGE_MID,
    &OUTDOOR_DRAG_SUPER_MID,
    &OUTDOOR_SPLITS_LARGE_SLOW,
    &OUTDOOR_SPLITS_LARGE_MID,
    &OUTDOOR_SPLITS_LARGE_FAST,
];

// Global compile-time guard: every profile in the registry must fit the
// piece-count ceiling and have matching waypoint/timestamp lengths. The
// per-profile `const _: () = assert!(...)` blocks above already check
// this for in-tree data; this loop is the backstop that catches a future
// profile added to `PROFILES` without its own asserts. If you bump
// `OFFLINE_MAX_PIECES`, you must also bump `MAX_PIECES` in
// `cybflight_core/src/trajectory_planning/mod.rs` to match.
const _: () = {
    let mut i = 0;
    while i < PROFILES.len() {
        let p = PROFILES[i];
        assert!(
            p.waypoints.len() == p.timestamps.len(),
            "MissionProfile waypoint/timestamp length mismatch",
        );
        assert!(
            p.waypoints.len() >= 1,
            "MissionProfile must have at least one waypoint",
        );
        assert!(
            p.waypoints.len() <= OFFLINE_MAX_PIECES,
            "MissionProfile exceeds OFFLINE_MAX_PIECES",
        );
        i += 1;
    }
};

// Default index must point at a real entry in PROFILES.
const _: () = assert!((DEFAULT_PROFILE_INDEX as usize) < PROFILES.len());

/// Default profile applied at boot when flash holds no valid setting, or
/// when a persisted index is incompatible with the current build env.
/// Index into [`PROFILES`]. Must reference a profile whose `env` matches
/// [`BUILD_ENV`] — the build-env split is gated below.
#[cfg(feature = "est_pos_mocap")]
pub const DEFAULT_PROFILE_INDEX: u8 = 0; // indoor_splits_slow
#[cfg(feature = "est_pos_gps")]
pub const DEFAULT_PROFILE_INDEX: u8 = 3; // outdoor_splits_slow

/// Active profile index. Read by `mission_planner::plan_offline()`,
/// written by `init_active_from_index()` (boot) and the shell.
static ACTIVE_PROFILE: AtomicUsize = AtomicUsize::new(DEFAULT_PROFILE_INDEX as usize);

/// Returns the currently active profile.
pub fn active() -> &'static MissionProfile {
    let raw = ACTIVE_PROFILE.load(Ordering::Acquire);
    let idx = if raw < PROFILES.len() {
        raw
    } else {
        DEFAULT_PROFILE_INDEX as usize
    };
    PROFILES[idx]
}

/// Returns the active profile's index in [`PROFILES`].
pub fn active_index() -> u8 {
    let raw = ACTIVE_PROFILE.load(Ordering::Acquire);
    if raw < PROFILES.len() {
        raw as u8
    } else {
        DEFAULT_PROFILE_INDEX
    }
}

/// Sets the active profile by index. Returns `false` if `idx` is
/// out of range or the profile's `env` does not match [`BUILD_ENV`].
/// Selection is left unchanged on rejection.
pub fn set_active(idx: u8) -> bool {
    let i = idx as usize;
    if i >= PROFILES.len() {
        return false;
    }
    if !PROFILES[i].is_compatible_with_build() {
        return false;
    }
    ACTIVE_PROFILE.store(i, Ordering::Release);
    true
}

/// Re-applies a persisted profile index at boot. Falls back to
/// [`DEFAULT_PROFILE_INDEX`] (which is itself env-gated) when the
/// persisted index is out of range OR points at a profile whose `env`
/// does not match [`BUILD_ENV`] — e.g., flash from a prior firmware
/// build that flew the same airframe in the other environment.
pub fn init_active_from_index(idx: u8) {
    let i = idx as usize;
    let resolved = if i < PROFILES.len() && PROFILES[i].is_compatible_with_build() {
        i
    } else {
        DEFAULT_PROFILE_INDEX as usize
    };
    ACTIVE_PROFILE.store(resolved, Ordering::Release);
}

/// Resolves `(env, variant, speed)` to a profile index. Case-insensitive.
/// Pure registry lookup — does not enforce build-env compatibility; the
/// shell layer (and `set_active`) gate that separately so the user gets a
/// distinct "incompatible env" error vs. "unknown mission".
pub fn find(env: &str, variant: &str, speed: &str) -> Option<u8> {
    PROFILES.iter().enumerate().find_map(|(i, p)| {
        (p.env.eq_ignore_ascii_case(env)
            && p.variant.eq_ignore_ascii_case(variant)
            && p.speed.eq_ignore_ascii_case(speed))
        .then_some(i as u8)
    })
}
