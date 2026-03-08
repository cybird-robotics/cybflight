use core::default::Default;
use core::marker::Copy;
use nalgebra as na;

use crate::rotation::hat;

pub trait Manifold {
    type Tangent;
    fn boxplus(&self, delta: &Self::Tangent) -> Self;
    fn boxminus(&self, other: &Self) -> Self::Tangent;
}

pub struct NominalState<T> {
    position: na::Vector3<T>,           // position
    orientation: na::UnitQuaternion<T>, // attitude
    velocity: na::Vector3<T>,           // velocity
    accel_bias: na::Vector3<T>,         // accelerometer bias
    gyro_bias: na::Vector3<T>,          // gyroscope bias
    grav_vector: na::Vector3<T>,        // gravity vector in body frame
}

impl Default for NominalState<f32> {
    fn default() -> Self {
        Self {
            position: na::Vector3::zeros(),
            orientation: na::UnitQuaternion::identity(),
            velocity: na::Vector3::zeros(),
            accel_bias: na::Vector3::zeros(),
            gyro_bias: na::Vector3::zeros(),
            grav_vector: na::Vector3::new(0.0, 0.0, -9.81),
        }
    }
}

impl<T: na::RealField + Copy> Manifold for NominalState<T> {
    type Tangent = na::SVector<T, 18>;
    fn boxplus(&self, delta: &Self::Tangent) -> Self {
        Self {
            position: self.position + delta.fixed_rows::<3>(0),
            orientation: self.orientation
                * na::UnitQuaternion::from_scaled_axis(delta.fixed_rows::<3>(3)),
            velocity: self.velocity + delta.fixed_rows::<3>(6),
            accel_bias: self.accel_bias + delta.fixed_rows::<3>(9),
            gyro_bias: self.gyro_bias + delta.fixed_rows::<3>(12),
            grav_vector: self.grav_vector + delta.fixed_rows::<3>(15),
        }
    }

    fn boxminus(&self, other: &Self) -> Self::Tangent {
        let mut delta = Self::Tangent::zeros();
        delta.fixed_rows_mut::<3>(0).copy_from(&(self.position - other.position));
        let q_err = other.orientation.inverse() * self.orientation;
        delta.fixed_rows_mut::<3>(3).copy_from(&q_err.scaled_axis());
        delta.fixed_rows_mut::<3>(6).copy_from(&(self.velocity - other.velocity));
        delta.fixed_rows_mut::<3>(9).copy_from(&(self.accel_bias - other.accel_bias));
        delta.fixed_rows_mut::<3>(12).copy_from(&(self.gyro_bias - other.gyro_bias));
        delta.fixed_rows_mut::<3>(15).copy_from(&(self.grav_vector - other.grav_vector));
        delta
    }
}

pub struct ImuInput<T> {
    accel: na::Vector3<T>,
    gyro: na::Vector3<T>,
}

pub struct ImuNoiseConfig<T> {
    accel_noise_density: T,
    gyro_noise_density: T,
    accel_bias_random_walk: T,
    gyro_bias_random_walk: T,
}

pub fn motion_model<T>(
    state: &NominalState<T>,
    input: &ImuInput<T>,
    dt: T,
    _cfg: &ImuNoiseConfig<T>,
) -> NominalState<T>
where
    T: na::RealField + Copy,
{
    let NominalState {
        position: p,
        orientation: q,
        velocity: v,
        accel_bias,
        gyro_bias,
        grav_vector,
    } = state;
    let ImuInput { accel, gyro } = input;
    let acc_unbiased = accel - accel_bias;
    let accel_world = q * acc_unbiased + grav_vector;
    let delta_velocity = accel_world * dt;
    let gyro_unbiased = gyro - gyro_bias;
    let delta_angle = gyro_unbiased * dt;

    // Simple discrete-time integration (Euler method)
    NominalState {
        position: p + v * dt,
        orientation: q * na::UnitQuaternion::from_scaled_axis(delta_angle),
        velocity: v + delta_velocity,
        accel_bias: *accel_bias,
        gyro_bias: *gyro_bias,
        grav_vector: *grav_vector,
    }
}

pub struct Jacobians<T> {
    pub fjac: na::SMatrix<T, 18, 18>,
    pub gjac: na::SMatrix<T, 18, 18>,
}

fn motion_jacobians<T>(
    state: &NominalState<T>,
    input: &ImuInput<T>,
    dt: T,
    cfg: &ImuNoiseConfig<T>,
) -> Jacobians<T>
where
    T: na::RealField + Copy,
{
    let NominalState {
        position: _p,
        orientation: q,
        velocity: _v,
        accel_bias,
        gyro_bias,
        grav_vector: _,
    } = state;
    let ImuInput { accel, gyro } = input;
    let acc_unbiased = accel - accel_bias;
    let gyro_unbiased = gyro - gyro_bias;
    let delta_angle = gyro_unbiased * dt;

    // F =
    //   [I, O,            dt * I, O,     O    , O   ;
    //    O, R(-dt*ω),     O,      O,     -dt*I, O   ;
    //    O, -dt*R*hat(a), I,      -dt*R, O    , dt*I;
    //    O, O,            O,      I,     O    , O   ;
    //    O, O,            O,      O,     I    , O   ;
    //    O, O,            O,      O,     O    , I   ];
    let mut fjac: na::SMatrix<T, 18, 18> = na::Matrix::zeros();
    // Position derivatives
    fjac.fixed_view_mut::<3, 3>(0, 0).copy_from(&na::Matrix::identity());
    fjac.fixed_view_mut::<3, 3>(0, 6).copy_from(&(na::Matrix::identity() * dt));

    // Orientation derivatives
    fjac.fixed_view_mut::<3, 3>(3, 3).copy_from(
        na::UnitQuaternion::from_scaled_axis(-delta_angle).to_rotation_matrix().matrix(),
    );
    fjac.fixed_view_mut::<3, 3>(3, 12).copy_from(&(-na::Matrix::identity() * dt));

    // Velocity derivatives
    let rmat: na::Matrix3<T> = *q.to_rotation_matrix().matrix();
    fjac.fixed_view_mut::<3, 3>(6, 3).copy_from(&(-rmat * hat(&acc_unbiased) * dt));
    fjac.fixed_view_mut::<3, 3>(6, 6).copy_from(&na::Matrix::identity());
    fjac.fixed_view_mut::<3, 3>(6, 9).copy_from(&(-rmat * dt));
    fjac.fixed_view_mut::<3, 3>(6, 15).copy_from(&(na::Matrix::identity() * dt));

    // Accel bias derivatives
    fjac.fixed_view_mut::<3, 3>(9, 9).copy_from(&na::Matrix::identity());

    // Gyro bias derivatives
    fjac.fixed_view_mut::<3, 3>(12, 12).copy_from(&na::Matrix::identity());

    // Gravity vector derivatives
    fjac.fixed_view_mut::<3, 3>(15, 15).copy_from(&na::Matrix::identity());

    // G = blkdiag(O, ...
    //             σ_gn * dt.^2 * I, ...
    //             σ_an * dt.^2 * I, ...
    //             σ_ab * dt * I,    ...
    //             σ_gb * dt * I,    ...
    //             O);
    let dt_sq = dt * dt;
    let ImuNoiseConfig {
        accel_noise_density,
        gyro_noise_density,
        accel_bias_random_walk,
        gyro_bias_random_walk,
    } = cfg;

    let mut gjac = na::SMatrix::<T, 18, 18>::identity();
    gjac.fixed_view_mut::<3, 3>(3, 3).fill_diagonal(*gyro_noise_density * dt_sq);
    gjac.fixed_view_mut::<3, 3>(6, 6).fill_diagonal(*accel_noise_density * dt_sq);
    gjac.fixed_view_mut::<3, 3>(9, 9).fill_diagonal(*accel_bias_random_walk * dt);
    gjac.fixed_view_mut::<3, 3>(12, 12).fill_diagonal(*gyro_bias_random_walk * dt);

    Jacobians { fjac, gjac }
}

struct Pose<T> {
    pub position: na::Vector3<T>,
    pub orientation: na::UnitQuaternion<T>,
}

impl<T: na::RealField> Default for Pose<T> {
    fn default() -> Self {
        Self {
            position: na::Vector3::zeros(),
            orientation: na::UnitQuaternion::identity(),
        }
    }
}

impl<T: na::RealField + Copy> Manifold for Pose<T> {
    type Tangent = na::SVector<T, 6>;
    fn boxplus(&self, delta: &Self::Tangent) -> Self {
        Self {
            position: self.position + delta.fixed_rows::<3>(0),
            orientation: self.orientation
                * na::UnitQuaternion::from_scaled_axis(delta.fixed_rows::<3>(3)),
        }
    }

    fn boxminus(&self, other: &Self) -> Self::Tangent {
        let mut delta = Self::Tangent::zeros();
        delta.fixed_rows_mut::<3>(0).copy_from(&(self.position - other.position));
        let q_err = other.orientation.inverse() * self.orientation;
        delta.fixed_rows_mut::<3>(3).copy_from(&q_err.scaled_axis());
        delta
    }
}

pub fn pose_observation<T: na::RealField + Copy>(state: &NominalState<T>) -> Pose<T> {
    Pose {
        position: state.position,
        orientation: state.orientation,
    }
}

pub fn pose_observation_jacobian<T: na::RealField + Copy>() -> na::SMatrix<T, 6, 18> {
    let mut jac = na::SMatrix::<T, 6, 18>::zeros();
    jac.fixed_view_mut::<6, 6>(0, 0).copy_from(&na::Matrix::identity());
    jac
}
