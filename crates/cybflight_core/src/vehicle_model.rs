use nalgebra as na;
use num_traits::{One, Zero};

pub trait VehicleModel {
    type Scalar: na::RealField + Copy + Zero + One;

    /// Mass in kilograms.
    fn mass(&self) -> Self::Scalar;

    fn inertia_ixx(&self) -> Self::Scalar;

    fn inertia_iyy(&self) -> Self::Scalar;

    fn inertia_izz(&self) -> Self::Scalar;

    fn inertia_ixy(&self) -> Self::Scalar {
        Self::Scalar::zero()
    }

    fn inertia_ixz(&self) -> Self::Scalar {
        Self::Scalar::zero()
    }

    fn inertia_iyz(&self) -> Self::Scalar {
        Self::Scalar::zero()
    }

    fn front_motor_position(&self) -> na::Vector2<Self::Scalar>;

    fn rear_motor_position(&self) -> na::Vector2<Self::Scalar>;

    fn torque_constant(&self) -> Self::Scalar;

    fn motor_time_constant_up(&self) -> Self::Scalar;

    fn motor_time_constant_down(&self) -> Self::Scalar {
        self.motor_time_constant_up()
    }

    /// Maximum thrust a single motor can produce (Newtons).
    fn max_thrust_per_motor(&self) -> Self::Scalar;

    fn inertia(&self) -> na::Matrix3<Self::Scalar> {
        na::Matrix3::new(
            self.inertia_ixx(),
            self.inertia_ixy(),
            self.inertia_ixz(),
            self.inertia_ixy(),
            self.inertia_iyy(),
            self.inertia_iyz(),
            self.inertia_ixz(),
            self.inertia_iyz(),
            self.inertia_izz(),
        )
    }

    fn allocation_matrix(&self) -> na::Matrix4<Self::Scalar> {
        let [fx, fy] = self.front_motor_position().into();
        let [bx, by] = self.rear_motor_position().into();
        let c = self.torque_constant();

        na::Matrix4::new(
            Self::Scalar::one(),
            Self::Scalar::one(),
            Self::Scalar::one(),
            Self::Scalar::one(), //
            -by,
            -fy,
            by,
            fy, //
            bx,
            -fx,
            bx,
            -fx, //
            c,
            -c,
            -c,
            c,
        )
    }

    /// Implementers of this trait are strongly encouraged to override this method with a
    /// precomputed inverse of the allocation matrix
    fn inv_allocation_matrix(&self) -> na::Matrix4<Self::Scalar> {
        self.allocation_matrix().try_inverse().unwrap()
    }
}

pub struct ThrustTorque<T> {
    pub collective_thrust_n: T,
    pub torque_n_m: na::Vector3<T>,
}

impl<T> From<(T, na::Vector3<T>)> for ThrustTorque<T> {
    fn from((collective_thrust_n, torque_n_m): (T, na::Vector3<T>)) -> Self {
        Self {
            collective_thrust_n,
            torque_n_m,
        }
    }
}

pub fn thrust_torque_to_motor_thrusts<T: na::RealField + Copy, Mdl: VehicleModel<Scalar = T>>(
    thrust_torque: &ThrustTorque<T>,
    mdl: &Mdl,
) -> na::Vector4<T> {
    let thrust_torque_vec = na::Vector4::new(
        thrust_torque.collective_thrust_n,
        thrust_torque.torque_n_m.x,
        thrust_torque.torque_n_m.y,
        thrust_torque.torque_n_m.z,
    );
    thrust_torque_vector_to_motor_thrusts(&thrust_torque_vec, mdl)
}

/// Convert per-motor thrust (Newtons) to normalized throttle [0, 1].
///
/// This is a linear stub: `throttle = thrust / max_thrust`.
/// A real implementation would use a polynomial or lookup table calibrated
/// from motor test stand data.
pub fn motor_thrust_to_throttle<T: na::RealField + Copy>(
    thrust_n: T,
    max_thrust_n: T,
) -> T {
    let zero = T::zero();
    let one = T::one();
    if thrust_n <= zero {
        return zero;
    }
    let t = thrust_n / max_thrust_n;
    if t > one { one } else { t }
}

pub fn thrust_torque_vector_to_motor_thrusts<
    T: na::RealField + Copy,
    Mdl: VehicleModel<Scalar = T>,
>(
    thrust_torque_vec: &na::Vector4<T>,
    mdl: &Mdl,
) -> na::Vector4<T> {
    mdl.inv_allocation_matrix() * thrust_torque_vec
}
