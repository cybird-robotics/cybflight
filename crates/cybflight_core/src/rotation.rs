// Deeper algebraic rotation utilities not even covered by nalgebra

use core::marker::Copy;
use nalgebra::{Matrix, Matrix3, Matrix4, RealField, Storage, UnitQuaternion, Vector3, U1, U3};

pub fn hat<T, S>(v: &Matrix<T, U3, U1, S>) -> Matrix3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U1>,
{
    Matrix3::new(
        T::zero(),
        -v[2],
        v[1],
        v[2],
        T::zero(),
        -v[0],
        -v[1],
        v[0],
        T::zero(),
    )
}

pub fn vee<T, S>(m: &Matrix<T, U3, U3, S>) -> Vector3<T>
where
    T: RealField + Copy,
    S: Storage<T, U3, U3>,
{
    Vector3::new(m[(2, 1)], m[(0, 2)], m[(1, 0)])
}

pub fn left_quaternion_matrix<T: RealField + Copy>(q: &UnitQuaternion<T>) -> Matrix4<T> {
    let mut m = Matrix4::zeros();

    let top_left = hat(&q.vector());
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&top_left);
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&q.vector());
    m.fixed_view_mut::<1, 3>(3, 0)
        .copy_from(&-q.vector().transpose());
    m.diagonal().fill(q.scalar());
    m
}

pub fn right_quaternion_matrix<T: RealField + Copy>(q: &UnitQuaternion<T>) -> Matrix4<T> {
    let mut m = Matrix4::zeros();

    let top_left = -hat(&q.vector());
    m.fixed_view_mut::<3, 3>(0, 0).copy_from(&top_left);
    m.fixed_view_mut::<3, 1>(0, 3).copy_from(&q.vector());
    m.fixed_view_mut::<1, 3>(3, 0)
        .copy_from(&-q.vector().transpose());
    m.diagonal().fill(q.scalar());
    m.transpose()
}
