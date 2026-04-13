#[allow(unused_imports)]
use num_traits::Float;

/// 3D vector stored as [x, y, z]. No heap allocation.
pub type Vec3 = [f32; 3];

/// 3D state: [pos, vel, acc, jerk] — 4 columns of Vec3.
/// Indexed as pvaj[col][dim]: pvaj[0] = pos, pvaj[1] = vel, etc.
pub type PVAJ3D = [[f32; 3]; 4];

/// 3D state: [pos, vel, acc] — 3 columns of Vec3.
pub type PVA3D = [[f32; 3]; 3];

/// Zero vector.
pub const ZERO3: Vec3 = [0.0, 0.0, 0.0];

/// Fused multiply-add for 3D vectors: a * s + b
#[inline(always)]
pub fn fma3(a: Vec3, s: f32, b: Vec3) -> Vec3 {
    [
        a[0].mul_add(s, b[0]),
        a[1].mul_add(s, b[1]),
        a[2].mul_add(s, b[2]),
    ]
}

/// Scale a 3D vector: a * s
#[inline(always)]
pub fn scale3(a: Vec3, s: f32) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

/// Dot product of two 3D vectors.
#[inline(always)]
pub fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

/// Squared norm of a 3D vector.
#[inline(always)]
pub fn norm_sq3(a: Vec3) -> f32 {
    dot3(a, a)
}

/// Add two 3D vectors.
#[inline(always)]
pub fn add3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Subtract two 3D vectors: a - b
#[inline(always)]
pub fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
