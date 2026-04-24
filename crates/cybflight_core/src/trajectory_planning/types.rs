use nalgebra::Vector3;

/// 3D vector. Alias for `nalgebra::Vector3<f32>`.
///
/// Layout: `#[repr(C)]` over `[f32; 3]`. Constructed via `Vector3::new(x, y, z)`
/// or `Vector3::from([x, y, z])`. Component access via `.x/.y/.z` or `[i]`.
pub type Vec3 = Vector3<f32>;

/// 3D state: [pos, vel, acc, jerk] — 4 columns of Vec3.
pub type PVAJ3D = [Vec3; 4];

/// 3D state: [pos, vel, acc] — 3 columns of Vec3.
pub type PVA3D = [Vec3; 3];

/// Zero vector. `nalgebra::Vector3::new` is `const` on 0.34, so this is a true
/// const initializer and can be used in array literals like `[ZERO3; N]`.
pub const ZERO3: Vec3 = Vector3::new(0.0, 0.0, 0.0);
