use super::types::{Vec3, ZERO3};

/// Maximum polynomial degree supported (snap = degree 7).
pub const MAX_COEFFS: usize = 8;

/// A single polynomial piece of degree up to 7 in 3D.
///
/// Coefficients stored in **natural (ascending) order**:
///   coeffs[0] = b₀ (constant), coeffs[1] = b₁, ..., coeffs[degree] = b_deg
///
/// So p(t) = b₀ + b₁t + b₂t² + ... + b_deg·t^deg
#[derive(Debug, Clone, Copy)]
pub struct Polynomial {
    pub degree: usize,
    pub duration: f32,
    /// Coefficients in ascending power order: coeffs[k] = coefficient for t^k.
    /// Only coeffs[0..=degree] are valid. Remaining entries are zero.
    pub coeffs: [Vec3; MAX_COEFFS],
}

impl Polynomial {
    /// Create a polynomial from coefficients in ascending order [b0, b1, ..., b_deg].
    #[inline]
    pub fn new(degree: usize, duration: f32, coeffs: &[Vec3]) -> Self {
        debug_assert!(degree + 1 == coeffs.len());
        debug_assert!(degree < MAX_COEFFS);
        let mut c = [ZERO3; MAX_COEFFS];
        c[..coeffs.len()].copy_from_slice(coeffs);
        Self {
            degree,
            duration,
            coeffs: c,
        }
    }

    /// Create a polynomial from coefficients given in **descending** order
    /// (highest degree first), matching the C++ convention.
    #[inline]
    pub fn from_descending(degree: usize, duration: f32, desc_coeffs: &[Vec3]) -> Self {
        debug_assert!(degree + 1 == desc_coeffs.len());
        debug_assert!(degree < MAX_COEFFS);
        let mut c = [ZERO3; MAX_COEFFS];
        for (i, coeff) in desc_coeffs.iter().enumerate() {
            c[degree - i] = *coeff;
        }
        Self {
            degree,
            duration,
            coeffs: c,
        }
    }

    /// Evaluate position using Horner's method.
    #[inline(always)]
    pub fn get_pos(&self, t: f32) -> Vec3 {
        let n = self.degree;
        let c = &self.coeffs;
        let mut r = c[n];
        let mut i = n;
        while i > 0 {
            i -= 1;
            r = r * t + c[i];
        }
        r
    }

    #[inline(always)]
    pub fn get_vel(&self, t: f32) -> Vec3 {
        let n = self.degree;
        if n == 0 {
            return ZERO3;
        }
        let c = &self.coeffs;
        let mut r = c[n] * (n as f32);
        let mut i = n;
        while i > 1 {
            i -= 1;
            r = r * t + c[i] * (i as f32);
        }
        r
    }

    #[inline(always)]
    pub fn get_acc(&self, t: f32) -> Vec3 {
        let n = self.degree;
        if n < 2 {
            return ZERO3;
        }
        let c = &self.coeffs;
        let mut r = c[n] * ((n * (n - 1)) as f32);
        let mut i = n;
        while i > 2 {
            i -= 1;
            r = r * t + c[i] * ((i * (i - 1)) as f32);
        }
        r
    }

    #[inline(always)]
    pub fn get_jerk(&self, t: f32) -> Vec3 {
        let n = self.degree;
        if n < 3 {
            return ZERO3;
        }
        let c = &self.coeffs;
        let mut r = c[n] * ((n * (n - 1) * (n - 2)) as f32);
        let mut i = n;
        while i > 3 {
            i -= 1;
            r = r * t + c[i] * ((i * (i - 1) * (i - 2)) as f32);
        }
        r
    }

    #[inline(always)]
    pub fn get_snap(&self, t: f32) -> Vec3 {
        let n = self.degree;
        if n < 4 {
            return ZERO3;
        }
        let c = &self.coeffs;
        let mut r = c[n] * ((n * (n - 1) * (n - 2) * (n - 3)) as f32);
        let mut i = n;
        while i > 4 {
            i -= 1;
            r = r * t + c[i] * ((i * (i - 1) * (i - 2) * (i - 3)) as f32);
        }
        r
    }

    #[inline(always)]
    pub fn get_crackle(&self, t: f32) -> Vec3 {
        let n = self.degree;
        if n < 5 {
            return ZERO3;
        }
        let c = &self.coeffs;
        let mut r = c[n] * ((n * (n - 1) * (n - 2) * (n - 3) * (n - 4)) as f32);
        let mut i = n;
        while i > 5 {
            i -= 1;
            r = r * t + c[i] * ((i * (i - 1) * (i - 2) * (i - 3) * (i - 4)) as f32);
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn test_constant() {
        let poly = Polynomial::new(0, 1.0, &[Vector3::new(1.0, 2.0, 3.0)]);
        let pos = poly.get_pos(0.5);
        assert!((pos[0] - 1.0).abs() < 1e-12);
        assert!((pos[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear() {
        let poly = Polynomial::new(
            1,
            2.0,
            &[Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 2.0, 3.0)],
        );
        let pos = poly.get_pos(1.0);
        assert!((pos[0] - 1.0).abs() < 1e-12);
        assert!((pos[1] - 2.0).abs() < 1e-12);

        let vel = poly.get_vel(1.0);
        assert!((vel[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_quadratic_horner() {
        let poly = Polynomial::new(
            2,
            2.0,
            &[
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::zeros(),
                Vector3::new(1.0, 0.0, 0.0),
            ],
        );
        let pos = poly.get_pos(2.0);
        assert!((pos[0] - 5.0).abs() < 1e-12);

        let vel = poly.get_vel(2.0);
        assert!((vel[0] - 4.0).abs() < 1e-12);

        let acc = poly.get_acc(1.0);
        assert!((acc[0] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_degree7_snap() {
        let mut c = [Vector3::<f32>::zeros(); 8];
        c[7] = Vector3::new(1.0, 0.0, 0.0);
        let poly = Polynomial::new(7, 2.0, &c);

        let s = poly.get_snap(1.0);
        assert!((s[0] - 840.0).abs() < 1e-10);

        let s2 = poly.get_snap(2.0);
        assert!((s2[0] - 840.0 * 8.0).abs() < 1e-8);
    }

    #[test]
    fn test_from_descending() {
        let poly = Polynomial::from_descending(
            2,
            2.0,
            &[
                Vector3::new(1.0, 0.0, 0.0),
                Vector3::zeros(),
                Vector3::new(1.0, 0.0, 0.0),
            ],
        );
        let pos = poly.get_pos(2.0);
        assert!((pos[0] - 5.0).abs() < 1e-12);
    }
}
