//! Short-range LLH → ENU conversion for GNSS-backed position estimators.
//!
//! The firmware ESKF operates in a local ENU frame anchored at an origin
//! set once per flight (first good GPS fix). A full ECEF round-trip is
//! overkill at quadrotor ranges: a flat-Earth tangent plane with Rm/Rn
//! evaluated at the origin is accurate to well under a metre inside a
//! ~10 km circle, which is two orders of magnitude beyond anything we
//! care about.
//!
//! Input lat/lon are kept in f64 (otherwise the 1e-7° resolution of
//! UBX-NAV-PVT is lost to float32 cancellation at typical mid-latitudes);
//! outputs are f32 to match the ESKF interface.

use nalgebra::Vector3;

/// WGS-84 semi-major axis [m].
const WGS84_A: f64 = 6_378_137.0;
/// WGS-84 first-eccentricity squared.
const WGS84_E2: f64 = 6.694_379_990_141_316e-3;

/// Local tangent-plane origin with precomputed Earth radii at `lat0`.
#[derive(Clone, Copy, Debug)]
pub struct LlhOrigin {
    pub lat_rad: f64,
    pub lon_rad: f64,
    pub h_m: f32,
    /// Meridional radius of curvature at `lat0` [m].
    rm: f64,
    /// Normal radius × cos(lat0) — the scale factor for Δlon → east [m].
    rn_cos_lat: f64,
}

impl LlhOrigin {
    /// Build an origin, precomputing the two radii once.
    pub fn new(lat_rad: f64, lon_rad: f64, h_m: f32) -> Self {
        let sin_lat = libm::sin(lat_rad);
        let cos_lat = libm::cos(lat_rad);
        let denom = 1.0 - WGS84_E2 * sin_lat * sin_lat;
        let rn = WGS84_A / libm::sqrt(denom);
        let rm = WGS84_A * (1.0 - WGS84_E2) / (denom * libm::sqrt(denom));
        Self {
            lat_rad,
            lon_rad,
            h_m,
            rm,
            rn_cos_lat: rn * cos_lat,
        }
    }

    /// Convert a point `(lat, lon, h)` to ENU metres relative to self.
    /// `(east, north, up)` in the usual right-handed frame.
    pub fn llh_to_enu(&self, lat_rad: f64, lon_rad: f64, h_m: f32) -> Vector3<f32> {
        let dlat = lat_rad - self.lat_rad;
        let dlon = lon_rad - self.lon_rad;
        let east = self.rn_cos_lat * dlon;
        let north = self.rm * dlat;
        let up = h_m - self.h_m;
        Vector3::new(east as f32, north as f32, up)
    }

    /// Convert a local ENU point (metres, relative to this origin) back to
    /// geodetic `(lat_rad, lon_rad, h_m)`. Inverse of [`llh_to_enu`] under the
    /// same flat-Earth tangent-plane approximation — used to re-express a drone's
    /// fused ENU position as an absolute lat/lon/alt for cross-drone sharing.
    pub fn enu_to_llh(&self, enu: Vector3<f32>) -> (f64, f64, f32) {
        let lat_rad = self.lat_rad + (enu.y as f64) / self.rm;
        let lon_rad = self.lon_rad + (enu.x as f64) / self.rn_cos_lat;
        let h_m = enu.z + self.h_m;
        (lat_rad, lon_rad, h_m)
    }
}

/// NED → ENU: swap x/y, flip z. UBX-NAV-PVT reports velocity in NED; the
/// ESKF consumes ENU.
pub fn ned_to_enu(v_ned: Vector3<f32>) -> Vector3<f32> {
    Vector3::new(v_ned.y, v_ned.x, -v_ned.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    const ZURICH_LAT: f64 = 47.376886;
    const ZURICH_LON: f64 = 8.541694;

    fn origin() -> LlhOrigin {
        LlhOrigin::new(ZURICH_LAT * PI / 180.0, ZURICH_LON * PI / 180.0, 400.0)
    }

    #[test]
    fn identity_is_origin() {
        let o = origin();
        let p = o.llh_to_enu(o.lat_rad, o.lon_rad, o.h_m);
        assert!(p.norm() < 1e-5, "expected zero at origin, got {p:?}");
    }

    #[test]
    fn one_arcsec_north_is_about_30m() {
        // One arc-second of latitude ≈ Rm · (π / 648000) rad.
        // At Zürich's latitude that's ≈ 30.88 m on the meridional circle.
        let o = origin();
        let one_arcsec_rad = PI / (180.0 * 3600.0);
        let p = o.llh_to_enu(o.lat_rad + one_arcsec_rad, o.lon_rad, o.h_m);
        assert!(p.x.abs() < 1e-3, "east drift: {}", p.x);
        assert!(
            (p.y - 30.88).abs() < 0.05,
            "expected ≈30.88 m north, got {} m",
            p.y
        );
        assert!(p.z.abs() < 1e-4);
    }

    #[test]
    fn one_arcsec_east_scales_with_cos_lat() {
        // Δeast = Rn · cos(lat0) · Δlon ≈ 20.98 m at Zürich.
        let o = origin();
        let one_arcsec_rad = PI / (180.0 * 3600.0);
        let p = o.llh_to_enu(o.lat_rad, o.lon_rad + one_arcsec_rad, o.h_m);
        assert!(p.y.abs() < 1e-3, "north drift: {}", p.y);
        assert!(
            (p.x - 20.98).abs() < 0.05,
            "expected ≈20.98 m east, got {} m",
            p.x
        );
    }

    #[test]
    fn altitude_passes_through() {
        let o = origin();
        let p = o.llh_to_enu(o.lat_rad, o.lon_rad, o.h_m + 17.5);
        assert!(p.x.abs() < 1e-4 && p.y.abs() < 1e-4);
        assert!((p.z - 17.5).abs() < 1e-4);
    }

    #[test]
    fn ned_to_enu_swap_and_flip() {
        let v = ned_to_enu(Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(v, Vector3::new(2.0, 1.0, -3.0));
    }

    #[test]
    fn enu_to_llh_inverts_llh_to_enu() {
        let o = origin();
        let enu = Vector3::new(120.0, 80.0, 12.0); // ~120 m E, 80 m N, 12 m up
        let (lat, lon, h) = o.enu_to_llh(enu);
        let back = o.llh_to_enu(lat, lon, h);
        assert!((back - enu).norm() < 1e-2, "round-trip drift {:?}", back - enu);
    }
}
