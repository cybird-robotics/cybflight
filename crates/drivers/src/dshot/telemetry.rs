//! DShot telemetry value interpretation.
//!
//! Implements eRPM decode (`dshot_decode_eRPM_telemetry_value`, BF `dshot.c:198-213`)
//! and EDT (Extended DShot Telemetry) type dispatch (`dshot.c:215-242`).
//!
//! The pure data types (`TelemetryValue`, `EdtType`, `EdtValue`) live in
//! `cybflight-msgs` so they can be shared with non-HAL crates. Re-exported here
//! for backwards compatibility.

pub use cybflight_msgs::dshot::{EdtType, EdtValue, TelemetryValue};

/// Decode eRPM from a 12-bit raw telemetry value.
///
/// Returns `Some(erpm_x100)` on success, `Some(0)` for stopped motor (0x0FFF),
/// or `None` for invalid/stale (period == 0).
///
/// Convention: the ESC sends raw `thiszctime` — one 60° electrical step in
/// 0.5 µs timer ticks. One electrical revolution = `period * 0.5µs * 6` =
/// `period * 3µs`, so `eRPM = 60_000_000 / (period * 3) = 20_000_000 / period`.
///
/// `period == 0` means the ESC had no fresh commutation event since the last
/// DShot frame (stale). Treated identically to a decode failure (returns None),
/// which causes the RPM estimator to coast on its model prediction.
pub const fn decode_erpm(raw: u16) -> Option<u32> {
    if raw == 0x0FFF {
        return Some(0);
    }

    // eeem_mmmm_mmmm: mantissa = lower 9 bits, exponent = upper 3 bits
    let mantissa = (raw & 0x1FF) as u32;
    let exponent = ((raw >> 9) & 0x7) as u32;
    let period = mantissa << exponent;

    if period == 0 {
        return None;
    }

    // erpm_x100 = 20_000_000 / 100 / period = 200_000 / period (rounded)
    Some((200_000 + period / 2) / period)
}

/// Interpret a raw 12-bit telemetry value as eRPM or EDT.
pub const fn interpret(raw: u16, edt_enabled: bool) -> TelemetryValue {
    // type_field is bits [11:8] of the 12-bit value
    let type_field = ((raw >> 8) & 0xF) as u8;

    // eRPM if: EDT disabled, or type_field is odd, or type_field == 0
    let is_erpm = !edt_enabled || (type_field & 0x01 != 0) || type_field == 0;

    if is_erpm {
        match decode_erpm(raw) {
            Some(0) => TelemetryValue::Stopped,
            Some(erpm) => TelemetryValue::Erpm(erpm),
            None => TelemetryValue::Invalid,
        }
    } else {
        // EDT: type_index = type_field >> 1, data = lower 8 bits
        let type_index = type_field >> 1;
        let data = (raw & 0xFF) as u8;
        let edt_type = match type_index {
            1 => EdtType::Temperature,
            2 => EdtType::Voltage,
            3 => EdtType::Current,
            4 => EdtType::Debug1,
            5 => EdtType::Debug2,
            6 => EdtType::Debug3,
            7 => EdtType::StateEvents,
            // type_index 0 shouldn't reach here (type_field==0 is handled as eRPM),
            // but handle defensively
            _ => EdtType::Debug1,
        };
        TelemetryValue::Edt(EdtValue { edt_type, data })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erpm_stopped() {
        assert_eq!(decode_erpm(0x0FFF), Some(0));
        assert_eq!(interpret(0x0FFF, false), TelemetryValue::Stopped);
        assert_eq!(interpret(0x0FFF, true), TelemetryValue::Stopped);
    }

    #[test]
    fn erpm_invalid() {
        // raw=0x0000: mantissa=0, exponent=0 → period=0 → invalid
        assert_eq!(decode_erpm(0x0000), None);
        assert_eq!(interpret(0x0000, false), TelemetryValue::Invalid);
    }

    #[test]
    fn erpm_value_1() {
        // raw=0x0001: mantissa=1, exp=0 → period=1 → erpm=(200000+0)/1=200000
        assert_eq!(decode_erpm(0x0001), Some(200_000));
        assert_eq!(interpret(0x0001, false), TelemetryValue::Erpm(200_000));
    }

    #[test]
    fn erpm_value_100() {
        // raw=0x0064: mantissa=0x64=100, exp=0 → period=100 → erpm=(200000+50)/100=2000
        assert_eq!(decode_erpm(0x0064), Some(2_000));
        assert_eq!(interpret(0x0064, false), TelemetryValue::Erpm(2_000));
    }

    #[test]
    fn edt_temperature() {
        // raw=0x0254: type_field = (0x254 >> 8) & 0xF = 2 (even, non-zero)
        // type_index = 2 >> 1 = 1 → Temperature, data = 0x54
        assert_eq!(
            interpret(0x0254, true),
            TelemetryValue::Edt(EdtValue {
                edt_type: EdtType::Temperature,
                data: 0x54
            })
        );
    }

    #[test]
    fn edt_voltage() {
        // raw=0x0480: type_field = 4 (even, non-zero)
        // type_index = 4 >> 1 = 2 → Voltage, data = 0x80
        assert_eq!(
            interpret(0x0480, true),
            TelemetryValue::Edt(EdtValue {
                edt_type: EdtType::Voltage,
                data: 0x80
            })
        );
    }

    #[test]
    fn edt_disabled_returns_erpm() {
        // Same raw value but EDT disabled → treat as eRPM
        // raw=0x0064 with edt_enabled=false → eRPM(2000)
        assert_eq!(interpret(0x0064, false), TelemetryValue::Erpm(2_000));
    }

    #[test]
    fn edt_odd_type_field_is_erpm() {
        // type_field=1 (odd) → always eRPM even with EDT enabled
        // raw=0x0164: mantissa=0x164&0x1FF=0x164=356, exp=0 → period=356
        // erpm = (200000+178)/356 = 562
        assert_eq!(interpret(0x0164, true), TelemetryValue::Erpm(562));
    }

    #[test]
    fn edt_current() {
        // raw=0x06FF: type_field=6, type_index=3 → Current, data=0xFF
        assert_eq!(
            interpret(0x06FF, true),
            TelemetryValue::Edt(EdtValue {
                edt_type: EdtType::Current,
                data: 0xFF
            })
        );
    }

    #[test]
    fn edt_state_events() {
        // raw=0x0E42: type_field=0xE=14, type_index=7 → StateEvents, data=0x42
        assert_eq!(
            interpret(0x0E42, true),
            TelemetryValue::Edt(EdtValue {
                edt_type: EdtType::StateEvents,
                data: 0x42
            })
        );
    }
}
