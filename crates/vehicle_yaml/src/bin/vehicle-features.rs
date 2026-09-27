//! Resolve firmware features from vehicle YAML, with explicit environment overrides.

use std::{env, fs, path::Path, process::ExitCode};

use serde::Deserialize;
use vehicle_yaml::BuildYaml;

#[derive(Default, Deserialize)]
struct AirframeName {
    name: Option<String>,
}

// Only build metadata is needed here. The firmware build validates the complete vehicle.
#[derive(Deserialize)]
struct VehicleMetadata {
    #[serde(default)]
    build: BuildYaml,
    #[serde(default)]
    airframe: AirframeName,
}

fn vehicles(root: &Path) -> Result<Vec<String>, String> {
    let entries = fs::read_dir(root.join("vehicles")).map_err(|e| e.to_string())?;
    let mut names = Vec::new();
    for entry in entries {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("yaml") {
            names.push(path.file_stem().unwrap().to_string_lossy().into_owned());
        }
    }
    names.sort();
    Ok(names)
}

fn read_vehicle(root: &Path, vehicle: &str) -> Result<VehicleMetadata, String> {
    let path = root.join("vehicles").join(format!("{vehicle}.yaml"));
    let contents = fs::read_to_string(&path).map_err(|e| {
        format!(
            "cannot read {}: {e}\navailable vehicles: {}\nrun `just vehicles` for details",
            path.display(),
            vehicles(root).unwrap_or_default().join(", ")
        )
    })?;
    serde_yaml::from_str(&contents).map_err(|e| format!("{}: {e}", path.display()))
}

fn resolve(
    mut build: BuildYaml,
    vehicle: &str,
    get_env: impl Fn(&str) -> Option<String>,
    explain: bool,
) -> Result<String, String> {
    macro_rules! knob {
        ($field:ident, $env:literal, $default:literal) => {{
            let declared = build.$field.as_deref();
            let override_value = get_env($env).filter(|s| !s.is_empty());
            let (value, source) = match override_value {
                Some(value) => {
                    if let Some(yaml) = declared {
                        if yaml != value {
                            eprintln!(
                                "vehicle-features: {}: env {}={value:?} overrides {vehicle}.yaml build.{}={yaml:?}",
                                stringify!($field), $env, stringify!($field)
                            );
                        }
                    }
                    (value, concat!("env ", $env).to_owned())
                }
                None => match declared {
                    Some(value) => (value.to_owned(), format!("{vehicle}.yaml")),
                    None => ($default.to_owned(), "default".to_owned()),
                },
            };
            if explain {
                eprintln!("vehicle-features: {:16} = {value:?} ({source})", stringify!($field));
            }
            build.$field = Some(value.clone());
            value
        }};
    }
    let board = knob!(board, "BOARD", "sakurah743");
    let rc = knob!(rc_protocol, "RC_PROTOCOL", "crsf");
    let outer = knob!(outer_loop, "OUTER_LOOP", "mpc");
    let pos = knob!(pos_source, "POS_SOURCE", "gps");
    let gps = knob!(gps_model, "GPS_MODEL", "ublox");
    let dual = knob!(gps_dual_antenna, "GPS_DUAL_ANTENNA", "no");
    let imu = knob!(imu_rate, "IMU_RATE", "8khz");
    let indi = knob!(indi, "INDI", "yes");
    let online = knob!(plan_online, "PLAN_ONLINE", "no");
    build.validate(vehicle)?;

    let estimator = get_env("ESTIMATOR")
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "eskf".into());
    if estimator != "eskf" {
        return Err(format!("ESTIMATOR must be eskf, got {estimator:?}"));
    }
    if explain {
        eprintln!("vehicle-features: estimator        = {estimator:?} (env-only)");
    }
    let outer_feature = match outer.as_str() {
        "mpc" => "outer_mpc",
        "mpc_full" => "outer_mpc_full",
        "cascade" => "outer_geometric",
        "rate" => "outer_rate",
        _ => unreachable!("validated outer_loop"),
    };
    let mut features = vec![
        format!("board_{board}"),
        format!("rx_{rc}"),
        format!("est_{estimator}"),
        format!("est_pos_{pos}"),
        outer_feature.to_owned(),
    ];
    for (enabled, feature) in [
        (gps == "unicore" && pos == "gps", "gps_unicore"),
        (dual == "yes" && pos == "gps", "gps_dual_antenna"),
        (
            get_env("DEFMT_UART").as_deref() == Some("true"),
            "defmt_uart",
        ),
        (imu == "1khz", "imu_1khz"),
        (indi == "no", "indi_off"),
        (online == "yes", "plan_online"),
    ] {
        if enabled {
            features.push(feature.to_owned());
        }
    }
    if outer == "mpc_full" && indi == "no" {
        eprintln!(
            "vehicle-features: mpc_full with indi=no uses static inversion without INDI increments; experimental bench configuration"
        );
    }
    Ok(features.join(","))
}

#[derive(Debug, PartialEq)]
enum Mode {
    Features { explain: bool },
    List,
}

fn mode(args: &[String]) -> Result<Mode, String> {
    match args {
        [] => Ok(Mode::Features { explain: false }),
        [arg] if arg == "--explain" => Ok(Mode::Features { explain: true }),
        [arg] if arg == "--list" => Ok(Mode::List),
        _ => Err("expected no arguments, --explain, or --list".into()),
    }
}

fn run() -> Result<(), String> {
    let mode = mode(&env::args().skip(1).collect::<Vec<_>>())?;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let current = env::var("VEHICLE")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "sakura_vicon".into());
    match mode {
        Mode::Features { explain } => {
            let vehicle = read_vehicle(&root, &current)?;
            println!(
                "{}",
                resolve(vehicle.build, &current, |key| env::var(key).ok(), explain)?
            );
        }
        Mode::List => {
            let mut rows = vec![
                [
                    "",
                    "vehicle",
                    "board",
                    "pos_source",
                    "outer_loop",
                    "airframe.name",
                ]
                .map(String::from),
            ];
            for name in vehicles(&root)? {
                let vehicle = read_vehicle(&root, &name)?;
                rows.push([
                    if name == current { "*" } else { "" }.into(),
                    name,
                    vehicle.build.board.unwrap_or_else(|| "-".into()),
                    vehicle.build.pos_source.unwrap_or_else(|| "-".into()),
                    vehicle.build.outer_loop.unwrap_or_else(|| "-".into()),
                    vehicle.airframe.name.unwrap_or_else(|| "<unset>".into()),
                ]);
            }
            let widths: [usize; 6] =
                std::array::from_fn(|i| rows.iter().map(|r| r[i].chars().count()).max().unwrap());
            for row in rows {
                println!(
                    "{}",
                    row.iter()
                        .zip(widths)
                        .map(|(cell, width)| format!("{cell:width$}"))
                        .collect::<Vec<_>>()
                        .join("  ")
                        .trim_end()
                );
            }
            println!(
                "\n* = current VEHICLE (.env / environment). Override per build: just build <vehicle>"
            );
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("vehicle-features: {error}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(yaml: &str, overrides: &[(&str, &str)]) -> Result<String, String> {
        let build = serde_yaml::from_str(yaml).unwrap();
        resolve(
            build,
            "test",
            |key| {
                overrides
                    .iter()
                    .find(|(k, _)| *k == key)
                    .map(|(_, v)| v.to_string())
            },
            false,
        )
    }

    #[test]
    fn release_vehicles_select_expected_features() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for (name, expected) in [
            (
                "sakura_vicon",
                "board_sakurah743,rx_crsf,est_eskf,est_pos_mocap,outer_mpc",
            ),
            (
                "sakura_um982",
                "board_sakurah743,rx_crsf,est_eskf,est_pos_gps,outer_mpc,gps_unicore",
            ),
            (
                "sakura_ublox_f9",
                "board_sakurah743,rx_crsf,est_eskf,est_pos_gps,outer_mpc",
            ),
        ] {
            let vehicle = read_vehicle(&root, name).unwrap();
            assert_eq!(
                resolve(vehicle.build, name, |_| None, false).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn environment_overrides_yaml_and_empty_overrides_are_ignored() {
        assert_eq!(
            features(
                "outer_loop: rate\npos_source: mocap",
                &[("OUTER_LOOP", "cascade"), ("POS_SOURCE", "")]
            )
            .unwrap(),
            "board_sakurah743,rx_crsf,est_eskf,est_pos_mocap,outer_geometric"
        );
        assert_eq!(
            features(
                "{}",
                &[
                    ("OUTER_LOOP", "mpc_full"),
                    ("IMU_RATE", "1khz"),
                    ("INDI", "no"),
                    ("PLAN_ONLINE", "yes"),
                    ("DEFMT_UART", "true")
                ]
            )
            .unwrap(),
            "board_sakurah743,rx_crsf,est_eskf,est_pos_gps,outer_mpc_full,defmt_uart,imu_1khz,indi_off,plan_online"
        );
    }

    #[test]
    fn heading_requires_unicore_and_gps_features_require_gps_position() {
        assert!(
            features("gps_dual_antenna: yes", &[])
                .unwrap_err()
                .contains("requires")
        );
        assert_eq!(
            features("gps_dual_antenna: yes\ngps_model: unicore", &[]).unwrap(),
            "board_sakurah743,rx_crsf,est_eskf,est_pos_gps,outer_mpc,gps_unicore,gps_dual_antenna"
        );
        assert_eq!(
            features(
                "gps_dual_antenna: yes\ngps_model: unicore\npos_source: mocap",
                &[]
            )
            .unwrap(),
            "board_sakurah743,rx_crsf,est_eskf,est_pos_mocap,outer_mpc"
        );
    }

    #[test]
    fn invalid_settings_and_missing_vehicles_fail() {
        for key in [
            "BOARD",
            "RC_PROTOCOL",
            "OUTER_LOOP",
            "POS_SOURCE",
            "GPS_MODEL",
            "GPS_DUAL_ANTENNA",
            "IMU_RATE",
            "INDI",
            "PLAN_ONLINE",
            "ESTIMATOR",
        ] {
            assert!(features("{}", &[(key, "typo")]).is_err(), "{key}");
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let error = read_vehicle(&root, "missing-vehicle").err().unwrap();
        assert!(error.contains("available vehicles:"));
        for name in ["sakura_ublox_f9", "sakura_um982", "sakura_vicon"] {
            assert!(error.contains(name), "{error}");
        }
        assert!(serde_yaml::from_str::<VehicleMetadata>("build:\n  imu_typo: 1khz").is_err());
    }

    #[test]
    fn cli_rejects_unknown_and_conflicting_options() {
        assert_eq!(mode(&[]).unwrap(), Mode::Features { explain: false });
        assert_eq!(
            mode(&["--explain".into()]).unwrap(),
            Mode::Features { explain: true }
        );
        assert_eq!(mode(&["--list".into()]).unwrap(), Mode::List);
        assert!(mode(&["--typo".into()]).is_err());
        assert!(mode(&["--list".into(), "--typo".into()]).is_err());
    }
}
