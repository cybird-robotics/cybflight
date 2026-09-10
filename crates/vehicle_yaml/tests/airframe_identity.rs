//! Cross-file invariant: `airframe.name` identifies a *physical drone*,
//! so every vehicle YAML that claims a given name must describe the same
//! hardware.
//!
//! The same airframe routinely needs two vehicle files — one indoor
//! (`pos_source: mocap`) and one outdoor (`pos_source: gps`) — because the
//! environment half of the config genuinely differs. The airframe half
//! does not: mass, inertia and motor geometry are facts about the machine
//! on the bench, and a drone does not grow different arms when it goes
//! outside.
//!
//! Without this check the two files are just a copy-paste that drifts. It
//! is the *identity* claim that makes the drift detectable, and detecting
//! it matters because these values feed the INDI control-effectiveness
//! matrix directly — a wrong motor position is mistuned attitude control,
//! not a cosmetic mismatch.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The `airframe:` block, kept as raw YAML so the comparison is on the
/// declared text's *meaning* (serde_yaml value equality) rather than on
/// whatever subset the firmware bake happens to project into params.
fn airframe_blocks() -> BTreeMap<String, Vec<(String, serde_yaml::Value)>> {
    let dir = vehicles_dir();
    let mut by_name: BTreeMap<String, Vec<(String, serde_yaml::Value)>> = BTreeMap::new();

    for entry in std::fs::read_dir(&dir).expect("read vehicles/ dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|s| s.to_str()) != Some("yaml") {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let raw = std::fs::read_to_string(&path).expect("read vehicle yaml");
        let doc: serde_yaml::Value = match serde_yaml::from_str(&raw) {
            Ok(v) => v,
            // Parse errors are the schema tests' business, not ours.
            Err(_) => continue,
        };
        let Some(airframe) = doc.get("airframe") else {
            continue;
        };
        let Some(name) = airframe.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        by_name
            .entry(name.to_string())
            .or_default()
            .push((stem, airframe.clone()));
    }
    by_name
}

fn vehicles_dir() -> PathBuf {
    // CARGO_MANIFEST_DIR = crates/vehicle_yaml
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vehicles")
        .canonicalize()
        .expect("locate vehicles/ dir")
}

#[test]
fn same_name_airframes_agree() {
    let mut failures = Vec::new();

    for (name, files) in airframe_blocks() {
        let Some(((first_stem, first), rest)) = files.split_first() else {
            continue;
        };
        for (stem, block) in rest {
            if block == first {
                continue;
            }
            // Report the specific keys that disagree — "the blocks differ"
            // is useless when the block is 10 lines of nested motors.
            let mut diffs = Vec::new();
            let empty = serde_yaml::Mapping::new();
            let a = first.as_mapping().unwrap_or(&empty);
            let b = block.as_mapping().unwrap_or(&empty);
            for key in a.keys().chain(b.keys()) {
                let (va, vb) = (a.get(key), b.get(key));
                if va != vb {
                    let k = key.as_str().unwrap_or("?");
                    if !diffs.iter().any(|(dk, _, _): &(String, _, _)| dk == k) {
                        diffs.push((k.to_string(), va.cloned(), vb.cloned()));
                    }
                }
            }
            let detail = diffs
                .iter()
                .map(|(k, va, vb)| {
                    format!(
                        "      airframe.{k}:\n        {first_stem}: {}\n        {stem}: {}",
                        yaml_inline(va),
                        yaml_inline(vb)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n");
            failures.push(format!(
                "  airframe.name = {name:?} is declared by both \
                 '{first_stem}.yaml' and '{stem}.yaml', but they describe \
                 different hardware:\n{detail}"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "vehicle YAMLs disagree about the airframe they name.\n\n{}\n\n\
         `airframe.name` identifies one physical drone. Two files may share \
         a name (e.g. an indoor and an outdoor build of the same machine), \
         but then the `airframe:` block — mass, inertia, motor geometry, \
         thrust model — must be identical in both: it describes the \
         hardware, not the environment. These values feed the INDI \
         control-effectiveness matrix, so a mismatch is a real tuning \
         fault.\n\nEither correct whichever file is wrong, or give the two \
         machines different names.",
        failures.join("\n\n")
    );
}

fn yaml_inline(v: &Option<serde_yaml::Value>) -> String {
    match v {
        None => "<absent>".to_string(),
        Some(v) => serde_yaml::to_string(v)
            .unwrap_or_else(|_| "<unprintable>".into())
            .trim_end()
            .replace('\n', " "),
    }
}
