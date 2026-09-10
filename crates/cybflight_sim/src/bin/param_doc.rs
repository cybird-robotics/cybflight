//! Generate the parameter reference (docs/parameters.md) from the live
//! registry. Run via `just params-doc`; commit the result whenever the
//! schema changes.

use cybflight_core::param_registry::{ParamGroup, ParamName};
use cybflight_core::params::{FirmwareConfig, PARAM_COUNT};

fn fmt_bound(v: f32) -> String {
    if v.is_infinite() {
        "—".into()
    } else {
        format!("{v}")
    }
}

fn main() {
    println!("# Parameter reference");
    println!();
    println!("Auto-generated from the `#[derive(Params)]` registry by");
    println!("`just params-doc` — do not edit by hand. {PARAM_COUNT} parameters.");
    println!();
    println!("Values shown by `param list`/`get`; set with `param set <name> <value>`");
    println!("(disarmed), persist with `param save`, inspect overrides with");
    println!("`param diff [--yaml]`, revert with `param reset <name>|all`.");
    println!("Baked defaults come from `vehicles/<VEHICLE>.yaml`; runtime overrides");
    println!("live in the flash KV store and win per-key — so a saved override");
    println!("shadows a re-flashed YAML edit. Follow `param reset` with");
    println!("`param save --prune` to drop the key from the store; a plain save");
    println!("only records the current baked value (the log has no tombstone).");
    println!();
    println!("| Name | Unit | Min | Max | Apply | Description |");
    println!("|---|---|---|---|---|---|");
    for idx in 0..PARAM_COUNT {
        let name = ParamName::of::<FirmwareConfig>(idx).expect("name");
        let meta = FirmwareConfig::param_meta(idx);
        // Escape the few markdown-sensitive characters docs use.
        let doc = meta.doc.replace('|', "\\|");
        println!(
            "| `{}` | {} | {} | {} | {} | {} |",
            name.as_str(),
            if meta.unit.is_empty() { "—" } else { meta.unit },
            fmt_bound(meta.min),
            fmt_bound(meta.max),
            if meta.reboot { "reboot" } else { "live" },
            doc,
        );
    }
}
