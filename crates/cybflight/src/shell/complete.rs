//! Tab completion glue: the shell's command grammar, its dynamic word
//! lists, and the insert / list-and-redraw terminal handling. Matching
//! itself lives in [`cybflight_core::shell_complete`] (host-tested).

use core::fmt::Write;

use cybflight_core::param_registry::ParamName;
use cybflight_core::params::{FirmwareConfig, PARAM_COUNT};
use cybflight_core::shell_complete::{complete, for_each_match, WordSource};
use embassy_usb::{
    class::cdc_acm::CdcAcmClass,
    driver::{Driver, EndpointError},
};

use super::{write_all, WriteBuf};

/// Every command `usb_serial::dispatch()` accepts, as completion
/// templates (grammar in [`cybflight_core::shell_complete`]). Keep in
/// sync with `dispatch()` / `HELP_TEXT` — a command missing here still
/// runs, it just won't complete.
const COMMANDS: &[&str] = &[
    "help",
    "imu1",
    "imu2",
    "imurate",
    "indistat",
    "att",
    "ocp",
    "rc",
    "rcstats",
    "attcontrol",
    "dshot",
    "power",
    "gps",
    "gpshealth",
    "gpsrtk",
    "fleetstate",
    "magext",
    "magint",
    "baro1",
    "baro2",
    "vicon",
    "timesync",
    "eskf",
    "health",
    "resetcause",
    "stream <topic> on",
    "stream <topic> off",
    "motor",
    "param list",
    "param get <param>",
    "param set <param>",
    "param diff --yaml",
    "param reset all",
    "param reset <param>",
    "param save --prune",
    "param defaults",
    #[cfg(feature = "outer_mpc")]
    "mission list",
    #[cfg(feature = "outer_mpc")]
    "mission get",
    #[cfg(feature = "outer_mpc")]
    "mission set <mission>",
    "led on",
    "led off",
    "blackbox record on",
    "blackbox record off",
    "blackbox status",
    "blackbox set none",
    "blackbox set small",
    "blackbox set mid",
    "blackbox set large",
    "blackbox set sysid",
    "blackbox ls",
    "blackbox get",
    "blackbox rm",
    "blackbox clean",
    #[cfg(feature = "postmortem")]
    "postmortem show",
    #[cfg(feature = "postmortem")]
    "postmortem clear",
    "reboot",
    "reboot --dfu",
];

/// `stream <topic> on|off` topics (the `stream … on/off` arms of
/// `dispatch()`).
const STREAM_TOPICS: &[&str] = &[
    "imu1",
    "imu2",
    "att",
    "ocp",
    "rc",
    "rcstats",
    "dshot",
    "power",
    "gps",
    "gpsrtk",
    "magext",
    "magint",
    "baro1",
    "baro2",
    "attcontrol",
    "vicon",
    "timesync",
    "eskf",
];

/// Past this many candidates an ambiguous Tab prints only the count —
/// `param set <Tab>` alone has hundreds.
const MAX_LISTED: usize = 64;
/// Wrap width for the candidate listing.
const LIST_WIDTH: usize = 80;

struct Words;

impl WordSource for Words {
    fn visit(&self, placeholder: &str, f: &mut dyn FnMut(&str)) {
        match placeholder {
            "<topic>" => STREAM_TOPICS.iter().for_each(|t| f(t)),
            "<param>" => {
                for idx in 0..PARAM_COUNT {
                    if let Some(name) = ParamName::of::<FirmwareConfig>(idx) {
                        f(name.as_str());
                    }
                }
            }
            #[cfg(feature = "outer_mpc")]
            "<mission>" => crate::control::offline_mission::PROFILES
                .iter()
                .for_each(|p| f(p.name)),
            _ => {}
        }
    }
}

/// Handle a Tab keypress on the partial line `buf[..*len]`.
///
/// Unique match (or an unambiguous common prefix): append it to the line
/// and echo it. Ambiguous with nothing to extend: list the candidates on
/// a fresh line, then redraw `prompt` and the line. No match (or no room
/// left in `buf`): ring the bell.
pub async fn tab_complete<'d>(
    class: &mut CdcAcmClass<'d, impl Driver<'d>>,
    buf: &mut [u8],
    len: &mut usize,
    prompt: &[u8],
) -> Result<(), EndpointError> {
    let Ok(line) = core::str::from_utf8(&buf[..*len]) else {
        return write_all(class, b"\x07").await;
    };
    let c = complete(COMMANDS, &Words, line);
    let ins = c.insert();
    if c.matches == 0 || *len + ins.len() > buf.len() {
        return write_all(class, b"\x07").await;
    }
    if !ins.is_empty() {
        buf[*len..*len + ins.len()].copy_from_slice(ins);
        *len += ins.len();
        return write_all(class, ins).await;
    }

    write_all(class, b"\r\n").await?;
    if c.matches > MAX_LISTED {
        let mut tmp = [0u8; 48];
        let mut w = WriteBuf::new(&mut tmp);
        write!(w, "({} matches)\r\n", c.matches).ok();
        write_all(class, w.as_slice()).await?;
    } else {
        list_matches(class, line).await?;
    }
    write_all(class, prompt).await?;
    write_all(class, &buf[..*len]).await
}

/// Print the candidates for `line` as wrapped, two-space-separated
/// columns. The matcher is a synchronous visitor and the USB write is
/// async, so fill one small buffer per pass, resuming at the first
/// candidate that did not fit.
async fn list_matches<'d>(
    class: &mut CdcAcmClass<'d, impl Driver<'d>>,
    line: &str,
) -> Result<(), EndpointError> {
    let mut next = 0usize;
    let mut col = 0usize;
    loop {
        let mut out = [0u8; 256];
        let mut w = WriteBuf::new(&mut out);
        let mut idx = 0usize;
        let mut full = false;
        for_each_match(COMMANDS, &Words, line, &mut |cand| {
            if idx >= next && !full {
                // Worst case: "\r\n" or "  " separator, then the word.
                if w.as_slice().len() + 2 + cand.len() > 256 {
                    full = true;
                } else {
                    if col > 0 && col + 2 + cand.len() > LIST_WIDTH {
                        w.write_str("\r\n").ok();
                        col = 0;
                    } else if col > 0 {
                        w.write_str("  ").ok();
                        col += 2;
                    }
                    w.write_str(cand).ok();
                    col += cand.len();
                    next = idx + 1;
                }
            }
            idx += 1;
        });
        write_all(class, w.as_slice()).await?;
        if !full {
            break;
        }
    }
    write_all(class, b"\r\n").await
}
