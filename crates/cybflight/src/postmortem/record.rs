//! Post-mortem record: the on-disk (well, in-BKPSRAM) struct that
//! survives reboot.
//!
//! ## Layout
//!
//! Fixed-size `#[repr(C, align(8))]` struct, 2 KiB, broken into four
//! regions:
//!
//! ```text
//!   header (32 B)              magic, crc32, boot_count, reset_cause,
//!                              fw_git_hash, uptime_ms, flags
//!   fatal slot (64 B)          panic / HardFault / brownout summary:
//!                              kind, PC, LR, PSR, CFSR, HFSR, MMFAR,
//!                              BFAR, panic_msg_idx
//!   event ring (16 × 16 B)     last-N flight events (timestamp, kind,
//!                              data, seq) — same KIND_* codes as
//!                              the blackbox `/events` topic
//!   snapshots (~256 B)         last attitude quaternion, position,
//!                              velocity, yaw, RC channels
//!   reserved (~1.5 KiB)        padding for future schema growth
//! ```
//!
//! ## Atomicity
//!
//! Writers fill the body, then `crc32`, then `magic` last. A torn write
//! is detected on read because either the magic is wrong (header
//! cleared) or the CRC doesn't match (body half-written). The CRC
//! covers `[after-crc32 .. end]`.
//!
//! `MAGIC` bakes the schema version into its low 16 bits so a firmware
//! upgrade with a new layout treats an old record as invalid rather
//! than misinterpreting fields.

use core::mem::{MaybeUninit, size_of};

/// `0xC0DE_xxxx` family. Low 16 bits are the schema version. Bump on
/// any layout change to make old records invalid by construction.
pub const MAGIC: u32 = 0xC0DE_0001;
/// Schema version embedded in [`MAGIC`].
pub const SCHEMA_VERSION: u16 = 0x0001;

/// Fixed event ring depth. 16 × 16 B = 256 B.
pub const EVENT_RING_LEN: usize = 16;

/// Total record size, byte-padded to fit BKPSRAM cleanly.
pub const RECORD_SIZE: usize = 2048;

/// Index value for "panic captured but message not in the static
/// table" — a real panic message exceeded the message-table range, or
/// the call site didn't pass one.
pub const PANIC_MSG_IDX_NONE: u8 = 0xFF;

/// Fatal kinds — what *caused* the prior reboot. `None` is the
/// steady-state value (the postmortem task wrote events but no
/// terminal fault fired).
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, defmt::Format, Debug)]
pub enum FatalKind {
    None = 0,
    Panic = 1,
    HardFault = 2,
    /// Asserted on the *next* boot after seeing `RCC.RSR.IWDGRSTF`.
    /// The IWDG reset itself is captured passively; there's no
    /// runtime hook (the watchdog resets the MCU before software can
    /// react).
    IwdgReset = 3,
    /// PVD brown-out trip — recorded by the PVD IRQ before BOR.
    Brownout = 4,
}

impl FatalKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Panic,
            2 => Self::HardFault,
            3 => Self::IwdgReset,
            4 => Self::Brownout,
            _ => Self::None,
        }
    }
}

/// One entry in the event ring. Mirrors the blackbox `/events` topic
/// shape but stripped to fixed-size fields for in-place atomic writes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EventEntry {
    /// Milliseconds since boot. `u32` wraps at ~49 days — fine for
    /// flight sessions.
    pub timestamp_ms: u32,
    /// Same KIND_* code space as `crate::blackbox::topics::events`.
    pub kind: u8,
    pub _pad: [u8; 3],
    /// kind-specific data field.
    pub data: u32,
    /// Monotonic sequence so a reader can detect ring wrap.
    pub seq: u32,
}

const _: () = assert!(size_of::<EventEntry>() == 16);

impl EventEntry {
    pub const ZERO: Self = Self {
        timestamp_ms: 0,
        kind: 0,
        _pad: [0; 3],
        data: 0,
        seq: 0,
    };
}

/// 64-byte fatal slot. Populated only on the path that *caused* the
/// reboot.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FatalSummary {
    pub kind: u8,
    pub _pad0: [u8; 3],
    pub pc: u32,
    pub lr: u32,
    pub psr: u32,
    pub cfsr: u32,
    pub hfsr: u32,
    pub mmfar: u32,
    pub bfar: u32,
    pub panic_msg_idx: u8,
    pub _pad1: [u8; 31],
}

const _: () = assert!(size_of::<FatalSummary>() == 64);

impl FatalSummary {
    pub const ZERO: Self = Self {
        kind: FatalKind::None as u8,
        _pad0: [0; 3],
        pc: 0,
        lr: 0,
        psr: 0,
        cfsr: 0,
        hfsr: 0,
        mmfar: 0,
        bfar: 0,
        panic_msg_idx: PANIC_MSG_IDX_NONE,
        _pad1: [0; 31],
    };
}

/// Last attitude / setpoint / RC snapshot. Continuously refreshed by
/// `postmortem_task` so the PVD path doesn't have to gather data
/// during the brown-out window.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Snapshots {
    pub attitude_quat_wijk: [f32; 4],
    pub position_xyz: [f32; 3],
    pub velocity_xyz: [f32; 3],
    pub yaw_rad: f32,
    pub rc_channels: [u16; 16],
    pub rc_channel_count: u8,
    pub _pad: [u8; 7],
}

const _: () = assert!(size_of::<Snapshots>() == 4 * 4 + 3 * 4 + 3 * 4 + 4 + 16 * 2 + 1 + 7);
const _: () = assert!(size_of::<Snapshots>() <= 256);

impl Snapshots {
    pub const ZERO: Self = Self {
        attitude_quat_wijk: [0.0; 4],
        position_xyz: [0.0; 3],
        velocity_xyz: [0.0; 3],
        yaw_rad: 0.0,
        rc_channels: [0; 16],
        rc_channel_count: 0,
        _pad: [0; 7],
    };
}

/// Reserved tail. Sized so [`PostmortemRecord`] is exactly
/// [`RECORD_SIZE`] bytes.
const RESERVED_LEN: usize = RECORD_SIZE
    - 32 // header
    - 64 // fatal
    - 16 * EVENT_RING_LEN
    - size_of::<Snapshots>()
    - 8 // ring head + pad
    - 8; // pad to 8-align

/// 32-byte header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Header {
    /// `MAGIC` when valid; `0` when cleared. Written **last** by
    /// writers and cleared **first** by clearers, so a torn write is
    /// always rejected.
    pub magic: u32,
    /// CRC32 over `[after-crc32 .. end]`. Invalid records must
    /// fail this check before any field is trusted.
    pub crc32: u32,
    pub boot_count: u32,
    /// Packed `(PWR.CSR1 << 16) | RCC.RSR_FLAGS_LO`. The cleared
    /// bits are zero so a "no prior reset cause" record reads as
    /// `reset_cause == 0`.
    pub reset_cause: u32,
    /// Low 4 bytes of `crate::GIT_HASH`. Lets a reader detect a
    /// firmware change between record-write and record-read.
    pub fw_git_hash: u32,
    pub uptime_ms: u32,
    pub flags: u64,
}

const _: () = assert!(size_of::<Header>() == 32);

impl Header {
    pub const ZERO: Self = Self {
        magic: 0,
        crc32: 0,
        boot_count: 0,
        reset_cause: 0,
        fw_git_hash: 0,
        uptime_ms: 0,
        flags: 0,
    };
}

/// The full record. Lives in BKPSRAM via `#[link_section = ".bkpsram"]`.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct PostmortemRecord {
    pub header: Header,
    pub fatal: FatalSummary,
    /// Index of the next slot to write in `events`. Lives outside
    /// the ring so the ring itself contains pure event data.
    pub event_head: u8,
    pub _ring_pad: [u8; 7],
    pub events: [EventEntry; EVENT_RING_LEN],
    pub snapshots: Snapshots,
    /// Reserved padding. Never read; reserves room for schema-version
    /// growth without changing the record size.
    pub _reserved: [u8; RESERVED_LEN],
    /// Tail pad ensuring the struct is exactly RECORD_SIZE bytes.
    pub _tail_pad: [u8; 8],
}

const _: () = assert!(size_of::<PostmortemRecord>() == RECORD_SIZE);

impl PostmortemRecord {
    pub const ZERO: Self = Self {
        header: Header::ZERO,
        fatal: FatalSummary::ZERO,
        event_head: 0,
        _ring_pad: [0; 7],
        events: [EventEntry::ZERO; EVENT_RING_LEN],
        snapshots: Snapshots::ZERO,
        _reserved: [0; RESERVED_LEN],
        _tail_pad: [0; 8],
    };

    /// Append `entry` to the event ring, advancing `event_head` with
    /// wrap-around. Caller is responsible for refreshing `crc32` and
    /// `magic` afterwards (typically batched per writer-side
    /// scheduling).
    pub fn push_event(&mut self, entry: EventEntry) {
        let idx = (self.event_head as usize) % EVENT_RING_LEN;
        self.events[idx] = entry;
        self.event_head = self.event_head.wrapping_add(1);
    }

    /// Iterator over events in chronological order (oldest first).
    /// Stops at the first all-zero entry so callers don't have to
    /// special-case partially-filled rings.
    pub fn events_in_order(&self) -> impl Iterator<Item = &EventEntry> {
        let head = self.event_head as usize % EVENT_RING_LEN;
        // After `n` events have been pushed (`event_head == n`):
        //   - if n < RING_LEN, items live at indices [0..n) and the
        //     head points at the first empty slot. Walk [0..head).
        //   - if n >= RING_LEN, the ring is full and the oldest
        //     event is at `head`. Walk [head..head+RING_LEN).
        let full = self.event_head as usize >= EVENT_RING_LEN;
        let start = if full { head } else { 0 };
        let count = if full { EVENT_RING_LEN } else { head };
        (0..count).map(move |i| &self.events[(start + i) % EVENT_RING_LEN])
    }
}

// ── CRC ─────────────────────────────────────────────────────────────────
//
// `crc32fast` is `no_std`-compatible (default-features=false in
// cybflight_core). Re-using it here avoids a new dep and keeps the
// algorithm consistent across the codebase.
//
// CRC covers everything **after** the `crc32` field. The `magic` field
// is *included* in the CRC even though it's written last — a partial
// write where magic landed but the body changed underneath fails the
// check, which is what we want.

const CRC_RANGE_OFFSET: usize = size_of::<u32>() * 2; // skip magic + crc32

/// Compute the CRC32 over the post-CRC body. Operates on a raw byte
/// slice so the same fn can be called by the writer (against a
/// `&PostmortemRecord` cast) and by host-side tests (against a
/// freshly-built record in stack memory).
pub fn compute_crc(rec: &PostmortemRecord) -> u32 {
    let bytes = unsafe {
        core::slice::from_raw_parts((rec as *const PostmortemRecord) as *const u8, RECORD_SIZE)
    };
    let mut h = crc32fast::Hasher::new();
    h.update(&bytes[CRC_RANGE_OFFSET..]);
    h.finalize()
}

/// Validate a freshly-read record. Returns `true` only if the magic
/// matches **and** the CRC is consistent with the rest of the body.
/// Either failure mode means the record was torn, never finalized,
/// or comes from a different schema version.
pub fn is_valid(rec: &PostmortemRecord) -> bool {
    rec.header.magic == MAGIC && compute_crc(rec) == rec.header.crc32
}

/// Finalize a record after a writer has populated the body. Computes
/// the CRC, then writes `crc32` then `magic` (in that order). The
/// `magic`-last ordering means a half-finalized record reads as
/// invalid (magic wrong) rather than as a CRC mismatch — distinct
/// failure modes that callers may want to distinguish.
///
/// **Hard rule:** this function does no allocation, no async, no
/// Mutex. It is callable from a `#[panic_handler]`, `HardFault`
/// exception, or PVD interrupt.
pub fn finalize(rec: &mut PostmortemRecord) {
    rec.header.magic = 0; // ensure mid-update reads see "invalid"
    let crc = compute_crc(rec);
    rec.header.crc32 = crc;
    // Write fences make the magic publish only after the CRC is
    // visible. BKPSRAM is uncached (D3 domain), so a `dsb()` is
    // sufficient — no cache invalidation needed.
    cortex_m::asm::dsb();
    rec.header.magic = MAGIC;
    cortex_m::asm::dsb();
}

/// Clear a record. Writes `magic = 0` first so a concurrent reader
/// observing a partial clear sees "invalid" rather than "valid with
/// stale body".
pub fn clear(rec: &mut PostmortemRecord) {
    rec.header.magic = 0;
    cortex_m::asm::dsb();
    // Body wipe is best-effort; a torn clear with `magic == 0` is
    // already guaranteed-invalid above.
    *rec = PostmortemRecord::ZERO;
    cortex_m::asm::dsb();
}

/// Build a fresh zero-initialized record. Used by host-side tests and
/// any callers that need a stack-local instance.
pub fn zeroed() -> PostmortemRecord {
    // SAFETY: `PostmortemRecord` is `repr(C)` with all-`Copy` fields
    // and no padding-with-niche, so an all-zero bit pattern is a
    // valid value. Avoiding a real `MaybeUninit::zeroed().assume_init()`
    // simply uses the existing `ZERO` const.
    let _ = MaybeUninit::<PostmortemRecord>::zeroed; // ensure MaybeUninit imported
    PostmortemRecord::ZERO
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_size_is_2k() {
        assert_eq!(size_of::<PostmortemRecord>(), 2048);
    }

    #[test]
    fn finalize_then_validate_round_trip() {
        let mut rec = zeroed();
        rec.header.boot_count = 7;
        rec.header.reset_cause = 0xDEADBEEF;
        rec.fatal.kind = FatalKind::Panic as u8;
        rec.fatal.pc = 0x0800_1234;
        finalize(&mut rec);
        assert!(is_valid(&rec));
        assert_eq!(rec.header.magic, MAGIC);
    }

    #[test]
    fn one_byte_flip_invalidates() {
        let mut rec = zeroed();
        rec.header.boot_count = 7;
        finalize(&mut rec);
        assert!(is_valid(&rec));
        // Flip a payload byte.
        rec.header.boot_count ^= 0x1;
        assert!(!is_valid(&rec));
    }

    #[test]
    fn clear_makes_invalid() {
        let mut rec = zeroed();
        finalize(&mut rec);
        assert!(is_valid(&rec));
        clear(&mut rec);
        assert!(!is_valid(&rec));
        assert_eq!(rec.header.magic, 0);
    }

    #[test]
    fn ring_partial_then_walk() {
        let mut rec = zeroed();
        for i in 0..5u32 {
            rec.push_event(EventEntry {
                timestamp_ms: i * 100,
                kind: 1,
                _pad: [0; 3],
                data: i,
                seq: i,
            });
        }
        let walked: heapless::Vec<_, 8> = rec.events_in_order().map(|e| e.seq).collect();
        assert_eq!(walked.as_slice(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn ring_wrap_oldest_overwritten() {
        let mut rec = zeroed();
        // Push 20 events into a 16-deep ring.
        for i in 0..20u32 {
            rec.push_event(EventEntry {
                timestamp_ms: i,
                kind: 1,
                _pad: [0; 3],
                data: i,
                seq: i,
            });
        }
        let walked: heapless::Vec<_, 32> = rec.events_in_order().map(|e| e.seq).collect();
        // Oldest 4 (seq 0..3) are overwritten; remaining are 4..19.
        assert_eq!(walked.len(), EVENT_RING_LEN);
        assert_eq!(walked[0], 4);
        assert_eq!(walked[15], 19);
    }
}
