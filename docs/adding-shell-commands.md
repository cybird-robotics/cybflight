# Adding a Shell Command

The USB serial shell lives in `crates/cybflight/src/usb_serial.rs`. Commands are
dispatched by an exhaustive `match` on the trimmed input line inside `dispatch()`.
Adding a command always requires the same two or three edits; extra steps are only
needed when the command displays a new message type or subscribes to a new channel.

---

## Quick checklist

1. Add a help line to `HELP_TEXT` in `usb_serial.rs`
2. Add a match arm in `dispatch()` in `usb_serial.rs`
3. *(If the command prints a `msgs::*` type)* Add a `ShellMsg` impl in `shell/format.rs`
4. *(If the command needs a new streaming channel)* Add a subscriber in `shell_loop()`
   and a `Printable` impl in `shell/format.rs`

---

## Case 1 — Simple command with no channel access

**Example: `version` — print firmware build info**

### Step 1 — Add the help line (`usb_serial.rs`)

```rust
const HELP_TEXT: &[u8] = b"\
  ...
  version          print firmware version and build info\r\n\   // ← add
  ...
";
```

### Step 2 — Add the match arm (`usb_serial.rs`, inside `dispatch()`)

```rust
"version" => {
    let mut buf = [0u8; 128];
    let mut w = WriteBuf::new(&mut buf);
    let _ = write!(
        w,
        "cybflight v{} ({}) {}\r\n",
        crate::BUILD_VERSION,
        crate::GIT_HASH,
        crate::BUILD_TIMESTAMP,
    );
    write_all(class, w.as_slice()).await?;
}
```

`write_all` handles chunking at 64 bytes and the ZLP; `WriteBuf` handles `write!`
in no_std. A `[u8; 128]` stack buffer is large enough for most single-line responses.

---

## Case 2 — One-shot read from an existing channel

**Example: `odom` — print one VehicleOdometry snapshot**

The channel and subscriber type must already exist. If they don't, follow Case 4.

### Step 1 — Add a `ShellMsg` impl (`shell/format.rs`)

`ShellMsg<'_, T>` is the newtype that implements `fmt::Display` for shell output.
`VehicleOdometry` already has one (see `shell/format.rs:68`), so this step is done.
For a new message type `msgs::Foo`:

```rust
// shell/format.rs
impl fmt::Display for ShellMsg<'_, msgs::Foo> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = self.0;
        write!(f, "Foo(field_a={}, field_b={})", s.field_a, s.field_b)
    }
}
```

### Step 2 — Add the help line and match arm (`usb_serial.rs`)

```rust
// HELP_TEXT
  odom             one-shot odometry snapshot\r\n\

// dispatch()
"odom" => {
    let msg = next_message(odom_sub).await;   // see Case 4 for odom_sub
    let mut buf = [0u8; 256];
    let mut w = WriteBuf::new(buf.as_mut_slice());
    write!(w, "{}\r\n", ShellMsg(&msg)).ok();
    write_all(class, w.as_slice()).await?;
}
```

---

## Case 3 — Side-effect command (no output / platform action)

**Example: `reboot` — already implemented**

These commands write a brief confirmation, then call a function that does not return.

```rust
// dispatch()
"my-action" => {
    write_all(class, b"performing action...\r\n").await?;
    crate::platform::some_action();   // -> !  or returns normally
}
```

If the function returns normally, `dispatch` continues and the prompt is reprinted.
If it is `-> !` (like `sys_reboot`), `write_all` must complete before calling it
because the compiler will not reorder them across an `.await`.

Platform-level functions (`sys_reboot`, `enter_dfu`, etc.) belong in
`crates/cybflight/src/platform.rs`, not in `usb_serial.rs`.

---

## Case 4 — New streaming command

Adding streaming for a new message type requires changes in both `shell_loop` and
`dispatch`, plus a `Printable` impl.

**Example: `stream odom on/off`**

### Step 1 — `ShellMsg` impl (`shell/format.rs`)

Already present for `VehicleOdometry` at line 68. Add one if your type is new.

### Step 2 — `Printable` impl (`shell/format.rs`)

`Printable` ties a message type to a field in `ShellState`:

```rust
// shell/format.rs
impl Printable for msgs::VehicleOdometry {
    fn should_print(&self, ctx: &ShellState) -> bool { ctx.stream_odom }
    fn write_to(&self, w: &mut dyn core::fmt::Write) -> core::fmt::Result {
        write!(w, "{}", ShellMsg(self))
    }
}
```

Already present. If your type is new, add the field to `ShellState`:

```rust
pub struct ShellState {
    pub stream_imu: bool,
    pub stream_att: bool,
    pub stream_odom: bool,
    pub stream_foo: bool,    // ← add
}
```

### Step 3 — Publish channel and subscriber type (`sensors/mod.rs`)

A `PubSubChannel` for the message type must exist as a `pub static`. If it doesn't:

```rust
// sensors/mod.rs
pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex, msgs::VehicleOdometry, 4, 4, 1,
> = PubSubChannel::new();
```

Increment the `SUBS` const if you are adding a subscriber beyond the current count.

### Step 4 — Add the subscriber to `shell_loop` (`usb_serial.rs`)

```rust
// Type alias at top of file
type OdomSub = Subscriber<'static, CriticalSectionRawMutex, msgs::VehicleOdometry, 4, 4, 1>;

// Inside shell_loop():
let mut odom_sub: OdomSub = match crate::sensors::VEHICLE_ODOMETRY.subscriber() {
    Ok(s) => s,
    Err(_) => {
        defmt::error!("USB shell: VEHICLE_ODOMETRY subscriber slots exhausted");
        return;
    }
};
```

Then extend the `select3(...)` call to `select4(...)` (or `select` + `join` if you
need more) and add the corresponding match arm:

```rust
Either4::Fourth(WaitResult::Message(odom)) => {
    if stream_odom {
        let mut buf = [0u8; 256];
        let mut w = WriteBuf::new(buf.as_mut_slice());
        let _ = write!(w, "{}\r\n", ShellMsg(&odom));
        if write_all(class, w.as_slice()).await.is_err() {
            return;
        }
    }
}
Either4::Fourth(WaitResult::Lagged(n)) => {
    defmt::warn!("USB shell: dropped {} odometry samples", n);
}
```

### Step 5 — Help text and match arms (`usb_serial.rs`)

```rust
// HELP_TEXT
  stream odom on   stream vehicle odometry\r\n\
  stream odom off  stop odometry stream\r\n\

// dispatch() — stream_odom must be threaded in via &mut bool parameter
"stream odom on"  => { *stream_odom = true;  write_all(class, b"odometry stream on\r\n").await?; }
"stream odom off" => { *stream_odom = false; write_all(class, b"odometry stream off\r\n").await?; }
```

Add `stream_odom: &mut bool` to the `dispatch` signature and thread it through from
`shell_loop`.

---

## Buffer sizing

| Content | Suggested buffer |
|---|---|
| Single short value | `[u8; 64]` |
| Single message with several fields | `[u8; 128]` |
| Multi-field nested message | `[u8; 256]` |
| Help text / multi-line | use `write_all(class, CONST_BYTES)` directly |

`WriteBuf` silently truncates if the buffer is too small (returns `fmt::Error` which
is ignored by `let _ = write!(...)`). If output is being silently cut off, increase
the buffer size.

---

## What not to put in `usb_serial.rs`

- **Platform actions** (`reboot`, DFU entry, watchdog kick): `platform.rs`
- **Message formatting** (`ShellMsg` impls, `Printable` impls): `shell/format.rs`
- **New pub/sub channels**: `sensors/mod.rs`
- **New message types**: `msgs.rs`

`usb_serial.rs` owns only the USB transport, the shell loop, and the command
dispatch table. Keep it that way.
