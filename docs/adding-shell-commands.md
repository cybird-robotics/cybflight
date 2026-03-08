# Adding a Shell Command

The USB serial shell lives in `crates/cybflight/src/usb_serial.rs`. Commands are
dispatched by an exhaustive `match` on the trimmed input line inside `dispatch()`.

## Architecture overview

```
publisher tasks ──► PubSubChannel ──► msg_stream_task ──► SHELL_OUT ──► shell_loop
                                            ▲                               │
                                     AtomicBool flag                  writes USB CDC
                                     (toggled by dispatch)
```

- **`shell_loop`** selects over two futures only: USB keyboard input and
  `SHELL_OUT.receive()`. It has no topic-specific code and never changes.
- **`msg_stream_task`** is a generic async fn. One concrete
  `#[embassy_executor::task]` wrapper runs permanently per topic.
- **`STREAM_*: AtomicBool`** statics gate each stream. The shell sets them;
  the stream tasks read them.
- **One-shot commands** create a temporary subscriber on demand inside `dispatch`,
  use it once, then drop it.

---

## Quick checklist

1. Add a help line to `HELP_TEXT` in `usb_serial.rs`
2. Add a match arm in `dispatch()` in `usb_serial.rs`
3. *(If the command prints a `msgs::*` type)* Add a `ShellMsg` impl in `shell/format.rs`
4. *(If the command needs a new streaming topic)* Also:
   - Add a `pub static STREAM_FOO: AtomicBool` in `usb_serial.rs`
   - Add a concrete task wrapper calling `msg_stream_task`
   - Spawn it from `main.rs`
   - Ensure the channel has a free subscriber slot

---

## Case 1 — Simple command with no channel access

**Example: `version` — print firmware build info**

### Step 1 — Add the help line (`usb_serial.rs`)

```rust
const HELP_TEXT: &[u8] = b"\
  ...
  version            print firmware version and build info\r\n\
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

### Step 1 — Add a `ShellMsg` impl (`shell/format.rs`)

`ShellMsg<'_, T>` is the newtype that implements `fmt::Display` for shell output.
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

Also implement `msgs::Message` for the type in `msgs.rs` if not already done:

```rust
impl Message for msgs::Foo {}
```

### Step 2 — Add the help line and match arm (`usb_serial.rs`)

```rust
// HELP_TEXT
  odom               one-shot odometry snapshot\r\n\

// dispatch()
"odom" => {
    let mut sub = match crate::sensors::VEHICLE_ODOMETRY.subscriber() {
        Ok(s) => s,
        Err(_) => return write_all(class, b"error: no subscriber slot\r\n").await,
    };
    oneshot(class, &mut sub, 256).await?;
}
```

`oneshot` waits up to 150 ms for a message, formats it via `ShellMsg`, and writes
it to the CDC class. It prints `"no data (timeout)\r\n"` if nothing arrives.

The subscriber is created on-demand and dropped at the end of the match arm, so it
does not consume a slot permanently. Verify the channel's `SUBS` const is at least
`(number of permanent stream tasks) + 1`.

---

## Case 3 — Side-effect command (no output / platform action)

**Example: `reboot` — already implemented**

```rust
// dispatch()
"my-action" => {
    write_all(class, b"performing action...\r\n").await?;
    crate::platform::some_action();   // -> !  or returns normally
}
```

If the function is `-> !` (like `sys_reboot`), `write_all` must complete before
calling it — the compiler will not reorder them across an `.await`.

Platform-level functions (`sys_reboot`, `enter_dfu`, etc.) belong in
`crates/cybflight/src/platform.rs`, not in `usb_serial.rs`.

---

## Case 4 — New streaming topic

**Example: `stream odom on/off`**

`shell_loop` never changes. Adding a new topic means adding a flag, a task, and
two match arms.

### Step 1 — `ShellMsg` impl (`shell/format.rs`)

As in Case 2, Step 1.

### Step 2 — Publish channel (`sensors/mod.rs` or wherever appropriate)

```rust
pub static VEHICLE_ODOMETRY: PubSubChannel<
    CriticalSectionRawMutex, msgs::VehicleOdometry, 4, 4, 1,
> = PubSubChannel::new();
```

Set `SUBS` to at least `(number of stream tasks for this channel) + 1` to leave
room for one-shot commands.

### Step 3 — Stream flag and task wrapper (`usb_serial.rs`)

```rust
// At module level, alongside STREAM_IMU etc.:
pub static STREAM_ODOM: AtomicBool = AtomicBool::new(false);

// Concrete task wrapper:
#[embassy_executor::task]
pub async fn odom_stream_task() {
    msg_stream_task(&crate::sensors::VEHICLE_ODOMETRY, &STREAM_ODOM).await
}
```

### Step 4 — Spawn the task (`main.rs`)

```rust
spawner
    .spawn(cybflight::usb_serial::odom_stream_task())
    .unwrap_or_else(|_| defmt::panic!("failed to spawn odometry stream task"));
```

### Step 5 — Reset the flag on connection (`shell_loop`, `usb_serial.rs`)

```rust
// Inside shell_loop(), alongside the other resets:
STREAM_ODOM.store(false, Ordering::Relaxed);
```

### Step 6 — Help text and match arms (`usb_serial.rs`)

```rust
// HELP_TEXT
  stream odom on     stream vehicle odometry\r\n\
  stream odom off    stop odometry stream\r\n\

// dispatch()
"stream odom on"  => { STREAM_ODOM.store(true,  Ordering::Relaxed); write_all(class, b"odometry stream on\r\n").await?; }
"stream odom off" => { STREAM_ODOM.store(false, Ordering::Relaxed); write_all(class, b"odometry stream off\r\n").await?; }
```

---

## Buffer sizing

| Content | Suggested buffer |
|---|---|
| Single short value | `[u8; 64]` |
| Single message with several fields | `[u8; 128]` |
| Multi-field nested message | `[u8; 256]` |
| Help text / multi-line | use `write_all(class, CONST_BYTES)` directly |

Pass the buffer size to `oneshot` as the second argument. It is capped at 256
internally. `WriteBuf` silently truncates on overflow — if output is cut off,
increase the buffer.

---

## What not to put in `usb_serial.rs`

- **Platform actions** (`reboot`, DFU entry, watchdog kick): `platform.rs`
- **Message formatting** (`ShellMsg` impls): `shell/format.rs`
- **New pub/sub channels**: `sensors/mod.rs` (or the appropriate module)
- **New message types**: `msgs.rs`

`usb_serial.rs` owns the USB transport, the stream flags, the stream task wrappers,
the shell loop, and the command dispatch table. Keep it that way.
