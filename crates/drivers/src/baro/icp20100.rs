//! ICP-20100 barometric pressure sensor driver (I2C only).
//!
//! FIFO-based readout with continuous measurement at Mode 1 (120 Hz ODR,
//! 30 Hz bandwidth).
//!
//! Reference: ArduPilot AP_Baro_ICP201XX.cpp

use embedded_hal_async::delay::DelayNs;
use embedded_hal_async::i2c::I2c;

use super::{BaroReading, ReadBaro};

// ---------------------------------------------------------------------------
// Register addresses (from ArduPilot AP_Baro_ICP201XX.h)
// ---------------------------------------------------------------------------

const REG_TRIM1_MSB: u8 = 0x05;
const REG_TRIM2_LSB: u8 = 0x06;
const REG_TRIM2_MSB: u8 = 0x07;
const REG_DEVICE_ID: u8 = 0x0C;
const REG_OTP_CONFIG1: u8 = 0xAC;
const REG_OTP_MR_LSB: u8 = 0xAD;
const REG_OTP_MR_MSB: u8 = 0xAE;
const REG_OTP_MRA_LSB: u8 = 0xAF;
const REG_OTP_MRA_MSB: u8 = 0xB0;
const REG_OTP_MRB_LSB: u8 = 0xB1;
const REG_OTP_MRB_MSB: u8 = 0xB2;
const REG_OTP_ADDRESS: u8 = 0xB5;
const REG_OTP_COMMAND: u8 = 0xB6;
const REG_OTP_DATA: u8 = 0xB8;
const REG_OTP_STATUS: u8 = 0xB9;
const REG_OTP_DBG2: u8 = 0xBC;
const REG_MASTER_LOCK: u8 = 0xBE;
const REG_OTP_STATUS2: u8 = 0xBF;
const REG_MODE_SELECT: u8 = 0xC0;
const REG_INTERRUPT_MASK: u8 = 0xC2;
const REG_FIFO_CONFIG: u8 = 0xC3;
const REG_FIFO_FILL: u8 = 0xC4;
const REG_DEVICE_STATUS: u8 = 0xCD;
const REG_VERSION: u8 = 0xD3;
const REG_FIFO_BASE: u8 = 0xFA;
const REG_EMPTY: u8 = 0x00;

const DEVICE_ID: u8 = 0x63;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum Error<E> {
    I2c(E),
    BadId(u8),
    FifoOverflow,
}

impl<E: defmt::Format> defmt::Format for Error<E> {
    fn format(&self, f: defmt::Formatter) {
        match self {
            Error::I2c(e) => defmt::write!(f, "I2c({})", e),
            Error::BadId(id) => defmt::write!(f, "BadId({:#x})", id),
            Error::FifoOverflow => defmt::write!(f, "FifoOverflow"),
        }
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

pub struct Icp20100<I2C> {
    i2c: I2C,
    addr: u8,
}

impl<I2C> Icp20100<I2C>
where
    I2C: I2c,
{
    /// Initialize the ICP-20100 barometer.
    /// No separate probe needed — returns `BadId` if the sensor is absent.
    pub async fn new(
        mut i2c: I2C,
        addr: u8,
        delay: &mut impl DelayNs,
    ) -> Result<Self, Error<I2C::Error>> {
        // 1. Soft reset — sensor may be in continuous mode from previous boot.
        //    Use raw write (ignore errors, device might NACK initially).
        let _ = i2c.write(addr, &[REG_MODE_SELECT, 0x00]).await;
        dummy(&mut i2c, addr).await;
        delay.delay_ms(10).await;

        // 2. Read Device ID (twice — first read may be stale per ArduPilot)
        let _ = read_reg(&mut i2c, addr, REG_DEVICE_ID).await;
        let id = read_reg(&mut i2c, addr, REG_DEVICE_ID).await?;
        if id != DEVICE_ID {
            return Err(Error::BadId(id));
        }

        // 3. Read version
        let version = read_reg(&mut i2c, addr, REG_VERSION).await?;
        defmt::debug!("ICP20100: version={:#x}", version);

        delay.delay_ms(10).await;

        // 4. Soft reset (proper — polls device status)
        soft_reset(&mut i2c, addr, delay).await?;

        // 5. Boot sequence (OTP calibration) — needed for version != 0xB2
        if version != 0xB2 {
            boot_sequence(&mut i2c, addr, delay).await?;
        }

        // 6. Configure for Mode 1 continuous P+T measurement
        // mode = (OP_MODE1 << 5) | (MEAS_CONTINUOUS << 3) | FIFO_PRES_TEMP
        //      = (1 << 5) | (1 << 3) | 0 = 0x28
        mode_select(&mut i2c, addr, 0x28, delay).await?;

        // 7. Wait for FIR filter settling — skip first 14 packets
        wait_read(&mut i2c, addr, delay).await?;

        defmt::debug!("ICP20100: init complete");
        Ok(Self { i2c, addr })
    }
}

impl<I2C> ReadBaro for Icp20100<I2C>
where
    I2C: I2c,
    I2C::Error: defmt::Format,
{
    type Error = Error<I2C::Error>;

    async fn read(&mut self) -> Result<BaroReading, Self::Error> {
        // Poll FIFO until data available — yield between iterations so other
        // tasks sharing this I2C bus can acquire the mutex.
        let mut packets: usize = 0;
        for _ in 0..200 {
            let fill = read_reg(&mut self.i2c, self.addr, REG_FIFO_FILL).await?;
            packets = (fill & 0x1F) as usize;
            if packets > 0 {
                break;
            }
            crate::yield_now().await;
        }

        if packets == 0 {
            return Err(Error::FifoOverflow);
        }

        if packets > 16 {
            flush_fifo(&mut self.i2c, self.addr).await?;
            return Err(Error::FifoOverflow);
        }

        // Burst read all FIFO packets in one I2C transaction (per ArduPilot)
        let byte_count = packets * 6;
        let mut fifo_data = [0u8; 96]; // max 16 packets * 6 bytes
        self.i2c
            .write_read(self.addr, &[REG_FIFO_BASE], &mut fifo_data[..byte_count])
            .await
            .map_err(Error::I2c)?;
        dummy(&mut self.i2c, self.addr).await;

        // Parse all packets
        let mut pressure_sum: f64 = 0.0;
        let mut temp_sum: f64 = 0.0;

        for i in 0..packets {
            let o = i * 6;
            // 20-bit signed, little-endian:
            // byte[0]=LSB, byte[1]=mid, byte[2] lower nibble=MSB
            let p_raw = ((fifo_data[o + 2] as i32 & 0x0F) << 16)
                | ((fifo_data[o + 1] as i32) << 8)
                | (fifo_data[o] as i32);
            let p_raw = sign_extend_20(p_raw);

            let t_raw = ((fifo_data[o + 5] as i32 & 0x0F) << 16)
                | ((fifo_data[o + 4] as i32) << 8)
                | (fifo_data[o + 3] as i32);
            let t_raw = sign_extend_20(t_raw);

            // Pressure: (POUT / 2^17) * 40 kPa + 70 kPa → Pa
            let pressure_pa = (p_raw as f64 / 131072.0) * 40000.0 + 70000.0;
            // Temperature: (TOUT / 2^18) * 65°C + 25°C
            let temp_c = (t_raw as f64 / 262144.0) * 65.0 + 25.0;

            pressure_sum += pressure_pa;
            temp_sum += temp_c;
        }

        let n = packets as f64;
        Ok(BaroReading {
            pressure_pa: (pressure_sum / n) as f32,
            temp_c: (temp_sum / n) as f32,
        })
    }

    async fn recover(&mut self) -> Result<(), Self::Error> {
        // Stop measurement (raw write, don't poll device status)
        write_reg(&mut self.i2c, self.addr, REG_MODE_SELECT, 0x00).await?;
        flush_fifo(&mut self.i2c, self.addr).await?;

        // Restart continuous mode (raw write for recovery)
        write_reg(&mut self.i2c, self.addr, REG_MODE_SELECT, 0x28).await?;

        // Poll until FIFO has data
        for _ in 0..2000 {
            let fill = read_reg(&mut self.i2c, self.addr, REG_FIFO_FILL).await?;
            if (fill & 0x1F) > 0 {
                break;
            }
            crate::yield_now().await;
        }
        flush_fifo(&mut self.i2c, self.addr).await?;

        Ok(())
    }

    fn sample_rate_hz(&self) -> f32 {
        120.0
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn sign_extend_20(val: i32) -> i32 {
    if val & 0x080000 != 0 {
        val | !0xFFFFF
    } else {
        val & 0xFFFFF
    }
}

/// Dummy register read after every I2C transaction (ICP-20100 quirk).
async fn dummy<I2C: I2c>(i2c: &mut I2C, addr: u8) {
    let _ = i2c.write_read(addr, &[REG_EMPTY], &mut [0u8]).await;
}

async fn read_reg<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    reg: u8,
) -> Result<u8, Error<I2C::Error>> {
    let mut buf = [0u8];
    i2c.write_read(addr, &[reg], &mut buf)
        .await
        .map_err(Error::I2c)?;
    dummy(i2c, addr).await;
    Ok(buf[0])
}

async fn write_reg<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    reg: u8,
    val: u8,
) -> Result<(), Error<I2C::Error>> {
    i2c.write(addr, &[reg, val]).await.map_err(Error::I2c)?;
    dummy(i2c, addr).await;
    Ok(())
}

/// Poll REG_DEVICE_STATUS bit 0, then write REG_MODE_SELECT.
async fn mode_select<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    mode: u8,
    delay: &mut impl DelayNs,
) -> Result<(), Error<I2C::Error>> {
    for _ in 0..100 {
        let status = read_reg(i2c, addr, REG_DEVICE_STATUS).await?;
        if status & 0x01 != 0 {
            break;
        }
        delay.delay_ms(1).await;
    }
    write_reg(i2c, addr, REG_MODE_SELECT, mode).await
}

async fn soft_reset<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    delay: &mut impl DelayNs,
) -> Result<(), Error<I2C::Error>> {
    mode_select(i2c, addr, 0x00, delay).await?;
    delay.delay_ms(2).await;
    flush_fifo(i2c, addr).await?;
    write_reg(i2c, addr, REG_FIFO_CONFIG, 0x00).await?;
    write_reg(i2c, addr, REG_INTERRUPT_MASK, 0xFF).await?;
    Ok(())
}

async fn flush_fifo<I2C: I2c>(i2c: &mut I2C, addr: u8) -> Result<(), Error<I2C::Error>> {
    let val = read_reg(i2c, addr, REG_FIFO_FILL).await?;
    write_reg(i2c, addr, REG_FIFO_FILL, val | 0x80).await
}

/// Wait for FIR settling (14+ packets), flush, then wait for new data.
async fn wait_read<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    delay: &mut impl DelayNs,
) -> Result<(), Error<I2C::Error>> {
    // Wait until at least 14 packets accumulated
    loop {
        delay.delay_ms(10).await;
        let fill = read_reg(i2c, addr, REG_FIFO_FILL).await?;
        if (fill & 0x1F) >= 14 {
            break;
        }
    }
    flush_fifo(i2c, addr).await?;

    // Wait for fresh data
    loop {
        delay.delay_ms(10).await;
        let fill = read_reg(i2c, addr, REG_FIFO_FILL).await?;
        if (fill & 0x1F) > 0 {
            break;
        }
    }
    Ok(())
}

async fn boot_sequence<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    delay: &mut impl DelayNs,
) -> Result<(), Error<I2C::Error>> {
    // Check if boot sequence already completed
    let boot_status = read_reg(i2c, addr, REG_OTP_STATUS2).await?;
    if boot_status & 0x01 != 0 {
        return Ok(());
    }

    // Enter power mode to activate OTP power domain
    mode_select(i2c, addr, 0x04, delay).await?;
    delay.delay_ms(4).await;

    // Unlock main registers
    write_reg(i2c, addr, REG_MASTER_LOCK, 0x1F).await?;

    // Enable OTP and write switch
    let cfg1 = read_reg(i2c, addr, REG_OTP_CONFIG1).await?;
    write_reg(i2c, addr, REG_OTP_CONFIG1, cfg1 | 0x03).await?;
    delay.delay_us(10).await;

    // Toggle OTP reset pin
    let dbg2 = read_reg(i2c, addr, REG_OTP_DBG2).await?;
    write_reg(i2c, addr, REG_OTP_DBG2, dbg2 | 0x80).await?;
    delay.delay_us(10).await;
    let dbg2 = read_reg(i2c, addr, REG_OTP_DBG2).await?;
    write_reg(i2c, addr, REG_OTP_DBG2, dbg2 & !0x80).await?;
    delay.delay_us(10).await;

    // Program redundant read configuration
    write_reg(i2c, addr, REG_OTP_MRA_LSB, 0x04).await?;
    write_reg(i2c, addr, REG_OTP_MRA_MSB, 0x04).await?;
    write_reg(i2c, addr, REG_OTP_MRB_LSB, 0x21).await?;
    write_reg(i2c, addr, REG_OTP_MRB_MSB, 0x20).await?;
    write_reg(i2c, addr, REG_OTP_MR_LSB, 0x10).await?;
    write_reg(i2c, addr, REG_OTP_MR_MSB, 0x80).await?;

    // Read OTP calibration values
    let offset = read_otp(i2c, addr, 0xF8, 0x10, delay).await?;
    let gain = read_otp(i2c, addr, 0xF9, 0x10, delay).await?;
    let hfosc = read_otp(i2c, addr, 0xFA, 0x10, delay).await?;

    defmt::debug!(
        "ICP20100 OTP: offset={:#x}, gain={:#x}, hfosc={:#x}",
        offset,
        gain,
        hfosc
    );
    delay.delay_us(10).await;

    // Apply trim values
    // TRIM1_MSB: offset in bits 5:0
    let trim1 = read_reg(i2c, addr, REG_TRIM1_MSB).await?;
    write_reg(i2c, addr, REG_TRIM1_MSB, (trim1 & !0x3F) | (offset & 0x3F)).await?;

    // TRIM2_MSB: gain in bits 6:4
    let trim2m = read_reg(i2c, addr, REG_TRIM2_MSB).await?;
    write_reg(i2c, addr, REG_TRIM2_MSB, (trim2m & !0x70) | ((gain & 0x07) << 4)).await?;

    // TRIM2_LSB: hfosc in bits 6:0
    let trim2l = read_reg(i2c, addr, REG_TRIM2_LSB).await?;
    write_reg(i2c, addr, REG_TRIM2_LSB, (trim2l & !0x7F) | (hfosc & 0x7F)).await?;

    delay.delay_us(10).await;

    // Disable OTP
    let cfg1 = read_reg(i2c, addr, REG_OTP_CONFIG1).await?;
    write_reg(i2c, addr, REG_OTP_CONFIG1, cfg1 & !0x03).await?;

    // Lock and standby
    write_reg(i2c, addr, REG_MASTER_LOCK, 0x00).await?;
    mode_select(i2c, addr, 0x00, delay).await?;

    Ok(())
}

async fn read_otp<I2C: I2c>(
    i2c: &mut I2C,
    addr: u8,
    otp_addr: u8,
    cmd: u8,
    delay: &mut impl DelayNs,
) -> Result<u8, Error<I2C::Error>> {
    write_reg(i2c, addr, REG_OTP_ADDRESS, otp_addr).await?;
    write_reg(i2c, addr, REG_OTP_COMMAND, cmd).await?;

    // Wait for OTP ready (status == 0)
    for _ in 0..100 {
        let status = read_reg(i2c, addr, REG_OTP_STATUS).await?;
        if status == 0 {
            break;
        }
        delay.delay_us(1).await;
    }

    read_reg(i2c, addr, REG_OTP_DATA).await
}
