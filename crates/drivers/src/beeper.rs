use embedded_hal::digital::{OutputPin, StatefulOutputPin};

/// Generic beeper driver with inversion support.
pub struct Beeper<P: OutputPin + StatefulOutputPin> {
    pin: P,
    inverted: bool,
}

impl<P: OutputPin + StatefulOutputPin> Beeper<P> {
    pub fn new(pin: P, inverted: bool) -> Self {
        Self { pin, inverted }
    }

    #[inline]
    pub fn on(&mut self) {
        if self.inverted {
            let _ = self.pin.set_low();
        } else {
            let _ = self.pin.set_high();
        }
    }

    #[inline]
    pub fn off(&mut self) {
        if self.inverted {
            let _ = self.pin.set_high();
        } else {
            let _ = self.pin.set_low();
        }
    }

    #[inline]
    pub fn toggle(&mut self) {
        let _ = self.pin.toggle();
    }
}
