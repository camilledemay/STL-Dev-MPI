//! The orientation-aligned frame (ADR 0002): the local preferred direction of
//! a field, per pixel and per level, and the arithmetic that turns an
//! ensemble of estimates into stencil offsets.
//!
//! The gauges of the wavelet span `pi` (`stencil::rotation_matrices`), so the
//! orientation amplitudes `|z_l|` have period `pi`: the first harmonic of the
//! *doubled* angle is the direction estimator, and `|harmonic| / amplitude`
//! is the **concentration**, which says whether a pixel has a preferred
//! direction at all.

use std::f64::consts::PI;

/// Accumulated doubled-angle harmonics of one level, summed over references.
#[derive(Clone, Debug, Default)]
pub struct DirectionEstimate {
    /// `sum_r sum_l A_l cos(2 l pi / L)`, one value per pixel.
    pub cos2: Vec<f64>,
    /// `sum_r sum_l A_l sin(2 l pi / L)`, one value per pixel.
    pub sin2: Vec<f64>,
    /// `sum_r sum_l A_l`, one value per pixel (the normalisation).
    pub amplitude: Vec<f64>,
}

impl DirectionEstimate {
    /// The zero estimate of a level with `npix` pixels.
    pub fn new(npix: usize) -> Self {
        Self { cos2: vec![0.0; npix], sin2: vec![0.0; npix], amplitude: vec![0.0; npix] }
    }

    /// Accumulates the estimate of one reference (same pixel count).
    pub fn add(&mut self, other: &DirectionEstimate) {
        assert_eq!(self.cos2.len(), other.cos2.len(), "estimates of different levels");
        for (a, b) in self.cos2.iter_mut().zip(&other.cos2) {
            *a += b;
        }
        for (a, b) in self.sin2.iter_mut().zip(&other.sin2) {
            *a += b;
        }
        for (a, b) in self.amplitude.iter_mut().zip(&other.amplitude) {
            *a += b;
        }
    }

    /// The local preferred direction of every pixel, in `[0, pi)` radians.
    pub fn angles(&self) -> Vec<f32> {
        self.cos2
            .iter()
            .zip(&self.sin2)
            .map(|(&c, &s)| {
                let mut phi = 0.5 * s.atan2(c);
                if phi < 0.0 {
                    phi += PI;
                }
                phi as f32
            })
            .collect()
    }

    /// `|harmonic| / amplitude` of every pixel, in `[0, 1]`: how strongly
    /// the amplitude profile of that pixel has a preferred direction (zero
    /// for a flat profile).  Not an agreement between estimates.
    pub fn concentration(&self) -> Vec<f64> {
        self.cos2
            .iter()
            .zip(&self.sin2)
            .zip(&self.amplitude)
            .map(|((&c, &s), &a)| if a > 0.0 { (c * c + s * s).sqrt() / a } else { 0.0 })
            .collect()
    }

    /// Mean over pixels of `cos 2 (phi - phi_other)`: 1 when the two
    /// estimates see the same direction everywhere, 0 when they are
    /// independent (the alignment concentration of two halves).
    pub fn agreement(&self, other: &DirectionEstimate) -> f64 {
        assert_eq!(self.cos2.len(), other.cos2.len(), "estimates of different levels");
        let mut sum = 0.0;
        let mut n = 0usize;
        for ((&c1, &s1), (&c2, &s2)) in self.cos2.iter().zip(&self.sin2).zip(other.cos2.iter().zip(&other.sin2)) {
            let (r1, r2) = ((c1 * c1 + s1 * s1).sqrt(), (c2 * c2 + s2 * s2).sqrt());
            if r1 > 0.0 && r2 > 0.0 {
                sum += (c1 * c2 + s1 * s2) / (r1 * r2);
                n += 1;
            }
        }
        sum / n.max(1) as f64
    }
}

/// Projection of the `L` orientation amplitudes of one reference onto the
/// doubled-angle harmonics (`amplitude[l * npix + p]`).
pub fn project(amplitude: &[f64], l: usize) -> DirectionEstimate {
    let npix = amplitude.len() / l;
    assert_eq!(amplitude.len(), l * npix, "amplitudes are [L, npix]");
    let mut e = DirectionEstimate::new(npix);
    for (li, chunk) in amplitude.chunks(npix).enumerate() {
        let theta = PI * li as f64 / l as f64;
        let (s, c) = (2.0 * theta).sin_cos();
        for p in 0..npix {
            let a = chunk[p];
            e.cos2[p] += a * c;
            e.sin2[p] += a * s;
            e.amplitude[p] += a;
        }
    }
    e
}

/// Nearest-parent upsampling: `values` at `nside >> bits` (NESTED order) to
/// `nside`, every pixel taking its ancestor's value.
pub fn upscale(values: &[f64], bits: u32) -> Vec<f64> {
    let shift = 2 * bits;
    (0..values.len() << shift).map(|p| values[p >> shift]).collect()
}
