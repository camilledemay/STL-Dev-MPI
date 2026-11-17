//! The orientation-aligned frame (ADR 0002): the per-pixel stencil offset,
//! its composition with the gauges, and the direction estimator.
//!
//! The reference-free checks here are exact or near-exact, so that a wrong
//! sign, a wrongly placed offset or a wrong harmonic shows up as a failure
//! rather than as a slightly different map.

use num_complex::Complex64;
use stl_core::aligned::{project, upscale, DirectionEstimate};
use stl_core::local::LocalGeometry;
use stl_core::stencil::Gauge;
use std::f64::consts::PI;

/// Deterministic pseudo-random map (splitmix64), so that the test is
/// reproducible without a dependency.
fn random_map(n: usize) -> Vec<f64> {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    (0..n)
        .map(|_| {
            s ^= s >> 30;
            s = s.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            s ^= s >> 27;
            s = s.wrapping_mul(0x94d0_49bb_1331_11eb);
            s ^= s >> 31;
            (s as f64 / u64::MAX as f64) - 0.5
        })
        .collect()
}

/// A zero offset is the sky frame: the wavelet field is bit-identical.
#[test]
fn zero_offset_is_the_sky_frame() {
    let (nside, l, kernel) = (4i64, 4usize, 5usize);
    let npix = (12 * nside * nside) as usize;
    let plain = LocalGeometry::new(nside, kernel, l, Gauge::Cosmo, npix);
    let mut rotated = plain.clone();
    rotated.set_rotation(vec![vec![0.0f32; npix]]);
    let block: Vec<i64> = (0..npix as i64).collect();
    let x = random_map(npix);
    let a = plain.wave(&x, &block, 0);
    let b = rotated.wave(&x, &block, 0);
    assert_eq!(a.z.len(), b.z.len());
    for (i, (u, v)) in a.z.iter().zip(&b.z).enumerate() {
        assert_eq!(u.re.to_bits(), v.re.to_bits(), "re {i}");
        assert_eq!(u.im.to_bits(), v.im.to_bits(), "im {i}");
    }
}

/// An offset of one gauge step moves every gauge onto its neighbour, in both
/// hemispheres: gauge `g` with the offset equals gauge `g + 1` without it.
/// This pins the direction *and* the sign of the offset, which is what the
/// estimator relies on.
#[test]
fn one_gauge_step_equals_the_next_gauge() {
    let (nside, l, kernel) = (8i64, 4usize, 5usize);
    let npix = (12 * nside * nside) as usize;
    let step = PI / l as f64;
    let plain = LocalGeometry::new(nside, kernel, l, Gauge::Cosmo, npix);
    let mut rotated = plain.clone();
    rotated.set_rotation(vec![vec![step as f32; npix]]);
    let block: Vec<i64> = (0..npix as i64).collect();
    let x = random_map(npix);
    let a = plain.wave(&x, &block, 0);
    let b = rotated.wave(&x, &block, 0);
    let scale = a.z.iter().fold(0f64, |m, v| m.max(v.norm())).max(1e-12);
    for g in 0..l - 1 {
        let mut err = 0f64;
        for p in 0..npix {
            err = err.max((b.z[g * npix + p] - a.z[(g + 1) * npix + p]).norm());
        }
        // A tap exactly between two pixels may bind to either one: the value
        // moves by one interpolation weight, not by the rotation.
        assert!(err <= 1e-6 * scale, "gauge {g}: {err:e} (scale {scale:e})");
    }
}

/// The estimator returns the doubled-angle direction of the amplitude
/// profile, and its concentration is one for a single orientation and zero
/// for a flat profile.
#[test]
fn estimator_direction_and_concentration() {
    let l = 4;
    let step = PI / l as f64;
    for (li, want) in [(0usize, 0.0), (1, step), (2, 2.0 * step), (3, 3.0 * step)] {
        let mut amp = vec![0.0; l];
        amp[li] = 1.0;
        let e = project(&amp, l);
        let phi = e.angles()[0] as f64;
        assert!((phi - want).abs() <= 1e-6, "gauge {li}: {phi} vs {want}");
        assert!((e.concentration()[0] - 1.0).abs() <= 1e-9);
    }
    // Two equal adjacent orientations point halfway between them.
    let mut amp = vec![0.0; l];
    amp[0] = 1.0;
    amp[1] = 1.0;
    let phi = project(&amp, l).angles()[0] as f64;
    assert!((phi - 0.5 * step).abs() <= 1e-6, "{phi}");
    // A flat profile has no direction.
    let flat = project(&vec![1.0; l], l);
    assert!(flat.concentration()[0] < 1e-9);
    // The profile is unaffected by an overall scale.
    let mut weak = vec![0.0; l];
    weak[3] = 0.25;
    let phi = project(&weak, l).angles()[0] as f64;
    assert!((phi - 3.0 * step).abs() <= 1e-6, "{phi}");
}

/// Estimates of successive references add up; the upsampling replicates each
/// coarse pixel into its four descendants.
#[test]
fn accumulate_and_upscale() {
    let (l, npix) = (4usize, 8usize);
    let amp = vec![1.0; l * npix];
    let mut total = DirectionEstimate::new(npix);
    total.add(&project(&amp, l));
    total.add(&project(&amp, l));
    assert!((total.amplitude[0] - 2.0 * l as f64).abs() < 1e-12);
    let up = upscale(&[1.0, 2.0], 1);
    assert_eq!(up, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]);
    // Upsampling a direction field keeps the direction of each ancestor.
    let mut e = DirectionEstimate::new(2);
    e.cos2 = vec![1.0, 0.0];
    e.sin2 = vec![0.0, 1.0];
    e.amplitude = vec![1.0, 1.0];
    let f = DirectionEstimate {
        cos2: upscale(&e.cos2, 1),
        sin2: upscale(&e.sin2, 1),
        amplitude: upscale(&e.amplitude, 1),
    };
    assert_eq!(f.angles()[0], f.angles()[3]);
    assert_eq!(f.angles()[4], f.angles()[7]);
}

/// Agreement of two estimates: 1 on the same direction, -1 opposite, 0 on
/// perpendicular directions, and averaged over the pixels.
#[test]
fn agreement_of_two_estimates() {
    let mut a = DirectionEstimate::new(2);
    a.cos2 = vec![1.0, 1.0];
    a.sin2 = vec![0.0, 0.0];
    a.amplitude = vec![1.0, 1.0];
    assert!((a.agreement(&a) - 1.0).abs() < 1e-12);
    let mut half = a.clone();
    half.cos2 = vec![1.0, 0.0];
    half.sin2 = vec![0.0, 1.0];
    assert!((a.agreement(&half) - 0.5).abs() < 1e-12);
    let mut opposite = a.clone();
    opposite.cos2 = vec![-1.0, -1.0];
    assert!((a.agreement(&opposite) + 1.0).abs() < 1e-12);
}

#[test]
fn adjoint_with_an_offset_field() {
    let (nside, l, kernel) = (4i64, 4usize, 5usize);
    let npix = (12 * nside * nside) as usize;
    let mut geo = LocalGeometry::new(nside, kernel, l, Gauge::Cosmo, npix);
    let field: Vec<f32> = (0..npix).map(|p| (p as f32) * 0.37).collect();
    geo.set_rotation(vec![field]);
    let block: Vec<i64> = (0..npix as i64).collect();
    let x = random_map(npix);
    let wave = geo.wave(&x, &block, 0);
    let gz: Vec<Complex64> = (0..l * npix).map(|i| Complex64::new((i % 7) as f64 - 3.0, (i % 5) as f64 - 2.0)).collect();
    let gx = geo.wave_adjoint(&gz, &block, &wave);
    let lhs: f64 = wave.z.iter().zip(&gz).map(|(a, b)| a.re * b.re + a.im * b.im).sum();
    let kd = wave.domain.len();
    let n = geo.pixels(0);
    let rhs: f64 = (0..gx.len()).map(|i| gx[i] * x[(i / kd) * n + wave.domain[i % kd] as usize]).sum();
    assert!((lhs - rhs).abs() <= 1e-10 * lhs.abs().max(1.0), "{lhs} vs {rhs}");
}
