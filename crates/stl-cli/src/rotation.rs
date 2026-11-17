//! Estimation of the orientation-aligned frame (ADR 0002) from a set of
//! reference maps.
//!
//! For every level the estimator takes the moduli `|z_l|` of the `L` wavelet
//! gauges of one reference, projects them onto the doubled-angle harmonics
//! (`aligned::project`), decimates the three projections a few times — the
//! local preferred direction is a large-scale property of the scanning, not
//! a per-pixel noise — and accumulates them over the references.  The
//! direction of a level is the argument of the accumulated harmonic, its
//! concentration the ratio of that argument's modulus to the summed
//! amplitude.


use std::path::{Path, PathBuf};
use stl_core::aligned::{project, upscale, DirectionEstimate};

use stl_core::local::LocalGeometry;
use stl_core::stencil::Gauge;
use stl_mpi::io;

/// Options of the estimator, which must be those of the run it feeds.
#[derive(Clone, Debug)]
pub struct FrameOptions {
    /// Orientations `L` of the wavelet (`--L` of the run).
    pub l: usize,
    /// Stencil side of the wavelet (`--kernel-size` of the run).
    pub kernel_size: usize,
    /// Pixels per block of the full-sky wavelet.
    pub block_pixels: usize,
    /// Decimations applied to the amplitude profile before the harmonic:
    /// the preferred direction is smoothed over `4^smooth` pixels.  A level
    /// stops smoothing before its `nside` falls below 2.
    pub smooth: usize,
}

impl Default for FrameOptions {
    fn default() -> Self {
        Self { l: 4, kernel_size: 5, block_pixels: 16384, smooth: 2 }
    }
}

/// Number of levels of a pyramid of level-0 `nside`: `log2(nside) + 1`.
pub fn levels(nside: i64) -> usize {
    64 - (nside as u64).leading_zeros() as usize
}

/// Per-level accumulation of the local preferred direction.
pub struct Frame {
    /// Geometry of the pyramid (identity frame: the estimator *derives* it).
    geo: LocalGeometry,
    /// Decimations applied at each level, for the upsampling.
    steps: Vec<u32>,
    /// Accumulated harmonics of each level, at the smoothed resolution.
    estimates: Vec<DirectionEstimate>,
    /// The same, kept in two halves of the references: their agreement is
    /// the alignment concentration (0: no common direction, 1: one).
    halves: [Vec<DirectionEstimate>; 2],
    /// References accumulated so far.
    pub references: usize,
}

impl Frame {
    /// An estimator of the full-sky pyramid above `nside`, level 0.
    pub fn new(nside: i64, options: &FrameOptions) -> Self {
        let geo = LocalGeometry::new(nside, options.kernel_size, options.l, Gauge::Cosmo, options.block_pixels);
        let n = levels(nside);
        let mut steps = Vec::with_capacity(n);
        let mut estimates = Vec::with_capacity(n);
        let mut halves = [Vec::with_capacity(n), Vec::with_capacity(n)];
        for dg in 0..n {
            let mut k = 0usize;
            while k < options.smooth && geo.nside(dg + k + 1) >= 2 {
                k += 1;
            }
            steps.push(k as u32);
            estimates.push(DirectionEstimate::new(geo.pixels(dg + k)));
            halves[0].push(DirectionEstimate::new(geo.pixels(dg + k)));
            halves[1].push(DirectionEstimate::new(geo.pixels(dg + k)));
        }
        Self { geo, steps, estimates, halves, references: 0 }
    }

    /// Adds one reference map (NESTED, level 0, `12 nside^2` pixels).
    pub fn push(&mut self, map: &[f64]) {
        let l = self.geo.l;
        let n0 = self.geo.pixels(0);
        assert_eq!(map.len(), n0, "reference has {} pixels, level 0 has {n0}", map.len());
        // Pyramid of the reference.
        let mut pyr = Vec::with_capacity(self.estimates.len());
        pyr.push(map.to_vec());
        for dg in 0..self.estimates.len() - 1 {
            let y = self.geo.decimate(&pyr[dg], dg);
            pyr.push(y);
        }
        for dg in 0..self.estimates.len() {
            let n = self.geo.pixels(dg);
            let mut amplitude = vec![0.0f64; l * n];
            for block in self.geo.blocks(dg) {
                let wave = self.geo.wave(&pyr[dg], &block, dg);
                for (o, chunk) in wave.z.chunks(block.len()).enumerate() {
                    // `wave.z` is `[1, L, |block|]`; a block is `[o, |block|]`.
                    debug_assert!(o < l);
                    for (i, &p) in block.iter().enumerate() {
                        amplitude[o * n + p as usize] = chunk[i].norm();
                    }
                }
            }
            let mut est = project(&amplitude, l);
            for step in 0..self.steps[dg] as usize {
                let level = dg + step;
                est.cos2 = self.geo.decimate(&est.cos2, level);
                est.sin2 = self.geo.decimate(&est.sin2, level);
                est.amplitude = self.geo.decimate(&est.amplitude, level);
            }
            assert_eq!(est.cos2.len(), self.estimates[dg].cos2.len(), "smoothed size of level {dg}");
            self.estimates[dg].add(&est);
            let half = (self.references % 2) as usize;
            self.halves[half][dg].add(&est);
        }
        self.references += 1;
    }

    /// The alignment concentration of every level: the mean over pixels of
    /// `cos 2(phi_A - phi_B)` between the two halves of the references.
    /// Close to 1 when they agree on the local preferred direction, close to
    /// 0 when they do not; needs at least two references.
    pub fn split_half_agreement(&self) -> Vec<f64> {
        self.halves[0]
            .iter()
            .zip(&self.halves[1])
            .map(|(a, b)| a.agreement(b))
            .collect()
    }

    /// The local preferred direction of every level, at full level
    /// resolution, in `[0, pi)` radians and NESTED order.
    pub fn angles(&self) -> Vec<Vec<f32>> {
        self.estimates
            .iter()
            .enumerate()
            .map(|(dg, e)| {
                let coarse: Vec<f64> = e.angles().iter().map(|&v| v as f64).collect();
                let up = upscale(&coarse, self.steps[dg]);
                assert_eq!(up.len(), self.geo.pixels(dg), "level {dg} after upsampling");
                up.into_iter().map(|v| v as f32).collect()
            })
            .collect()
    }

    /// The anisotropy strength of every level (`aligned::DirectionEstimate`).
    pub fn concentration(&self) -> Vec<Vec<f64>> {
        self.estimates
            .iter()
            .enumerate()
            .map(|(dg, e)| upscale(&e.concentration(), self.steps[dg]))
            .collect()
    }
}

/// Writes the frame of every level (`angle_dg{d}.npy`, `concentration_dg{d}.npy`)
/// and its description (`rotation.json`) to `dir`.
pub fn write(dir: &Path, nside: i64, options: &FrameOptions, references: &[PathBuf], frame: &Frame) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let angles = frame.angles();
    let concentration = frame.concentration();
    let agreement = frame.split_half_agreement();
    let mut files = Vec::new();
    for (dg, (a, c)) in angles.iter().zip(&concentration).enumerate() {
        let a_path = dir.join(format!("angle_dg{dg}.npy"));
        let c_path = dir.join(format!("concentration_dg{dg}.npy"));
        let a64: Vec<f64> = a.iter().map(|&v| v as f64).collect();
        io::write_npy(&a_path, &a64)?;
        io::write_npy(&c_path, c)?;
        files.push(serde_json::json!({
            "dg": dg,
            "nside": nside >> dg,
            "angle": a_path.file_name().unwrap().to_string_lossy(),
            "concentration": c_path.file_name().unwrap().to_string_lossy(),
            "mean_concentration": c.iter().sum::<f64>() / c.len().max(1) as f64,
            "split_half_agreement": agreement[dg],
        }));
    }
    let report = serde_json::json!({
        "schema": "stl-ensemble-rotation-1",
        "nside": nside,
        "orientations": options.l,
        "kernel": options.kernel_size,
        "smooth": options.smooth,
        "gauge": "cosmo",
        "references": references.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "levels": files,
    });
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("rotation.json"), text).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(())
}
