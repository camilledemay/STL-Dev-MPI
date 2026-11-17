//! Targets of the ensemble objective from a population of reference maps
//! (`stl_port/calibration.py`, `build_calibration`): the pooled LUT, the
//! mean of the features over a bootstrap draw of the references and their
//! unbiased variance over all of them, the mean within-map SC sigmas, the
//! PDF support and the C_l of the initial Gaussian fields.  The source fixes
//! 54 references (TRAIN54) and a numpy draw of 10; here the references are
//! any list of maps and the draw is given (all the references when there
//! are as many as members).

use super::fields::{band_powers, lp_windows, Cic, Minkowski};
use super::losses::ScCoordinates;
use super::lut::Lut;
use num_complex::Complex64;
use serde_json::{json, Value};
use stl_core::plan::{Coefficient, Family};
use stl_core::sht::Sht;

type C64 = Complex64;

/// Targets of the five blocks: bootstrap means and population variances
/// of the features of the references, with the SC sigmas, PDF support and
/// initial C_l.
#[derive(Clone, Debug, Default)]
pub struct Targets {
    /// Bootstrap mean of the SC features (native SC coordinates) of the
    /// Gaussian maps g (GSC block).
    pub gsc_mean: Vec<C64>,
    /// Population variance of the SC features of g (GSC block).
    pub gsc_var: Vec<f64>,
    /// Mean within-map sigma of the SC coefficients of g.
    pub gsc_sigma: Vec<f64>,
    /// Bootstrap mean of the SC features of kappa (PSC block).
    pub psc_mean: Vec<C64>,
    /// Population variance of the SC features of kappa (PSC block).
    pub psc_var: Vec<f64>,
    /// Mean within-map sigma of the SC coefficients of kappa.
    pub psc_sigma: Vec<f64>,
    /// Bootstrap mean of the C_l band powers of kappa.
    pub power_mean: Vec<f64>,
    /// Population variance of the band powers.
    pub power_var: Vec<f64>,
    /// Bootstrap mean of the PDF (CIC histogram, 100 bins) of kappa.
    pub pdf_mean: Vec<f64>,
    /// Population variance of the PDF bins.
    pub pdf_var: Vec<f64>,
    /// Bootstrap mean of the Minkowski functionals `[3, k]` of kappa.
    pub mf_mean: Vec<f64>,
    /// Population variance of the Minkowski functionals `[3, k]`.
    pub mf_var: Vec<f64>,
    /// `[lo, hi]` of the PDF bins: the range of the references widened by 2 %
    /// on each side.
    pub pdf_support: [f64; 2],
    /// C_l (`l = 0..=3 nside - 1`) of the optimization maps, averaged over the
    /// bootstrap draw: the spectrum of the initial Gaussian fields.
    pub initialization_cl: Vec<f64>,
    /// Pixel mean of the Gaussian map g of the first bootstrap reference.
    pub initialization_reference_mean: f64,
}

/// The calibration: configuration of the plan, LUT and targets.
#[derive(Clone, Debug)]
pub struct Calibration {
    /// Direct centered physical maps, without a LUT or Gaussian SC block.
    pub physical_only: bool,
    /// Only the physical SC block; valid only with `physical_only`.
    pub sc_only: bool,
    /// HEALPix nside of the members and of the references.
    pub nside: usize,
    /// Number of scales `J` of the plan.
    pub scales: usize,
    /// Number of orientations `L` (4).
    pub orientations: usize,
    /// Wavelet stencil size (5).
    pub kernel: usize,
    /// Number of coefficients of the plan.
    pub plan_len: usize,
    /// Names of the reference maps (the population of the variances).
    pub references: Vec<String>,
    /// Indices in `references` of the bootstrap draw (means of the targets).
    pub bootstrap_ids: Vec<usize>,
    /// LUT pooled over all the references, `kappa = f(g)`; empty and unused
    /// in physical-only mode.
    pub lut: Lut,
    /// Targets of the five blocks.
    pub targets: Targets,
}

/// Mean over `ids` and unbiased variance over all rows of `rows` `[R, n]`
/// (`population_targets`).
pub fn population<T: Copy + Into<C64>>(rows: &[Vec<T>], ids: &[usize]) -> (Vec<C64>, Vec<f64>) {
    let n = rows[0].len();
    let r = rows.len() as f64;
    let c = |v: T| -> C64 { v.into() };
    let mean: Vec<C64> = (0..n).map(|i| ids.iter().map(|&k| c(rows[k][i])).sum::<C64>() / ids.len() as f64).collect();
    let all: Vec<C64> = (0..n).map(|i| rows.iter().map(|row| c(row[i])).sum::<C64>() / r).collect();
    let var = (0..n).map(|i| rows.iter().map(|row| (c(row[i]) - all[i]).norm_sqr()).sum::<f64>() / (r - 1.0)).collect();
    (mean, var)
}

/// Block means of a NESTED map down to `nside` through the source's steps:
/// to nside 512 (as `hp.ud_grade` of the 2048 maps) rounded to `f32`, then
/// to `nside`, rounded to `f32` (means in `f64`).
pub fn degrade(x: &[f64], nside: usize) -> Result<Vec<f64>, String> {
    let n_in = x.len();
    let n_out = 12 * nside * nside;
    if n_in % n_out != 0 || !(n_in / n_out).is_power_of_two() {
        return Err(format!("reference of {n_in} pixels cannot be degraded to nside {nside}"));
    }
    let mean = |v: &[f64], f: usize| -> Vec<f64> { v.chunks(f).map(|c| ((c.iter().sum::<f64>() / f as f64) as f32) as f64).collect() };
    let at512 = 12 * 512 * 512;
    if n_in > at512 && n_out < at512 {
        let mid = mean(x, n_in / at512);
        Ok(mean(&mid, at512 / n_out))
    } else {
        Ok(mean(x, n_in / n_out))
    }
}

/// The calibration from the physical reference maps (already at the
/// target nside, `f32` values) and the SC statistics of the plan:
/// `stats(map) -> (values, spatial variances)` (the engine, `moments`).
pub fn calibrate(
    physical: &[Vec<f64>],
    names: Vec<String>,
    ids: Vec<usize>,
    plan: &[Coefficient],
    scales: usize,
    nbins: usize,
    stats: impl FnMut(&[f64]) -> (Vec<C64>, Vec<f64>),
) -> Result<Calibration, String> {
    calibrate_mode(physical, names, ids, plan, scales, nbins, false, false, stats)
}

/// The same physical targets, with centered references, no Gaussian
/// features or learned transform, and physical initial-field spectra.
pub fn calibrate_physical(
    physical: &[Vec<f64>],
    names: Vec<String>,
    ids: Vec<usize>,
    plan: &[Coefficient],
    scales: usize,
    stats: impl FnMut(&[f64]) -> (Vec<C64>, Vec<f64>),
) -> Result<Calibration, String> {
    calibrate_mode(physical, names, ids, plan, scales, 0, true, false, stats)
}

/// Centered physical SC targets only. The reference C_l is retained solely
/// to initialize Gaussian random fields; PS, PDF and MF targets are absent.
pub fn calibrate_physical_sc(
    physical: &[Vec<f64>],
    names: Vec<String>,
    ids: Vec<usize>,
    plan: &[Coefficient],
    scales: usize,
    stats: impl FnMut(&[f64]) -> (Vec<C64>, Vec<f64>),
) -> Result<Calibration, String> {
    calibrate_mode(physical, names, ids, plan, scales, 0, true, true, stats)
}

#[allow(clippy::too_many_arguments)]
fn calibrate_mode(
    physical: &[Vec<f64>],
    names: Vec<String>,
    ids: Vec<usize>,
    plan: &[Coefficient],
    scales: usize,
    nbins: usize,
    physical_only: bool,
    sc_only: bool,
    mut stats: impl FnMut(&[f64]) -> (Vec<C64>, Vec<f64>),
) -> Result<Calibration, String> {
    if physical.len() < 2 || ids.is_empty() || ids.iter().any(|&i| i >= physical.len()) {
        return Err("calibration: at least two references, bootstrap ids among them".into());
    }
    let npix = physical[0].len();
    let nside = ((npix / 12) as f64).sqrt() as usize;
    if physical.iter().any(|x| x.len() != npix || x.iter().any(|v| !v.is_finite())) {
        return Err("calibration: equally sized finite full-sky references required".into());
    }
    // The same monopole projection is used on targets and synthesized maps.
    let physical: std::borrow::Cow<'_, [Vec<f64>]> = if physical_only {
        std::borrow::Cow::Owned(physical.iter().map(|x| {
            let mean = x.iter().sum::<f64>() / npix as f64;
            x.iter().map(|v| v - mean).collect()
        }).collect())
    } else {
        std::borrow::Cow::Borrowed(physical)
    };
    let (lut, gauss) = if physical_only {
        // Empty storage only: physical mode never calls a LUT operation.
        (Lut { probabilities: vec![], physical: vec![], gaussian: vec![] }, vec![])
    } else {
        let pool: Vec<f64> = physical.iter().flatten().copied().collect();
        let lut = Lut::from_values(pool, nbins)?;
        let gauss: Vec<Vec<f64>> = physical.iter().map(|x| x.iter().map(|&v| lut.gaussianize(v)).collect()).collect();
        (lut, gauss)
    };
    let coords = ScCoordinates::new(plan, nside)?;
    let n = plan.len();
    let mut t = Targets::default();
    for (maps, physical_sc) in [(gauss.as_slice(), false), (physical.as_ref(), true)] {
        if physical_only && !physical_sc {
            continue;
        }
        let mut rows = Vec::new();
        let mut sigma = vec![0.0; n];
        for x in maps.iter() {
            let (v, s) = stats(x);
            let f = coords.forward(&v, n)?;
            coords.sigma(&s).iter().zip(sigma.iter_mut()).for_each(|(a, b)| *b += a);
            rows.push(f);
        }
        sigma.iter_mut().for_each(|v| *v /= maps.len() as f64);
        let (mean, var) = population(&rows, &ids);
        if physical_sc {
            (t.psc_mean, t.psc_var, t.psc_sigma) = (mean, var, sigma);
        } else {
            (t.gsc_mean, t.gsc_var, t.gsc_sigma) = (mean, var, sigma);
        }
    }
    let lmax = 3 * nside - 1;
    let sht = Sht::new(nside, lmax);
    let mut gcl = Vec::new();
    if sc_only {
        // Initialization only: no harmonic/PDF/MF loss targets are computed.
        for x in physical.iter() {
            gcl.push(sht.power(&sht.analysis(x)));
        }
    } else {
        let (windows, _) = lp_windows(lmax);
        let lo = physical.iter().flatten().fold(f64::INFINITY, |m, &v| m.min(v));
        let hi = physical.iter().flatten().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
        t.pdf_support = [lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo)];
        let cic = Cic::new(t.pdf_support[0], t.pdf_support[1], 100);
        let mk = Minkowski::default();
        let (mut power, mut pdf, mut mf) = (Vec::new(), Vec::new(), Vec::new());
        for (i, x) in physical.iter().enumerate() {
            let (bands, alm) = band_powers(&sht, &windows, x);
            power.push(bands);
            pdf.push(cic.forward(x));
            mf.push(mk.forward(&sht, x).0);
            gcl.push(if physical_only { sht.power(&alm) } else { sht.power(&sht.analysis(&gauss[i])) });
        }
        let re = |v: Vec<C64>| v.into_iter().map(|c| c.re).collect::<Vec<f64>>();
        let (m, v) = population(&power, &ids);
        (t.power_mean, t.power_var) = (re(m), v);
        let (m, v) = population(&pdf, &ids);
        (t.pdf_mean, t.pdf_var) = (re(m), v);
        let (m, v) = population(&mf, &ids);
        (t.mf_mean, t.mf_var) = (re(m), v);
    }
    t.initialization_cl = (0..=lmax).map(|l| ids.iter().map(|&k| gcl[k][l]).sum::<f64>() / ids.len() as f64).collect();
    t.initialization_reference_mean = if physical_only { 0.0 } else { gauss[ids[0]].iter().sum::<f64>() / npix as f64 };
    // Every variance the ratio terms read must be finite and positive: the
    // whole plan for g, S1..S4 for kappa.
    let s0 = |i: usize| matches!(plan[i].family, Family::Mean | Family::Var);
    let bad: Vec<String> = (0..n)
        .filter(|&i| !physical_only && !(t.gsc_var[i].is_finite() && t.gsc_var[i] > 0.0))
        .map(|i| format!("gsc[{i}] = {}", t.gsc_var[i]))
        .chain((0..n).filter(|&i| !s0(i) && !(t.psc_var[i].is_finite() && t.psc_var[i] > 0.0)).map(|i| format!("psc[{i}] = {}", t.psc_var[i])))
        .collect();
    if !bad.is_empty() {
        return Err(format!("calibration: SC variance targets must be finite and positive: {}", bad[..bad.len().min(8)].join(", ")));
    }
    Ok(Calibration {
        physical_only,
        sc_only,
        nside,
        scales,
        orientations: 4,
        kernel: 5,
        plan_len: n,
        references: names,
        bootstrap_ids: ids,
        lut,
        targets: t,
    })
}

fn cplx(v: &[C64]) -> Value {
    json!({"re": v.iter().map(|c| c.re).collect::<Vec<_>>(), "im": v.iter().map(|c| c.im).collect::<Vec<_>>()})
}

fn floats(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    v[key].as_array().ok_or(format!("calibration: no {key}"))?.iter().map(|x| x.as_f64().ok_or(format!("calibration: {key}"))).collect()
}

fn complexes(v: &Value, key: &str) -> Result<Vec<C64>, String> {
    let (re, im) = (floats(&v[key], "re")?, floats(&v[key], "im")?);
    Ok(re.into_iter().zip(im).map(|(a, b)| C64::new(a, b)).collect())
}

impl Calibration {
    /// JSON of schema `stl-ensemble-calibration-1`: configuration,
    /// references, bootstrap ids, LUT knots and targets.
    pub fn to_json(&self) -> Value {
        let t = &self.targets;
        json!({
            "schema": if self.sc_only { "stl-ensemble-physical-sc-calibration-1" } else if self.physical_only { "stl-ensemble-physical-calibration-1" } else { "stl-ensemble-calibration-1" },
            "method": if self.sc_only { "Physical-only v1.1.0 (2026-11-04), physical SC only, centered physical variable" } else if self.physical_only { "Physical-only v1.1.0 (2026-11-04), four blocks, centered physical variable" } else { "P. Campeti, STL_ensemble v2 (variance ratio), five blocks" },
            "physical_only": self.physical_only, "sc_only": self.sc_only,
            "nside": self.nside, "scales": self.scales, "orientations": self.orientations, "kernel": self.kernel,
            "gauge": "cosmo", "fewer": false, "plan_len": self.plan_len,
            "references": self.references, "bootstrap_ids": self.bootstrap_ids,
            "lut": if self.physical_only { Value::Null } else { json!({"probabilities": self.lut.probabilities, "physical_knots": self.lut.physical}) },
            "targets": {
                "gsc_mean": cplx(&t.gsc_mean), "gsc_var": t.gsc_var, "gsc_sigma": t.gsc_sigma,
                "psc_mean": cplx(&t.psc_mean), "psc_var": t.psc_var, "psc_sigma": t.psc_sigma,
                "power_mean": t.power_mean, "power_var": t.power_var,
                "pdf_mean": t.pdf_mean, "pdf_var": t.pdf_var, "mf_mean": t.mf_mean, "mf_var": t.mf_var,
                "pdf_support": t.pdf_support, "initialization_cl": t.initialization_cl,
                "initialization_reference_mean": t.initialization_reference_mean,
            },
        })
    }

    /// Reads a calibration written by [`Calibration::to_json`]; errors on
    /// another schema or a missing field.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let (physical_only, sc_only) = match v["schema"].as_str() {
            Some("stl-ensemble-calibration-1") => (false, false),
            Some("stl-ensemble-physical-calibration-1") => (true, false),
            Some("stl-ensemble-physical-sc-calibration-1") => (true, true),
            _ => return Err("not a stl-ensemble calibration".into()),
        };
        // Legacy calibrations did not include these flags; if present they
        // must agree with the schema, which fixes the objective semantics.
        for (key, expected) in [("physical_only", physical_only), ("sc_only", sc_only)] {
            if v.get(key).is_some_and(|flag| flag.as_bool() != Some(expected)) {
                return Err(format!("calibration: {key} disagrees with schema"));
            }
        }
        let u = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("calibration: no {k}"));
        let t = &v["targets"];
        let sup = floats(t, "pdf_support")?;
        Ok(Calibration {
            physical_only,
            sc_only,
            nside: u("nside")?,
            scales: u("scales")?,
            orientations: u("orientations")?,
            kernel: u("kernel")?,
            plan_len: u("plan_len")?,
            references: v["references"].as_array().map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default(),
            bootstrap_ids: v["bootstrap_ids"].as_array().ok_or("calibration: no bootstrap_ids")?.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect(),
            lut: if physical_only { Lut { probabilities: vec![], physical: vec![], gaussian: vec![] } }
                 else { Lut::new(floats(&v["lut"], "probabilities")?, floats(&v["lut"], "physical_knots")?)? },
            targets: Targets {
                gsc_mean: complexes(t, "gsc_mean")?,
                gsc_var: floats(t, "gsc_var")?,
                gsc_sigma: floats(t, "gsc_sigma")?,
                psc_mean: complexes(t, "psc_mean")?,
                psc_var: floats(t, "psc_var")?,
                psc_sigma: floats(t, "psc_sigma")?,
                power_mean: floats(t, "power_mean")?,
                power_var: floats(t, "power_var")?,
                pdf_mean: floats(t, "pdf_mean")?,
                pdf_var: floats(t, "pdf_var")?,
                mf_mean: floats(t, "mf_mean")?,
                mf_var: floats(t, "mf_var")?,
                pdf_support: [sup[0], sup[1]],
                initialization_cl: floats(t, "initialization_cl")?,
                initialization_reference_mean: t["initialization_reference_mean"].as_f64().ok_or("calibration: initialization_reference_mean")?,
            },
        })
    }
}
