//! Regression checks for the optional four-block and SC-only physical objectives.
//! Synthetic nside=8 skies; no external fixtures or GPU are required.

use num_complex::Complex64;
use std::path::PathBuf;
use stl_cli::ensemble::calibration::{calibrate_physical, calibrate_physical_sc, Calibration};
use stl_cli::ensemble::cli::{run_synthesis, setup, Args};
use stl_cli::ensemble::objective::EnsembleObjective;
use stl_core::sht::Sht;
use stl_mpi::synthesis::Objective;

const NSIDE: usize = 8;
const MEMBERS: usize = 3;

fn references() -> Vec<Vec<f64>> {
    (0..4)
        .map(|r| {
            (0..12 * NSIDE * NSIDE)
                .map(|i| {
                    let g = stl_cli::pixel_normal(735 + r as u64, i as i64);
                    (0.017 + 0.003 * r as f64) * (g + 0.12 * (g * g - 1.0)) + 0.1 * r as f64
                })
                .collect()
        })
        .collect()
}

fn calibration() -> Calibration {
    calibration_mode(false)
}

fn calibration_mode(sc_only: bool) -> Calibration {
    let refs = references();
    let scales = NSIDE.ilog2() as usize + 1;
    let (geo, cfg, plan) = setup(NSIDE, scales, 256);
    stl_mpi::threads::run_ranks(1, |comm| {
        let dummy = vec![Complex64::default(); plan.len()];
        let probe = Objective::new(&geo, &cfg, &plan, &dummy, 1, 1, comm);
        let mut calls = 0;
        let stats = |x: &[f64]| {
            calls += 1;
            assert!(x.iter().sum::<f64>().abs() / x.len() as f64 <= 1e-14, "uncentered reference");
            probe.statistics_moments(&probe.slice(x))
        };
        let cal = if sc_only {
            calibrate_physical_sc(&refs, vec![], vec![0, 2, 3], &plan, scales, stats)
        } else {
            calibrate_physical(&refs, vec![], vec![0, 2, 3], &plan, scales, stats)
        }.unwrap();
        assert_eq!(calls, refs.len(), "only physical SC may be calibrated");
        cal
    })
    .remove(0)
}

#[test]
fn physical_calibration_has_no_lut_or_gaussian_targets() {
    let cal = calibration();
    assert!(cal.physical_only);
    assert!(!cal.sc_only);
    assert!(cal.lut.probabilities.is_empty() && cal.lut.physical.is_empty() && cal.lut.gaussian.is_empty());
    assert!(cal.targets.gsc_mean.is_empty() && cal.targets.gsc_var.is_empty() && cal.targets.gsc_sigma.is_empty());
    assert_eq!(cal.targets.initialization_reference_mean, 0.0);
    let value = cal.to_json();
    assert_eq!(value["schema"], "stl-ensemble-physical-calibration-1");
    assert!(value["lut"].is_null());
    let restored = Calibration::from_json(&value).unwrap();
    assert!(restored.physical_only);
    assert_eq!(restored.to_json(), value);
    let mut wrong_schema = value.clone();
    wrong_schema["schema"] = "stl-ensemble-calibration-1".into();
    assert!(Calibration::from_json(&wrong_schema).is_err(), "legacy mode must require its LUT");

    // The initial random fields must use the physical references' spectrum.
    let sht = Sht::new(NSIDE, 3 * NSIDE - 1);
    let spectra: Vec<Vec<f64>> = references()
        .iter()
        .map(|x| {
            let mean = x.iter().sum::<f64>() / x.len() as f64;
            let centered: Vec<f64> = x.iter().map(|v| v - mean).collect();
            sht.power(&sht.analysis(&centered))
        })
        .collect();
    for (l, &actual) in cal.targets.initialization_cl.iter().enumerate() {
        let expected = [0, 2, 3].iter().map(|&r| spectra[r][l]).sum::<f64>() / MEMBERS as f64;
        assert!((actual - expected).abs() <= 1e-13 * expected.abs().max(1e-30));
    }
}

#[test]
fn physical_objective_projects_maps_and_uses_four_blocks() {
    let cal = calibration();
    let (geo, cfg, plan) = setup(NSIDE, cal.scales, 256);
    let npix = 12 * NSIDE * NSIDE;
    let x: Vec<f64> = references().into_iter().take(MEMBERS).flatten().collect();
    stl_mpi::threads::run_ranks(1, |comm| {
        let dummy = vec![Complex64::default(); MEMBERS * plan.len()];
        let og = Objective::new(&geo, &cfg, &plan, &dummy, MEMBERS, 1, comm);
        let ok = Objective::with_partition(&geo, &cfg, &plan, &dummy, MEMBERS, 1, comm, og.dec.clone(), og.part.clone());
        let obj = EnsembleObjective::new(&og, &ok, None, &cal, &plan, MEMBERS, 0, (3 * NSIDE - 1) as f64).unwrap();
        let (k, slope) = obj.physical(&x);
        assert!(slope.iter().all(|&v| v == 1.0));
        for (input, output) in x.chunks(npix).zip(k.chunks(npix)) {
            let mean = input.iter().sum::<f64>() / npix as f64;
            assert!(input.iter().zip(output).all(|(a, b)| (a - mean - b).abs() <= 1e-15));
        }
        let (loss, gradient) = obj.loss_and_grad(&x);
        assert!(loss.is_finite() && gradient.iter().all(|v| v.is_finite()));
        let blocks = *obj.last.borrow();
        assert_eq!(blocks[0].loss, 0.0, "Gaussian SC is inactive");
        assert!(!obj.timers.borrow().contains_key("gaussian_sc"), "Gaussian SC must not execute");
        let expected = blocks[1..].iter().map(|b| b.loss).sum::<f64>() / 4.0;
        assert!((loss - expected).abs() <= 1e-14 * expected.abs().max(1.0));
        for row in gradient.chunks(npix) {
            assert!(row.iter().sum::<f64>().abs() <= 1e-12 * row.iter().map(|v| v.abs()).sum::<f64>().max(1.0), "monopole cotangent is not projected");
        }
        let shifted: Vec<f64> = x.iter().enumerate().map(|(i, v)| v + 0.031 * (1 + i / npix) as f64).collect();
        let (shifted_loss, shifted_gradient) = obj.loss_and_grad(&shifted);
        assert!((shifted_loss - loss).abs() <= 1e-9 * loss.abs().max(1.0));
        let scale = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
        assert!(gradient.iter().zip(&shifted_gradient).all(|(a, b)| (a - b).abs() <= 1e-8 * scale));

        // Per-member radial direction: checks the physical chain and /4 scaling.
        // The inherited MF cotangent detaches its curvature floor, and CIC is
        // piecewise differentiable. Retain the upstream 1% full-objective FD
        // tolerance; this does not claim an exact derivative of that MF floor.
        let direction: Vec<f64> = k.iter().enumerate().map(|(i, v)| v * (1.0 + 0.1 * (i / npix) as f64)).collect();
        let h = 1e-6;
        let at = |step: f64| obj.loss_and_grad(&x.iter().zip(&direction).map(|(a, b)| a + step * b).collect::<Vec<_>>()).0;
        let numeric = (at(h) - at(-h)) / (2.0 * h);
        let analytic: f64 = gradient.iter().zip(&direction).map(|(a, b)| a * b).sum();
        assert!((analytic - numeric).abs() <= 1e-2 * numeric.abs().max(1e-8), "physical directional derivative: {analytic:e} vs {numeric:e}");
    });
}

#[test]
fn physical_cli_writes_only_physical_maps_and_rejects_mode_mismatch() {
    let cal = calibration();
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("stl_physical_only_{}_{unique}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("cal.json");
    std::fs::write(&path, serde_json::to_string(&cal.to_json()).unwrap()).unwrap();
    let prefix: PathBuf = dir.join("ens");
    let argv: Vec<String> = [
        "run", "--physical-only", "--calibration", path.to_str().unwrap(),
        "--ensemble", "3", "--groups", "1", "--niter", "1", "--max-eval", "2",
        "--checkpoint", "1", "--ps-variance-upper", "23", "--block-pixels", "256",
        "--out", prefix.to_str().unwrap(), "--quiet",
    ]
    .iter().map(|v| v.to_string()).collect();
    let args = Args::parse(&argv).unwrap();
    stl_mpi::threads::run_ranks(1, |comm| {
        let mut wrong = args.clone();
        wrong.physical_only = false;
        assert!(run_synthesis(&wrong, comm).is_err(), "a physical calibration needs explicit physical mode");
        let report = run_synthesis(&args, comm).unwrap().unwrap();
        assert_eq!(report["blocks"].as_array().unwrap().len(), 4);
        assert_eq!(report["blocks_final"].as_array().unwrap().len(), 4);
        for m in 0..MEMBERS {
            for tag in ["", "_checkpoint"] {
                let map = dir.join(format!("ens{tag}_m{m:03}_physical_nested.npy"));
                assert!(map.is_file(), "missing {}", map.display());
                let values = stl_mpi::io::read_npy(&map).unwrap();
                assert!(values.iter().sum::<f64>().abs() / values.len() as f64 <= 1e-14);
            }
        }
        let checkpoint: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ens_checkpoint.json")).unwrap()).unwrap();
        assert!(checkpoint["restart"].as_str().unwrap().contains("physical_nested.npy"));
        assert!(std::fs::read_dir(&dir).unwrap().all(|f| !f.unwrap().file_name().to_string_lossy().contains("gaussian")));
    });
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sc_only_calibration_keeps_sc_and_initialization_but_no_other_targets() {
    let cal = calibration_mode(true);
    assert!(cal.physical_only && cal.sc_only);
    let t = &cal.targets;
    for values in [&t.gsc_var, &t.gsc_sigma, &t.power_mean, &t.power_var,
                   &t.pdf_mean, &t.pdf_var, &t.mf_mean, &t.mf_var] {
        assert!(values.is_empty(), "inactive loss targets must not be computed");
    }
    assert!(t.gsc_mean.is_empty() && cal.lut.physical.is_empty());
    let four = calibration();
    assert_eq!(t.psc_mean, four.targets.psc_mean);
    assert_eq!(t.psc_var, four.targets.psc_var);
    assert_eq!(t.psc_sigma, four.targets.psc_sigma);
    assert_eq!(t.initialization_cl, four.targets.initialization_cl);
    let value = cal.to_json();
    assert_eq!(value["schema"], "stl-ensemble-physical-sc-calibration-1");
    assert!(value["lut"].is_null());
    assert_eq!(Calibration::from_json(&value).unwrap().to_json(), value);
    for key in ["physical_only", "sc_only"] {
        let mut wrong = value.clone();
        wrong[key] = false.into();
        assert!(Calibration::from_json(&wrong).is_err(), "mode metadata must agree with schema");
    }
    let mut wrong = value.clone();
    wrong["schema"] = "stl-ensemble-physical-calibration-1".into();
    assert!(Calibration::from_json(&wrong).is_err());
}

#[test]
fn sc_only_objective_skips_other_blocks_and_has_projected_gradient() {
    let cal = calibration_mode(true);
    let (geo, cfg, plan) = setup(NSIDE, cal.scales, 256);
    let npix = 12 * NSIDE * NSIDE;
    let x: Vec<f64> = references().into_iter().take(MEMBERS).flatten().collect();
    let mut serial = None;
    // Thread ranks exercise the same within-group reductions as MPI ranks.
    for ranks in [1, 2] {
        let outputs = stl_mpi::threads::run_ranks(ranks, |comm| {
            let dummy = vec![Complex64::default(); MEMBERS * plan.len()];
            let og = Objective::new(&geo, &cfg, &plan, &dummy, MEMBERS, 1, comm);
            let ok = Objective::with_partition(&geo, &cfg, &plan, &dummy, MEMBERS, 1, comm, og.dec.clone(), og.part.clone());
            // A power cutoff is not required when the power block is disabled.
            let obj = EnsembleObjective::new(&og, &ok, None, &cal, &plan, MEMBERS, 0, 0.0).unwrap();
            assert!(!obj.device_terms());
            let local_x = og.slice(&x);
            let (loss, gradient) = obj.loss_and_grad(&local_x);
            assert!(loss.is_finite() && gradient.iter().all(|v| v.is_finite()));
            let blocks = *obj.last.borrow();
            assert_eq!(loss, blocks[1].loss, "SC-only loss must not be divided by four");
            for i in [0, 2, 3, 4] {
                assert_eq!(blocks[i].loss, 0.0, "inactive block {i}");
            }
            for name in ["gaussian_sc", "power_pdf_forward", "power_pdf_reverse", "mf_forward", "mf_reverse"] {
                assert!(!obj.timers.borrow().contains_key(name), "inactive phase {name} executed");
            }
            let full_grad = obj.whole(&gradient).into_owned();
            for row in full_grad.chunks(npix) {
                assert!(row.iter().sum::<f64>().abs() <= 1e-12 * row.iter().map(|v| v.abs()).sum::<f64>().max(1.0));
            }
            if ranks == 1 {
                let (k, _) = obj.physical(&x);
                let direction: Vec<f64> = k.iter().enumerate().map(|(i, v)| v * (1.0 + 0.1 * (i / npix) as f64)).collect();
                let h = 1e-6;
                let at = |step: f64| obj.loss_and_grad(&x.iter().zip(&direction).map(|(a, b)| a + step * b).collect::<Vec<_>>()).0;
                let numeric = (at(h) - at(-h)) / (2.0 * h);
                let analytic: f64 = gradient.iter().zip(&direction).map(|(a, b)| a * b).sum();
                assert!((analytic - numeric).abs() <= 1e-5 * numeric.abs().max(1e-8), "SC-only derivative: {analytic:e} vs {numeric:e}");
                let shifted: Vec<f64> = x.iter().enumerate().map(|(i, v)| v + 0.031 * (1 + i / npix) as f64).collect();
                let (other_loss, other_gradient) = obj.loss_and_grad(&shifted);
                assert!((loss - other_loss).abs() <= 1e-9 * loss.abs().max(1.0));
                let scale = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
                assert!(gradient.iter().zip(&other_gradient).all(|(a, b)| (a - b).abs() <= 1e-8 * scale));
            }
            (loss, full_grad)
        });
        for result in &outputs {
            if let Some((ref_loss, ref_gradient)) = &serial {
                let ref_loss: &f64 = ref_loss;
                let ref_gradient: &Vec<f64> = ref_gradient;
                assert!((result.0 - ref_loss).abs() <= 1e-10 * ref_loss.abs().max(1.0));
                let scale = ref_gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs())).max(1.0);
                assert!(result.1.iter().zip(ref_gradient).all(|(a, b)| (a - b).abs() <= 1e-9 * scale));
            }
        }
        if ranks == 1 {
            serial = outputs.into_iter().next();
        }
    }
}

#[test]
fn sc_only_cli_requires_matching_calibration_and_reports_one_block() {
    let cal = calibration_mode(true);
    let unique = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
    let dir = std::env::temp_dir().join(format!("stl_sc_only_{}_{unique}", std::process::id()));
    std::fs::create_dir(&dir).unwrap();
    let path = dir.join("cal.json");
    std::fs::write(&path, serde_json::to_string(&cal.to_json()).unwrap()).unwrap();
    let prefix = dir.join("ens");
    let argv: Vec<String> = [
        "run", "--physical-only", "--sc-only", "--calibration", path.to_str().unwrap(),
        "--ensemble", "3", "--groups", "1", "--niter", "1", "--max-eval", "2",
        "--checkpoint", "1", "--block-pixels", "256", "--out", prefix.to_str().unwrap(), "--quiet",
    ].iter().map(|v| v.to_string()).collect();
    let args = Args::parse(&argv).unwrap();
    let no_physical: Vec<String> = argv.iter().filter(|v| v.as_str() != "--physical-only").cloned().collect();
    assert!(Args::parse(&no_physical).is_err());
    let mut ignored_harmonic = argv.clone();
    ignored_harmonic.extend(["--harmonic".into(), "gpu".into()]);
    assert!(Args::parse(&ignored_harmonic).is_ok(), "disabled harmonic terms must not require a GPU");
    let no_sc: Vec<String> = argv.iter().filter(|v| v.as_str() != "--sc-only").cloned().collect();
    let wrong = Args::parse(&no_sc).unwrap();
    stl_mpi::threads::run_ranks(1, |comm| {
        let error = run_synthesis(&wrong, comm).unwrap_err();
        assert!(error.contains("mode mismatch"));
        let report = run_synthesis(&args, comm).unwrap().unwrap();
        assert_eq!(report["sc_only"], true);
        assert_eq!(report["blocks"], serde_json::json!(["physical_sc"]));
        assert_eq!(report["blocks_final"].as_array().unwrap().len(), 1);
        assert_eq!(report["harmonic"], "disabled");
        assert!(report["ps_variance_upper"].is_null());
        for row in report["blocks_history"].as_array().unwrap() {
            assert_eq!(row.as_array().unwrap().len(), 1);
        }
        let checkpoint: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ens_checkpoint.json")).unwrap()).unwrap();
        assert_eq!(checkpoint["block_names"], serde_json::json!(["physical_sc"]));
        assert_eq!(checkpoint["sc_only"], true);
        let mut four = cal.clone();
        four.sc_only = false;
        std::fs::write(&path, serde_json::to_string(&four.to_json()).unwrap()).unwrap();
        assert!(run_synthesis(&args, comm).unwrap_err().contains("mode mismatch"));
    });
    std::fs::remove_dir_all(&dir).unwrap();
}
