//! `stl-rotation`: the orientation-aligned frame of a reference population
//! (ADR 0002).
//!
//! ```text
//! stl-rotation --out runs/frame_T -- ref_000.npy ref_001.npy ...
//! ```
//!
//! Writes, in `--out`, `angle_dg<d>.npy` (the local preferred direction of
//! every pixel of level `d`, radians in `[0, pi)`, NESTED order) and
//! `concentration_dg<d>.npy` (how much the references agree on it, `0..1`),
//! then `rotation.json`; the run reads the same directory with `--rotation`.

use std::path::PathBuf;
use stl_cli::rotation::{self, Frame, FrameOptions};
use stl_mpi::io;

const USAGE: &str = "\
stl-rotation --out DIR -- MAP...

  --out DIR             directory written (angle_dg<d>.npy, concentration_dg<d>.npy,
                        rotation.json)
  --L N                 orientations of the run (default 4)
  --kernel-size N       stencil of the run (default 5)
  --smooth N            decimations of the amplitude profile before the direction, so that
                        the preferred direction is a large-scale property (default 2)
  --block-pixels N      block of the full-sky wavelet (default 16384)
  -- MAP...             reference maps (NESTED .npy / .fits); the nside is the first map's
";

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some(split) = argv.iter().position(|a| a == "--") else {
        eprintln!("{USAGE}");
        std::process::exit(2);
    };
    let maps: Vec<PathBuf> = argv[split + 1..].iter().map(PathBuf::from).collect();
    let mut out: Option<PathBuf> = None;
    let mut options = FrameOptions::default();
    let mut it = argv[..split].iter();
    while let Some(k) = it.next() {
        let num = |v: Option<&String>| -> usize { v.expect("value").parse().expect("integer") };
        match k.as_str() {
            "--out" => match it.next() {
                Some(v) => out = Some(PathBuf::from(v)),
                None => fail("--out needs a directory"),
            },
            "--L" => options.l = num(it.next()),
            "--kernel-size" => options.kernel_size = num(it.next()),
            "--smooth" => options.smooth = num(it.next()),
            "--block-pixels" => options.block_pixels = num(it.next()),
            "-h" | "--help" => {
                eprintln!("{USAGE}");
                return;
            }
            other => fail(&format!("unknown option {other}")),
        }
    }
    let Some(out) = out else { fail("--out is required") };
    if maps.is_empty() {
        fail("at least one reference map is required after --");
    }
    if options.l < 2 {
        fail("--L must be at least 2");
    }
    let mut first = Some(read(&maps[0]));
    let npix = first.as_ref().unwrap().len();
    let nside = match io::nside_of(npix) {
        Ok(n) => n,
        Err(e) => fail(&format!("{}: {e}", maps[0].display())),
    };
    let started = std::time::Instant::now();
    let mut frame = Frame::new(nside, &options);
    for (i, p) in maps.iter().enumerate() {
        let x = match first.take() {
            Some(x) => x,
            None => read(p),
        };
        if x.len() != npix {
            fail(&format!("{}: {} pixels, the first map has {}", p.display(), x.len(), npix));
        }
        frame.push(&x);
        println!("[stl-rotation] {}/{} references", i + 1, maps.len());
    }
    if let Err(e) = rotation::write(&out, nside, &options, &maps, &frame) {
        fail(&e);
    }
    let concentration = frame.concentration();
    let agreement = frame.split_half_agreement();
    for (dg, (c, a)) in concentration.iter().zip(&agreement).enumerate() {
        let mean = c.iter().sum::<f64>() / c.len().max(1) as f64;
        println!(
            "[stl-rotation] dg{dg} nside {}: anisotropy {mean:.3}, split-half agreement {a:.3}",
            nside >> dg
        );
    }
    println!("[stl-rotation] {} written in {:.1} s", out.display(), started.elapsed().as_secs_f64());
}

fn read(path: &std::path::Path) -> Vec<f64> {
    match io::read_map(path) {
        Ok(x) => x,
        Err(e) => fail(&format!("{}: {e}", path.display())),
    }
}

fn fail(message: &str) -> ! {
    eprintln!("stl-rotation: {message}\n\n{USAGE}");
    std::process::exit(2);
}
