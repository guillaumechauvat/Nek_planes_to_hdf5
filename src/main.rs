extern crate hdf5;
extern crate clap;
extern crate ndarray;

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::path::{Path, PathBuf};
use clap::{Arg, App};
use ndarray::{Array, Array2, Array3};

#[derive(PartialEq, Eq)]
enum Verbosity {None, Info, Debug}

struct Planes {
    nplanes: usize,
    nw: usize,
    nh: usize,
    z: Array2<f64>,
    h: Array2<f64>,
    u: Array3<f64>,
    v: Array3<f64>,
    w: Array3<f64>,
    p: Array3<f64>
}

fn main() {
    let matches = App::new("Nek planes to HDF5")
                      .version("0.1.0")
                      .author("Guillaume Chauvat <guillaume@chauvat.eu>")
                      .about("Converts binay data interpolated on planes from Nek5000 to HDF5 format")
                      .arg(Arg::with_name("INPUT")
                          .help("Sets the input directory to use")
                          .index(1))
                      .arg(Arg::with_name("v")
                          .short("v")
                          .multiple(true)
                          .help("Sets the level of verbosity"))
                      .get_matches();

    let directory = if let Some(input) = matches.value_of("INPUT") {
        Path::new(input)
    } else {
        Path::new("./")
    };
    let verbosity = match matches.occurrences_of("v") {
        0 => Verbosity::None,
        1 => Verbosity::Info,
        _ => Verbosity::Debug,
    };

    // loop over files as long as they exist
    for i in 0.. {
        let filename = format!("uint_{:08}.dat", i);
        let filename = directory.join(filename);
        if verbosity != Verbosity::None {
            println!("reading {}", filename.to_str().unwrap());
        }

        // open the file; return if not found
        if let Ok(file) = File::open(filename) {
            // read header
            let mut buf_reader = BufReader::new(file);
            let mut header_buf = [0; 4];
            // read bytes 4 at a time
            buf_reader.read_exact(&mut header_buf).unwrap();
            let endian = f32::from_ne_bytes(header_buf);
            buf_reader.read_exact(&mut header_buf).unwrap();
            let nplanes = u32::from_ne_bytes(header_buf) as usize;
            buf_reader.read_exact(&mut header_buf).unwrap();
            let nw = u32::from_ne_bytes(header_buf) as usize;
            buf_reader.read_exact(&mut header_buf).unwrap();
            let nh = u32::from_ne_bytes(header_buf) as usize;
            if verbosity == Verbosity::Debug {
                println!("endian: {}, {} planes, {}×{} points", endian, nplanes, nw, nh);
            }
            // now read the data
            let mut z = Array::zeros((nw, nh));
            let mut h = Array::zeros((nw, nh));
            let mut u = Array::zeros((nplanes, nw, nh));
            let mut v = Array::zeros((nplanes, nw, nh));
            let mut w = Array::zeros((nplanes, nw, nh));
            let mut p = Array::zeros((nplanes, nw, nh));

            // read geometry
            read_plane_geom(&mut z, &mut buf_reader, nw, nh);
            read_plane_geom(&mut h, &mut buf_reader, nw, nh);

            // read flow
            for plane in 0..nplanes {
                if verbosity == Verbosity::Debug {
                    println!("reading plane {}", plane);
                }
                read_plane(&mut u, &mut buf_reader, plane, nw, nh);
                read_plane(&mut v, &mut buf_reader, plane, nw, nh);
                read_plane(&mut w, &mut buf_reader, plane, nw, nh);
                read_plane(&mut p, &mut buf_reader, plane, nw, nh);
            }

            let planes = Planes {
                nplanes,
                nw,
                nh,
                z,
                h,
                u,
                v,
                w,
                p
            };

            // write results
            let filename = format!("uint_{:08}.h5", i);
            let filename = directory.join(filename);
            save_hdf5(&filename, &planes).unwrap();
        } else {
            return;
        }
    }

}

fn read_plane(data: &mut Array3<f64>, buf_reader: &mut BufReader<File>, plane: usize, nw: usize, nh: usize) {
    for iw in 0..nw {
        for ih in 0..nh {
            let mut buffer = [0; 8];
            match buf_reader.read_exact(&mut buffer) {
                Ok(_) => {
                    data[[plane, iw, ih]] = f64::from_ne_bytes(buffer);
                    //println!("{}: {:.9e}", idx, data[idx]);
                },
                Err(e) => {
                    eprintln!("error reading flow at plane {}, point {}, {}: {}", plane, iw, ih, e);
                }
            }
        }
    }
}

fn read_plane_geom(data: &mut Array2<f64>, buf_reader: &mut BufReader<File>, nw: usize, nh: usize) {
    for iw in 0..nw {
        for ih in 0..nh {
            let mut buffer = [0; 8];
            match buf_reader.read_exact(&mut buffer) {
                Ok(_) => {
                    data[[iw, ih]] = f64::from_ne_bytes(buffer);
                },
                Err(e) => {
                    eprintln!("error reading geometry at point {}, {}: {}", iw, ih, e);
                }
            }
        }
    }
}

fn save_hdf5(filename: &PathBuf, planes: &Planes) -> Result<(), hdf5::Error> {
    let file = hdf5::File::create(filename).unwrap();

    // create separate groups for geometry and flow params
    let geom = file.create_group("geometry")?;
    let flow = file.create_group("flow")?;
    let (nplanes, nw, nh) = (planes.nplanes, planes.nw, planes.nh);

    // write geometry
    let z = geom.new_dataset::<f64>().create("z", (nw, nh))?;
    let h = geom.new_dataset::<f64>().create("h", (nw, nh))?;
    z.write(&planes.z)?;
    h.write(&planes.h)?;

    // write flow
    let u = flow.new_dataset::<f64>().create("u", (nplanes, nw, nh))?;
    let v = flow.new_dataset::<f64>().create("v", (nplanes, nw, nh))?;
    let w = flow.new_dataset::<f64>().create("w", (nplanes, nw, nh))?;
    let p = flow.new_dataset::<f64>().create("p", (nplanes, nw, nh))?;
    u.write(&planes.u)?;
    v.write(&planes.v)?;
    w.write(&planes.w)?;
    p.write(&planes.p)?;
    Ok(())
}
