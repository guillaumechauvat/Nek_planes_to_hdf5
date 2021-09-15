extern crate hdf5;
extern crate clap;

use std::io::prelude::*;
use std::io::BufReader;
use std::fs::File;
use std::path::Path;
use std::vec::Vec;
use clap::{Arg, App};

#[derive(PartialEq, Eq)]
enum Verbosity {None, Info, Debug}

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
            let npts = nplanes * nw * nh;
            let mut z = vec![0.; nw*nh];
            let mut h = vec![0.; nw*nh];
            let mut u = vec![0.; npts];
            let mut v = vec![0.; npts];
            let mut w = vec![0.; npts];
            let mut p = vec![0.; npts];

            // read geometry
            read_plane(&mut z, &mut buf_reader, 0, nw, nh);
            read_plane(&mut h, &mut buf_reader, 0, nw, nh);

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
            for i in 0..350 {
                println!("{:.9e}, {:.9e} | {:.9e}, {:.9e}, {:.9e} | {:.9e}", z[i], h[i], u[i], v[i], w[i], p[i]);
            }
            println!();
        } else {
            return;
        }
    }

}

fn read_plane(data: &mut Vec<f64>, buf_reader: &mut BufReader<File>, plane: usize, nw: usize, nh: usize) {
    for iw in 0..nw {
        for ih in 0..nh {
            let mut buffer = [0; 8];
            match buf_reader.read_exact(&mut buffer) {
                Ok(_) => {
                    let idx = ih + iw * nh + plane * nh * nw;
                    data[idx] = f64::from_ne_bytes(buffer);
                    //println!("{}: {:.9e}", idx, data[idx]);
                },
                Err(e) => {
                    eprintln!("error reading p at plane {}, point {}, {}: {}", plane, iw, ih, e);
                }
            }
        }
    }
}
