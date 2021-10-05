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

            // endianness support could be added later but this is probably only going to be used on little-endian.
            // Check anyway to make sure we don't write garbage.
            assert!(
                endian == 6.54321f32,
                "Wrong endianness bytes, expected 6.54321, found {}", endian
            );

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

    // add hardcoded x/c locations for reference
    let xc0 = vec![0.145, 0.15286, 0.16072, 0.16858, 0.17644, 0.18032, 0.1881, 0.1919, 0.19583, 0.198, 0.199, 0.1997, 0.201, 0.202, 0.203, 0.20359, 0.205, 0.2075, 0.2114, 0.2153, 0.2191, 0.223, 0.2269, 0.2308, 0.2386, 0.2464, 0.26476, 0.28312, 0.30148, 0.31984, 0.3382, 0.35656, 0.37492, 0.39328, 0.41164, 0.43, 0.44836, 0.46672, 0.48508];
    let x00 = vec![0.13445711, 0.14160301, 0.14874611, 0.15588668, 0.16302493, 0.16654776, 0.17360974, 0.17705812, 0.18062388, 0.18259251, 0.18349966, 0.18413464, 0.18531385, 0.18622089, 0.1871279, 0.18766303, 0.18894183, 0.19120906, 0.19474557, 0.19828162, 0.20172656, 0.20526172, 0.20879643, 0.2123307, 0.21939795, 0.22646352, 0.24308815, 0.25970264, 0.27630862, 0.29290621, 0.30949487, 0.32607525, 0.34264764, 0.35921075, 0.37576532, 0.39231124, 0.40884687, 0.42537378, 0.44189054];
    let y00 = vec![0.0706838, 0.07154527, 0.07236408, 0.07314439, 0.07388938, 0.07424376, 0.07492624, 0.07524568, 0.07556696, 0.07574062, 0.07581978, 0.07587488, 0.07597653, 0.07605414, 0.07613124, 0.0761765, 0.07628398, 0.07647224, 0.07676018, 0.07704117, 0.07730832, 0.07757574, 0.07783637, 0.07809029, 0.07857845, 0.0790412, 0.0800293, 0.08086329, 0.08156768, 0.08214457, 0.08258564, 0.08290063, 0.08309419, 0.08314658, 0.08306903, 0.08285979, 0.0824942, 0.08199589, 0.08134318];
    let nx00 = vec![-0.1226994, -0.11670294, -0.11115984, -0.10620119, -0.10134532, -0.09882103, -0.09351754, -0.09098475, -0.08852079, -0.08722797, -0.0866484, -0.08624879, -0.08551992, -0.084971, -0.08443229, -0.08411924, -0.08338549, -0.08212545, -0.08018047, -0.07824982, -0.07638247, -0.0744801, -0.07259254, -0.07073232, -0.06711011, -0.0636187, -0.05477735, -0.04590126, -0.03878684, -0.03058349, -0.02267676, -0.01542306, -0.00761873, 0.00107841, 0.00825644, 0.01748842, 0.02629739 , 0.03438209, 0.04467383];
    let ny00 = vec![0.99244388, 0.99316687, 0.99380254, 0.99434466, 0.99485131, 0.99510522, 0.99561763, 0.99585229, 0.99607433, 0.99618838, 0.99623895, 0.99627363, 0.99633646, 0.99638342, 0.99642922, 0.9964557, 0.99651737, 0.996622, 0.99678036, 0.99693378, 0.99707859, 0.9972225, 0.99736168, 0.99749533, 0.99774558, 0.99797428, 0.99849859, 0.99894598, 0.99924751, 0.99953222, 0.99974285, 0.99988106, 0.99997098, 0.99999942, 0.99996592, 0.99984707, 0.99965416, 0.99940876, 0.99900163];
    let xc = geom.new_dataset::<f64>().create("x_c", (xc0.len(),))?;
    let x0 = geom.new_dataset::<f64>().create("x0", (x00.len(),))?;
    let y0 = geom.new_dataset::<f64>().create("y0", (y00.len(),))?;
    let nx0 = geom.new_dataset::<f64>().create("nx0", (nx00.len(),))?;
    let ny0 = geom.new_dataset::<f64>().create("ny0", (ny00.len(),))?;
    xc.write(&xc0)?;
    x0.write(&x00)?;
    y0.write(&y00)?;
    nx0.write(&nx00)?;
    ny0.write(&ny00)?;

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
