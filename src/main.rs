/// opencv
/// https://docs.rs/opencv/latest/opencv/all.html
///
/// nalgebra
/// https://docs.rs/nalgebra/latest/nalgebra/
///
/// ndarray
/// https://docs.rs/ndarray/latest/ndarray/all.html
///
// image path /home/qi/V201/mav0/cam0/data/ **
extern crate opencv;

mod Estimator;
mod camera;
mod config;
mod estimator;
mod feature_trakcer; // 特征追踪

use std::io::BufRead;
use std::path::Path;

use opencv::core::MatTraitConst;
use opencv::highgui;
use opencv::imgcodecs;

use crate::camera::PinholeCamera;

fn read_csv(path: &Path) -> Vec<(f64, String)> {
    let path = path.join("data.csv");
    let mut result = Vec::new();
    let file = std::fs::File::open(path).unwrap();
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        if let Ok(line) = line {
            if line.starts_with("#") {
                continue;
            }
            let mut iter = line.split(',');
            if let (Some(timestamp), Some(path)) = (iter.next(), iter.next()) {
                result.push((timestamp.parse::<f64>().unwrap(), path.to_string()));
            }
        }
    }
    result
}

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_nanos()
        .init();

    let path = "/Users/qi/Resourse/Dataset/V201/mav0/cam0/";
    let path = Path::new(path);
    let list = read_csv(path);
    log::info!("path: {:?}", path);

    // let camera_file = "configs/cam0_pinhole.yaml";
    // let camera = camera::PinholeCamera::new(camera_file);
    // let mut feature_tracker = feature_trakcer::FeatureTracker::new_with_camera(camera);

    let mut estimator = estimator::Estimator::<PinholeCamera>::default();

    const FREQUENCY: i32 = 30;
    let path = path.join("data");
    for (timestamp, name) in list {
        let path = path.join(name);
        let path = path.to_str().unwrap();

        let _img = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE).unwrap();
        if _img.empty() {
            log::error!("empty image");
            continue;
        }
        // feature_tracker.track_image(timestamp, &_img);
        estimator.input_image(timestamp, &_img);

        // let img = feature_tracker.get_track_image().clone();

        // highgui::imshow("Raw", &_img).unwrap();
        // highgui::imshow("Tracker", &img).unwrap();
        highgui::wait_key(1000 / FREQUENCY).unwrap();
    }
}
