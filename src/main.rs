/// opencv
/// https://docs.rs/opencv/latest/opencv/all.html
///
// image path /home/qi/V201/mav0/cam0/data/ **
extern crate opencv;

mod camera;
mod config;
mod feature_trakcer;

use std::io::BufRead;
use std::path::Path;

use camera::MyCamera;
use opencv::core::MatTraitConst;
use opencv::highgui;
use opencv::imgcodecs;

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
    let path = "/home/qi/V201/mav0/cam0/";
    let path = Path::new(path);
    let list = read_csv(path);

    let mut feature_tracker = feature_trakcer::FeatureTracker::<MyCamera>::new();

    const FREQUENCY: i32 = 30;
    let path = path.join("data");
    for (timestamp, name) in list {
        let path = path.join(name);
        let path = path.to_str().unwrap();

        let _img = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE).unwrap();
        if _img.empty() {
            println!("empty image");
            continue;
        }
        feature_tracker.track_image(timestamp, &_img);

        let img = feature_tracker.get_track_image().clone();

        // 拼接
        // let show_image = Mat::zeros(_img.rows(), _img.cols() )
        // opencv::core::hconcat2(, src2, dst)
        highgui::imshow("Raw", &_img).unwrap();
        highgui::imshow("Tracker", &img).unwrap();
        highgui::wait_key(1000 / FREQUENCY).unwrap();
    }
}
