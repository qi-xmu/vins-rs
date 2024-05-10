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

mod camera;
mod config;
mod dataset;
mod estimator;
mod feature_manager;
mod feature_trakcer; // 特征追踪

use opencv::core::Mat;
use opencv::core::MatTraitConst;
use opencv::highgui;
use opencv::imgcodecs;
use opencv::imgproc::COLOR_GRAY2BGR;

use crate::camera::PinholeCamera;
use crate::dataset::DatasetTrait;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_nanos()
        .init();

    let path = "/home/qi/V201";
    log::info!("path: {:?}", path);
    let dataset = dataset::DefaultDataset::new(path);

    let camera_file = "configs/cam0_pinhole.yaml";
    let camera = PinholeCamera::new(camera_file);
    let mut feature_tracker = feature_trakcer::FeatureTracker::new_with_camera(camera);
    let mut estimator = estimator::Estimator::<PinholeCamera>::default();

    const FREQUENCY: i32 = 30;
    let mut img_convert = Mat::default();
    let mut img_show = Mat::default();
    for (timestamp, path) in dataset.read_t_cam0_list() {
        if let Ok(img) = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE) {
            let timestamp = *timestamp;
            feature_tracker.track_image(timestamp, &img);
            // estimator.input_image(timestamp, &img);

            let img_tracker = feature_tracker.get_track_image();
            opencv::imgproc::cvt_color(&img, &mut img_convert, COLOR_GRAY2BGR, 0).unwrap();
            opencv::core::hconcat2(&img_convert, img_tracker, &mut img_show).unwrap();
            highgui::imshow("Raw Tracker", &img_show).unwrap();
            highgui::wait_key(1000 / FREQUENCY).unwrap();
        }
    }
}
