//! VINS 解析
//! https://github.com/StevenCui/VIO-Doc
//!
//!
//! 使用的第三方库
//!
//! [opencv] 图像处理库
//! https://docs.rs/opencv/latest/opencv/all.html
//!
//! [nalgebra] 线性代数库
//!  https://docs.rs/nalgebra/latest/nalgebra/
//!
//! [ndarray] 数组库：暂时没用到
//! https://docs.rs/ndarray/latest/ndarray/all.html
//!
//! image path /home/qi/V201/mav0/cam0/data/ /Users/qi/Resources/Dataset/V201
//!

extern crate opencv;

mod camera;
mod config;
mod dataset;
mod feature_trakcer;
mod global_cast; // 全局转换
mod global_types;
mod pose_estimator;
mod save;

use opencv::core::Mat;
use opencv::highgui;
use opencv::imgcodecs;
use opencv::imgproc;

use crate::camera::PinholeCamera;
use crate::dataset::DatasetTrait;

fn main() {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_nanos()
        .init();

    let path = "/home/qi/V201";
    // let path = "/Users/qi/Resources/Dataset/V201";
    log::info!("path: {:?}", path);
    let dataset = dataset::DefaultDataset::new(path);

    let camera_file = "configs/cam0_pinhole.yaml";
    let camera = PinholeCamera::new(camera_file);
    let mut feature_tracker = feature_trakcer::FeatureTracker::new_with_camera(camera);
    let mut estimator = pose_estimator::Estimator::new();

    const FREQUENCY: i32 = 20;
    let mut img_convert = Mat::default();
    let mut img_show = Mat::default();
    for (timestamp, path) in dataset.read_t_cam0_list() {
        if let Ok(img) = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE) {
            let feature_frame = feature_tracker.track_image(*timestamp, &img);

            let vecs = nalgebra::Vector3::new(0.0, 0.0, 0.0);
            estimator.input_feature(&feature_frame).unwrap();
            estimator.input_imu(*timestamp, &vecs, &vecs).unwrap();

            let img_tracker = feature_tracker.get_track_image();
            opencv::imgproc::cvt_color(&img, &mut img_convert, imgproc::COLOR_GRAY2BGR, 0).unwrap();
            opencv::core::hconcat2(&img_convert, img_tracker, &mut img_show).unwrap();
            highgui::imshow("Raw Tracker", &img_show).unwrap();
            // highgui::wait_key(1000 / FREQUENCY).unwrap();
            if let Ok(key) = highgui::wait_key(1) {
                if key == 27 {
                    break;
                }
            };
        }
    }
}

#[test]
fn test_feature_tracker() {
    let path = "/Users/qi/Resources/Dataset/V201";
    let dataset = dataset::DefaultDataset::new(path);
    let camera_file = "configs/cam0_pinhole.yaml";
    let camera = PinholeCamera::new(camera_file);
    let mut feature_tracker = feature_trakcer::FeatureTracker::new_with_camera(camera);
    for (timestamp, path) in dataset.read_t_cam0_list() {
        let img = imgcodecs::imread(path, imgcodecs::IMREAD_GRAYSCALE).unwrap();
        let feature_frame = feature_tracker.track_image(*timestamp, &img);

        let fps: crate::save::pts::FramePointsSave = feature_frame.into();
        let ser_fps = serde_json::to_string(&fps).unwrap();
        // save pts
        let tmp_dir = "temp";

        // 文件夹是否存在
        if !std::path::Path::new(tmp_dir).exists() {
            std::fs::create_dir_all(tmp_dir).unwrap();
        }
        std::fs::write(format!("{}/{}.json", tmp_dir, (*timestamp / 1000)), ser_fps).unwrap();
    }
}
