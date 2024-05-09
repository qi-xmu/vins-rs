mod image_frame;

use std::collections::VecDeque;

use crate::config::*;
use crate::feature_trakcer::FeatureFrame;
use crate::{camera::CameraTrait, feature_trakcer::FeatureTracker};

use opencv::core::Mat;

#[derive(Debug, Default)]
pub struct Estimator<Camera>
where
    Camera: CameraTrait,
{
    pub input_image_cnt: i32,
    /// 输入图像计数
    pub prev_time: f64,
    pub cur_time: f64,
    // td 校准时间
    pub td: f64,

    // IMU buffer
    pub acce_buf: VecDeque<(f64, [f64; 3])>,
    pub gyro_buf: VecDeque<(f64, [f64; 3])>,

    // Frame
    pub frame_count: usize,
    pub images: [(Mat, Mat); (WINDOW_SIZE + 1) as usize],

    // FeatureTracker
    feature_tracker: FeatureTracker<Camera>,
    feature_frame_buf: VecDeque<FeatureFrame>,
}

impl<Camera> Estimator<Camera>
where
    Camera: CameraTrait,
{
    pub fn input_image(&mut self, timestamp: f64, img: &Mat) {
        self.input_image_cnt += 1;
        let nimg = img.clone();
        let feature_frame = self.feature_tracker.track_image(timestamp, &img);
        let img_track = self.feature_tracker.get_track_image().clone();

        if MULTIPLE_THREAD {
            self.feature_frame_buf.push_back(feature_frame);
            self.process_measurements();
        } else {
            // self.feature_frame_buf.push_back(feature_frame);
            // self.process_measurements();
        }
    }

    fn process_image(&mut self, frame: FeatureFrame, cur_img: &Mat, img_tracker: &Mat) {
        log::info!("process_image");
        //
        self.images[self.frame_count] = frame.image.clone();
        // TODO:addFeatureCheckParallax
    }

    #[inline]
    fn imu_available(&self, t: f64) -> bool {
        if !self.acce_buf.is_empty() && t <= self.acce_buf.back().unwrap().0 {
            true
        } else {
            false
        }
    }

    fn process_measurements(&mut self) {
        // TODO: process measurements
        loop {
            //
            if !self.feature_frame_buf.is_empty() {
                let feature = self.feature_frame_buf.pop_front().unwrap();
                let cur_time = feature.timestamp + self.td; // 校准时间
                loop {
                    // TODO: imu_available()
                    // 使用IMU等待IMU加载
                    if !USE_IMU || self.imu_available(cur_time) {
                        //
                        break;
                    } else {
                        log::info!("wait for imu data");
                        if !MULTIPLE_THREAD {
                            return;
                        }
                        // TODO sleep()
                    }
                }

                // TODO: USE_IMU getIMUInterval

                // TODO: USE_IMU initFirstIMUPose and processIMU

                // TODO: process_image()
                // self.process_image(feature.0, &feature.1, &feature.2);

                break;
            }
        }
    }
}
