//! 估计器
//! 通过特征点和IMU数据估计相机位姿

mod image_frame;

use anyhow::Result;
use std::collections::VecDeque;

use crate::config::*;
use crate::feature_manager::FeatureManager;
use crate::feature_trakcer::FeatureFrame;
use crate::{camera::CameraTrait, feature_trakcer::FeatureTracker};

use opencv::core::Mat;

#[allow(dead_code)]
#[derive(Debug, Default)]
enum MarginalizationFlag {
    /// MarginOld
    #[default]
    MarginOld,
    /// MARGIN_SECOND_NEW
    MarginSecondNew,
}

#[allow(dead_code)]
#[derive(Debug, Default)]
enum SolverFlag {
    /// NON_LINEAR
    NonLinear,
    /// LINEARIZE
    // Linearize,
    /// INITIAL
    #[default]
    Initial,
}

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
    pub frame_count: i32,
    pub images: [(Mat, Mat); (WINDOW_SIZE + 1) as usize],
    pub timestamps: [f64; (WINDOW_SIZE + 1) as usize],

    initial_timestamp: f64,

    // FeatureTracker
    feature_tracker: FeatureTracker<Camera>,
    feature_frame_buf: VecDeque<FeatureFrame>,

    // FeatureManager
    feature_manager: FeatureManager,

    // Flag
    marginalization_flag: MarginalizationFlag,
    solver_flag: SolverFlag,
}

impl<Camera> Estimator<Camera>
where
    Camera: CameraTrait,
{
    pub fn input_feature(&mut self, timestamp: f64, feature_frame: &FeatureFrame) -> Result<()> {
        Ok(())
    }

    #[allow(dead_code)]
    #[deprecated = "use input_feature instead"]
    pub fn input_image(&mut self, timestamp: f64, img: &Mat) {
        self.input_image_cnt += 1;
        let feature_frame = self.feature_tracker.track_image(timestamp, &img);
        // let nimg = img.clone();
        // let img_track = self.feature_tracker.get_track_image().clone();

        if MULTIPLE_THREAD {
            self.feature_frame_buf.push_back(feature_frame);
            self.process_measurements();
        } else {
            // self.feature_frame_buf.push_back(feature_frame);
            // self.process_measurements();
        }
    }

    pub fn slide_window(&mut self) {
        //
        match self.marginalization_flag {
            MarginalizationFlag::MarginOld => {
                //
                let t_0 = self.timestamps[0];
                // TODO:back_R0 and back_P0
            }
            _ => {}
        }
    }

    fn process_image(&mut self, frame: &FeatureFrame, timestamp: f64) {
        log::info!("process_image");
        // TODO self.images[self.frame_count as usize] = frame.image.clone();
        // self.images[self.frame_count as usize] = frame.image.clone();
        // TODO:addFeatureCheckParallax
        if self.feature_manager.add_feature_check_parallax(
            self.frame_count,
            &frame.point_features,
            self.td,
        ) {
            self.marginalization_flag = MarginalizationFlag::MarginOld;
        } else {
            self.marginalization_flag = MarginalizationFlag::MarginSecondNew;
        };

        self.timestamps[self.frame_count as usize] = timestamp;

        // TODO:ImageFrame
        let mut image_frame = image_frame::ImageFrame::new(timestamp, &frame.point_features);
        image_frame.pre_integration = image_frame::IntegrationBase::default(); // FIXME:tmp_pre_integration

        // TODO:all_image_frame and tmp_pre_integration

        // TODO:ESTIMATE_EXTRINSIC
        match self.solver_flag {
            // sadf
            SolverFlag::Initial => {
                // Initial
                // TODO:STEREO
                if USE_IMU {
                    //
                    let mut result = false;
                    if self.frame_count == WINDOW_SIZE {
                        //
                        if timestamp - self.initial_timestamp > 0.1 {
                            // TODO:initialStructure
                            self.initial_timestamp = timestamp;
                        }
                    }
                    if result {
                        // TODO:optimization
                        // TODO:updateLatestStates
                        self.solver_flag = SolverFlag::NonLinear;
                        // TODO:slideWindow
                    } else {
                        // TODO:slideWindow
                    }
                }

                //
                if self.frame_count < WINDOW_SIZE {
                    self.frame_count += 1;
                    let prev_frame_count = self.frame_count - 1;
                    // Ps[frame_count] = Ps[prev_frame];
                    // Vs[frame_count] = Vs[prev_frame];
                    // Rs[frame_count] = Rs[prev_frame];
                    // Bas[frame_count] = Bas[prev_frame];
                    // Bgs[frame_count] = Bgs[prev_frame];
                }
            }
            SolverFlag::NonLinear => {
                // NonLinear
                if !USE_IMU {
                    // TODO:self.feature_manager initFramePoseByPnP
                    // self.feature_manager;
                }
                // TODO:triangulate
                // TODO:optimization
                // TODO:outliersRejection
                // TODO:self.feature_manager removeOutlier
                if !MULTIPLE_THREAD {
                    // TODO:featureTracker.removeOutliers(removeIndex);?
                    // TODO:predictPtsInNextFrame
                }
                // TODO failureDetection

                if false {
                    // ? self.failure_occur = true;
                    // TODO:clearState
                    // TODO:setParameter
                    return;
                }
                // TODO:slideWindow

                // TODO:f_manager.removeFailures();
                // TODO:key_poses.clear();

                // TODO:updateLatestStates
            } // _ => {}
        }
        todo!()
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
                let frame = self.feature_frame_buf.pop_front().unwrap();
                let cur_time = frame.timestamp + self.td; // 校准时间
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

                self.process_image(&frame, frame.timestamp);

                break;
            }
        }
    }
}
