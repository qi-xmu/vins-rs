//! 估计器
//! 通过特征点和IMU数据估计相机位姿

mod image_frame;

use anyhow::Result;
use std::collections::{HashMap, VecDeque};

use crate::config::*;
use crate::feature_manager::FeatureManager;
use crate::feature_trakcer::FeatureFrame;
use crate::{camera::CameraTrait, feature_trakcer::FeatureTracker};

use opencv::core::Mat;

use self::image_frame::ImageFrame;

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
    /// 输入图像计数
    pub input_image_cnt: i32,

    pub prev_time: f64,
    pub cur_time: f64,
    /// td 校准时间
    pub td: f64,
    /// 初始化时间
    pub initial_timestamp: f64,

    // IMU buffer
    pub acce_buf: VecDeque<(f64, [f64; 3])>,
    pub gyro_buf: VecDeque<(f64, [f64; 3])>,

    /// 帧计数
    pub frame_count: i32,

    /* ? 窗口 是否可以合并成一个窗口 */
    /// 时间戳窗口
    pub timestamps: VecDeque<f64>,
    /// 时间间隔窗口
    pub diff_times: VecDeque<f64>,
    /// ? 图像窗口, 缓冲的必要性？
    pub images: VecDeque<Mat>,
    /// 旋转矩阵窗口
    pub rot_mats: VecDeque<nalgebra::Matrix3<f64>>,
    /// 平移向量窗口
    pub trans_vecs: VecDeque<nalgebra::Vector3<f64>>,
    /// 速度向量窗口
    pub vel_vecs: VecDeque<nalgebra::Vector3<f64>>,

    /// 加速度窗口
    pub acce_vecs: VecDeque<nalgebra::Vector3<f64>>,
    /// 角速度窗口
    pub gyro_vecs: VecDeque<nalgebra::Vector3<f64>>,
    /// 加速度偏置窗口
    pub bias_acces: VecDeque<nalgebra::Vector3<f64>>,
    /// 角速度偏置窗口
    pub bias_gyros: VecDeque<nalgebra::Vector3<f64>>,

    pub t_image_frame_map: HashMap<i64, ImageFrame>,

    /* 特征 */
    /// 提取特征点
    #[deprecated]
    feature_tracker: FeatureTracker<Camera>,
    /// 每一帧的特征点缓冲：包括时间戳，图像，该图像所有特征点。
    feature_frame_buf: VecDeque<FeatureFrame>,
    /// 特征管理器，提取每一帧的特征点，按照时间顺序管理特征点。
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
        let _t = timestamp;
        let _f = feature_frame;
        let _a = &feature_frame.image;
        let feature_frame = feature_frame.clone();
        //
        self.feature_frame_buf.push_back(feature_frame);
        // self.process_measurements();
        Ok(())
    }

    #[allow(dead_code)]
    #[deprecated = "use input_feature instead"]
    pub fn input_image(&mut self, timestamp: f64, img: &Mat) {
        // self.input_image_cnt += 1;
        // let feature_frame = self.feature_tracker.track_image(timestamp, &img);
        // let nimg = img.clone();
        // let img_track = self.feature_tracker.get_track_image().clone();

        if MULTIPLE_THREAD {
            // self.feature_frame_buf.push_back(feature_frame);
            // self.process_measurements();
        } else {
            // self.feature_frame_buf.push_back(feature_frame);
            // self.process_measurements();
        }
    }

    pub fn slide_window(&mut self) {
        //
        match self.marginalization_flag {
            MarginalizationFlag::MarginOld => {
                // ? Headers back_R0 back_P0
                if self.frame_count == WINDOW_SIZE {
                    let t_front = *self.timestamps.front().unwrap() as i64;
                    match self.solver_flag {
                        SolverFlag::Initial => {
                            // TODO: all_image_frame
                            // let a = self.t_image_frame_map;
                            // let a = self.t_image_frame_map.get(&t_front).unwrap();
                            self.t_image_frame_map.remove(&t_front);
                            // self.t_image_frame_map.re
                        }
                        _ => {}
                    }
                    //
                    self.timestamps.pop_front();
                    self.images.pop_front();
                    self.rot_mats.pop_front();
                    self.trans_vecs.pop_front();
                    // TODO: USE_IMU
                }
            }
            MarginalizationFlag::MarginSecondNew => {
                //
            }
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
