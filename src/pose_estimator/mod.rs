//! 估计器
//! 通过特征点和IMU数据估计相机位姿

mod feature_manager;
mod image_frame;
mod sfm;

use anyhow::Result;
use nalgebra::AbstractRotation;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::feature_trakcer::FeatureFrame;
use crate::{config::*, global_cast};

use crate::{camera::CameraTrait, feature_trakcer::FeatureTracker};
use feature_manager::FeatureManager;

use opencv::core::{Mat, Point2d, Vector};

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

    /* ? 窗口 是否可以合并成一个窗口, 不合并的问题：长度管理可能出问题 */
    /// 时间戳窗口
    pub timestamps: VecDeque<f64>,
    /// 时间间隔窗口
    pub diff_times: VecDeque<f64>,
    /// ? 图像窗口, 缓冲的必要性？
    pub images: VecDeque<Mat>,
    /// 旋转矩阵窗口
    pub rot_mats: VecDeque<nalgebra::Rotation3<f64>>,
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

    /// IMU坐标系到相机坐标系的变换
    pub imu_rot_to_cam: nalgebra::Rotation3<f64>, // ric
    pub imu_trans_to_cam: nalgebra::Vector3<f64>, // tic

    /// 时间戳和图像帧的映射
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
    // pub key_poses: Vec<nalgebra::Vector3<f64>>,
}

impl<Camera> Estimator<Camera>
where
    Camera: CameraTrait,
{
    #[allow(dead_code)]
    pub fn new() -> Self {
        let mut _self = Self {
            ..Default::default()
        };
        let size = (WINDOW_SIZE + 1) as usize;
        _self.timestamps.reserve(size);
        _self.diff_times.reserve(size);
        _self.images.reserve(size);
        _self.rot_mats.reserve(size);
        _self.trans_vecs.reserve(size);
        _self.vel_vecs.reserve(size);
        _self.acce_vecs.reserve(size);
        _self.gyro_vecs.reserve(size);
        _self.bias_acces.reserve(size);
        _self.bias_gyros.reserve(size);
        _self.t_image_frame_map.reserve(size);
        _self.feature_frame_buf.reserve(size);

        // ? add default
        _self.rot_mats.push_back(Default::default());
        _self.trans_vecs.push_back(Default::default());
        _self.vel_vecs.push_back(Default::default());
        _self.bias_acces.push_back(Default::default());
        _self.bias_gyros.push_back(Default::default());

        _self
    }
    pub fn input_feature(&mut self, timestamp: f64, feature_frame: &FeatureFrame) -> Result<()> {
        let _t = timestamp;
        let _f = feature_frame;
        let _a = &feature_frame.image;
        let feature_frame = feature_frame.clone();
        //
        self.feature_frame_buf.push_back(feature_frame);
        self.process_measurements();

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
        // TODO slide_window
        match self.marginalization_flag {
            MarginalizationFlag::MarginOld => {
                // ? Headers back_R0 back_P0
                if self.frame_count == WINDOW_SIZE {
                    let t_front = *self.timestamps.front().unwrap() as i64;
                    match self.solver_flag {
                        SolverFlag::Initial => {
                            // [x] all_image_frame -> t_image_frame_map
                            self.t_image_frame_map.remove(&t_front);
                        }
                        _ => {}
                    }
                    //
                    self.timestamps.pop_front();
                    self.images.pop_front();
                    self.rot_mats.pop_front();
                    self.trans_vecs.pop_front();
                    // [ ] USE_IMU
                    // [ ] slideWindowOld --> feature_manager

                    match self.solver_flag {
                        SolverFlag::Initial => {
                            self.feature_manager.remove_old();
                        }
                        SolverFlag::NonLinear => {
                            // TODO 这里需要移除深度
                            // self.feature_manager.remove_old();
                        }
                    }
                }
            }
            MarginalizationFlag::MarginSecondNew => {
                self.timestamps.pop_front();
                self.images.pop_front();
                self.rot_mats.pop_front();
                self.trans_vecs.pop_front();

                // [ ] USE_IMU
                // [ ] slideWindowNew 边缘化
                // README https://zhuanlan.zhihu.com/p/499831202
                // README https://zhuanlan.zhihu.com/p/430996372

                self.feature_manager.remove_new(self.frame_count);
            }
        }
    }

    /// outliersRejection
    fn outliers_rejection(&mut self) -> HashSet<i32> {
        // TODO outliersRejection
        HashSet::new()
    }

    /// failureDetection
    fn failure_detection(&mut self) -> bool {
        // TODO failureDetection
        log::warn!("failure_detection");
        return false;
    }

    /// setParameters
    fn set_parameters(&mut self) {
        // TODO setParameters
        // [ ] TIC RIC
        // self.imu_rot_to_cam
        // self.imu_trans_to_cam
        // [ ] td and g

        // [ ]: MULTIPLE_THREAD
    }

    /// ? updateLatestStates
    fn update_latest_states(&mut self) {
        // TODO updateLatestStates
    }

    /// 非线性优化
    fn optimization(&mut self) {
        // TODO:optimization 非线性优化
    }
    /// relativePose 使用五点法计算相对位姿
    fn relative_pose(&self) -> Option<(i32, nalgebra::Rotation3<f64>, nalgebra::Vector3<f64>)> {
        for i in 0..WINDOW_SIZE {
            // matched points
            let (corres, average_parallax) = self
                .feature_manager
                .get_corresponding_points(i, WINDOW_SIZE);

            if corres.len() > 20 && average_parallax > 30.0 {
                if let Some((_rot, _trans)) = self.feature_manager.solve_relative_rt(&corres) {
                    return Some((i, _rot, _trans));
                }
            }
        }
        None
    }
    /// initial_structure
    fn initial_structure(&mut self) -> bool {
        // TODO initial_structure
        if let Some((i, rot, trans)) = self.relative_pose() {
            // SFM
            // log::info!("initial_structure i:{} rot:{} trans:{}", i, rot, trans);
            // frame i, rot_i: rot , trans_i : trans
            // 求解 i --> frame_count 的相对位姿
            // 条件：
            // 1. i 时刻的姿态，位移。

            // 路标i第 j 帧的特征
            let mut sfm_list = sfm::SFMFeatureList::new();
            for it in self.feature_manager.features.iter() {
                let point2s = it
                    .point_features
                    .iter()
                    .map(|it| Point2d::new(it.0.point.x, it.0.point.y))
                    .collect::<Vector<_>>();

                let item = sfm::SFMFeature {
                    state: false,
                    start_frame: it.start_frame,
                    feature_id: it.feature_id,
                    point2s,
                    ..Default::default()
                };
                sfm_list.push(item);
            }
            let i = i as usize; // 开始
            let e = self.frame_count as usize + 1;
            if let Some((c_poses, track_points)) =
                sfm::sfm_construct(i, e, rot, trans, &mut sfm_list)
            {
                // solve pnp for all frame

                for (i, it) in self.t_image_frame_map.iter_mut().enumerate() {
                    //
                    if *it.0 == self.timestamps[i] as i64 {
                        it.1.is_key_frame = true;
                        it.1.rot_matrix = c_poses[i].rotation.inverse().to_rotation_matrix()
                            * self.imu_rot_to_cam;
                        it.1.trans_vector =
                            -(c_poses[i].rotation.inverse() * c_poses[i].translation.vector);
                        continue;
                    } else if *it.0 > self.timestamps[i] as i64 {
                        // continue;
                    }

                    let mut rvec: Mat = global_cast::Quaterniond(c_poses[i].rotation).into();
                    let mut tvec: Mat = global_cast::Vector3d(c_poses[i].translation.vector).into();
                    //
                    it.1.is_key_frame = false;
                    // let pts_3_vector = Vector::new();
                    // let pts_2_vector = Vector::new();

                    for (feature_id, point_feature) in it.1.points.iter() {
                        if let Some(pt) = track_points.get(&feature_id) {
                            // ????!!!
                        }
                    }
                }
                true
            } else {
                self.marginalization_flag = MarginalizationFlag::MarginOld;
                false
            }
        } else {
            log::info!("initial_structure failed");
            false
        }
    }
    fn process_image(&mut self, frame: &FeatureFrame, timestamp: f64) {
        log::info!("process_image frame_count: {}", self.frame_count);
        self.timestamps.push_back(timestamp);

        // [x] addFeatureCheckParallax
        self.marginalization_flag = if self.feature_manager.add_feature_check_parallax(
            self.frame_count,
            &frame.point_features,
            self.td,
        ) {
            MarginalizationFlag::MarginOld
        } else {
            MarginalizationFlag::MarginSecondNew
        };
        log::info!("marginalization_flag: {:?}", self.marginalization_flag);

        let mut image_frame = image_frame::ImageFrame::new(timestamp, &frame.point_features);
        image_frame.pre_integration = image_frame::IntegrationBase::default(); // FIXME:tmp_pre_integration

        // [x] all_image_frame and tmp_pre_integration
        self.t_image_frame_map.insert(timestamp as i64, image_frame);

        // TODO ESTIMATE_EXTRINSIC == 2

        match self.solver_flag {
            // sadf
            SolverFlag::Initial => {
                // Initial
                if true {
                    let result = false;
                    if self.frame_count == WINDOW_SIZE {
                        if timestamp - self.initial_timestamp > 0.1 {
                            // 初始化
                            log::info!("initial_structure");
                            self.initial_structure();
                            self.initial_timestamp = timestamp;
                        }
                    }
                    if result {
                        // [ ] optimization
                        self.optimization();
                        // [ ] updateLatestStates
                        self.update_latest_states();
                        self.solver_flag = SolverFlag::NonLinear;
                    }
                    // [ ] slideWindow
                    self.slide_window();
                }

                // ? 填充之前的数据 add: >=1
                if self.frame_count < WINDOW_SIZE {
                    self.frame_count += 1;
                    self.rot_mats
                        .push_back(self.rot_mats.back().unwrap().clone());
                    self.trans_vecs
                        .push_back(self.trans_vecs.back().unwrap().clone());
                    self.vel_vecs
                        .push_back(self.vel_vecs.back().unwrap().clone());
                    self.bias_acces
                        .push_back(self.bias_acces.back().unwrap().clone());
                    self.bias_gyros
                        .push_back(self.bias_gyros.back().unwrap().clone());
                }
            }
            SolverFlag::NonLinear => {
                // NonLinear
                if !USE_IMU {
                    self.rot_mats.make_contiguous();
                    // [x] self.feature_manager initFramePoseByPnP
                    self.feature_manager.init_frame_pose_by_pnp(
                        self.frame_count,
                        &mut self.rot_mats,
                        &mut self.trans_vecs,
                        &self.imu_rot_to_cam,
                        &self.imu_trans_to_cam,
                    );
                }
                self.feature_manager.triangulate();
                self.optimization();

                // [x] outliersRejection 移除异常点
                let remove_ids = self.outliers_rejection();
                // [x] self.feature_manager removeOutlier
                self.feature_manager.remove_outlier(remove_ids);
                // [ ] MULTIPLE_THREAD

                // [x] failureDetection
                if self.failure_detection() {
                    // [ ] ? self.failure_occur = true;
                    // [x] clearState
                    self.clear_state();
                    // [x] setParameter
                    self.set_parameters();
                    return;
                }
                self.slide_window();

                // [x] f_manager.removeFailures();
                self.feature_manager.remove_failures();
                // [ ] key_poses;

                // [ ] last_R
                self.update_latest_states();
            }
        }
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

    /// clearState
    fn clear_state(&mut self) {
        self.acce_buf.clear();
        self.gyro_buf.clear();
        self.feature_frame_buf.clear();

        self.prev_time = -1.0;
        self.cur_time = 0.0;

        self.input_image_cnt = 0;
        self.initial_timestamp = 0.0;

        self.frame_count = 0;
        self.diff_times.clear();
        self.rot_mats.clear();
        self.trans_vecs.clear();
        self.vel_vecs.clear();
        self.acce_vecs.clear();
        self.gyro_vecs.clear();
        self.bias_acces.clear();
        self.bias_gyros.clear();

        // TODO pre_integrations

        self.imu_rot_to_cam = nalgebra::Rotation3::identity();
        self.imu_trans_to_cam = nalgebra::Vector3::zeros();

        self.solver_flag = SolverFlag::Initial;
        // 图像帧的映射
        self.t_image_frame_map.clear();

        // feature manager
        self.feature_manager.clear_state();
    }
}
