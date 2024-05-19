//! 估计器
//! 通过特征点和IMU数据估计相机位姿

mod feature_manager;
mod image_frame;
mod sfm;

use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::feature_trakcer::FeatureFrame;
use crate::{config::*, global_cast};

use feature_manager::FeatureManager;

use opencv::core::{Mat, Point2d, Point3d, Vector};

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

/// 根据帧号维护呀一个窗口大小为 [WINDOW_SIZE] 的时间戳窗口
#[derive(Debug, Default)]
struct EstimatorImageWindows {
    pub timestamps: [u64; WINDOW_SIZE + 1],
    // pub diff_times: [i64; WINDOW_SIZE + 1],
    pub images: [Mat; WINDOW_SIZE + 1],
    pub rot_mats: [nalgebra::Rotation3<f64>; WINDOW_SIZE + 1],
    pub trans_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
    // pub vel_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
}

impl EstimatorImageWindows {
    pub fn forword(&mut self) {
        for i in 0..WINDOW_SIZE {
            self.timestamps.swap(i, i + 1);
            self.images.swap(i, i + 1);
            self.rot_mats.swap(i, i + 1);
            self.trans_vecs.swap(i, i + 1);
        }
        self.timestamps[WINDOW_SIZE] = self.timestamps[WINDOW_SIZE - 1];
        self.images[WINDOW_SIZE] = self.images[WINDOW_SIZE - 1].clone();
        self.rot_mats[WINDOW_SIZE] = self.rot_mats[WINDOW_SIZE - 1];
        self.trans_vecs[WINDOW_SIZE] = self.trans_vecs[WINDOW_SIZE - 1];
    }

    fn clear(&mut self) {
        self.timestamps = [0; WINDOW_SIZE + 1];
        // ? images
        self.rot_mats = [nalgebra::Rotation3::identity(); WINDOW_SIZE + 1];
        self.trans_vecs = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
    }
}

#[derive(Debug, Default)]
pub struct Estimator {
    /// 输入图像计数
    pub input_image_cnt: i32,

    pub prev_time: u64,
    pub cur_time: u64,
    /// td 校准时间
    pub td: u64,
    /// 初始化时间
    pub initial_timestamp: u64,

    // IMU buffer
    pub acce_buf: VecDeque<(u64, [f64; 3])>,
    pub gyro_buf: VecDeque<(u64, [f64; 3])>,

    /// 帧计数
    pub frame_count: usize,

    /* ? 窗口 是否可以合并成一个窗口, 不合并的问题：长度管理可能出问题 */
    image_window: EstimatorImageWindows,
    /// 时间戳窗口
    // pub timestamps: VecDeque<u64>,
    /// 时间间隔窗口
    // pub diff_times: VecDeque<f64>,
    /// ? 图像窗口, 缓冲的必要性？
    pub images: VecDeque<Mat>,
    /// 旋转矩阵窗口
    // pub rot_mats: VecDeque<nalgebra::Rotation3<f64>>,
    /// 平移向量窗口
    // pub trans_vecs: VecDeque<nalgebra::Vector3<f64>>,
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

    /// body 到相机坐标系的变换
    pub imu_rot_to_cam: nalgebra::Rotation3<f64>, // ric
    pub imu_trans_to_cam: nalgebra::Vector3<f64>, // tic

    /// 时间戳和图像帧的映射
    pub t_image_frame_map: HashMap<u64, ImageFrame>,

    /// 每一帧的特征点缓冲：包括时间戳，图像，该图像所有特征点。
    feature_frame_buf: VecDeque<FeatureFrame>,
    /// 特征管理器，提取每一帧的特征点，按照时间顺序管理特征点。
    feature_manager: FeatureManager,

    // Flag
    marginalization_flag: MarginalizationFlag,
    solver_flag: SolverFlag,
    // pub key_poses: Vec<nalgebra::Vector3<f64>>,
}

impl Estimator {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Default::default()
    }
    pub fn input_feature(&mut self, feature_frame: &FeatureFrame) -> Result<()> {
        let feature_frame = feature_frame.clone();
        self.feature_frame_buf.push_back(feature_frame);
        self.process_measurements();
        Ok(())
    }

    pub fn slide_window(&mut self) {
        // TODO slide_window
        match self.marginalization_flag {
            MarginalizationFlag::MarginOld => {
                let t_front = self.image_window.timestamps[0];
                let first_rot = self.image_window.rot_mats[0];
                let first_trans = self.image_window.trans_vecs[0];
                if self.frame_count == WINDOW_SIZE {
                    self.image_window.forword();

                    match self.solver_flag {
                        SolverFlag::Initial => {
                            self.t_image_frame_map.remove(&t_front);
                        }
                        _ => {}
                    }

                    // TODO USE_IMU

                    match self.solver_flag {
                        SolverFlag::Initial => {
                            self.feature_manager.remove_old();
                        }
                        SolverFlag::NonLinear => {
                            // TODO 这里需要移除深度
                            let _f_rot = first_rot;
                            let _f_trans = first_trans;
                            // self.feature_manager.remove_old();
                        }
                    }
                }
            }
            MarginalizationFlag::MarginSecondNew => {
                self.image_window.forword();

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
    fn relative_pose(&self) -> Option<(usize, nalgebra::Rotation3<f64>, nalgebra::Vector3<f64>)> {
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

    /// visualInitialAlign
    fn visual_initial_align(&mut self) {
        // TODO visualInitialAlign
        // TODO VisualIMUAlignment

        // 状态同步到窗口中
        for i in 0..self.frame_count + 1 {
            let t = self.image_window.timestamps[i];
            self.image_window.rot_mats[i] = self.t_image_frame_map[&t].rot_matrix;
            self.image_window.trans_vecs[i] = self.t_image_frame_map[&t].trans_vector;
            self.t_image_frame_map.get_mut(&t).unwrap().is_key_frame = true;
        }
    }
    /// initial_structure
    fn initial_structure(&mut self) -> bool {
        // TODO initial_structure
        if let Some((i, rot, trans)) = self.relative_pose() {
            // SFM
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

            if let Some((c_poses, track_points)) =
                sfm::sfm_construct(i, self.frame_count + 1, rot, trans, &mut sfm_list)
            {
                let mut i = 0;
                // index feature_id, image_frame
                for (&timestamp, image_frame) in self.t_image_frame_map.iter_mut() {
                    if timestamp == self.image_window.timestamps[i] {
                        image_frame.is_key_frame = true;
                        image_frame.rot_matrix = c_poses[i].rotation.inverse().to_rotation_matrix()
                            * self.imu_rot_to_cam;
                        image_frame.trans_vector =
                            -(c_poses[i].rotation.inverse() * c_poses[i].translation.vector);
                        i += 1;
                        continue;
                    } else if timestamp > self.image_window.timestamps[i] {
                        i += 1;
                    }

                    image_frame.is_key_frame = false;
                    let mut rvec: Mat = global_cast::Quaterniond(c_poses[i].rotation).into();
                    let mut tvec: Mat = global_cast::Vector3d(c_poses[i].translation.vector).into();
                    //
                    let mut pts_3_vector = Vector::<Point3d>::new();
                    let mut pts_2_vector = Vector::<Point2d>::new();

                    for (feature_id, point_feature) in image_frame.points.iter() {
                        if let Some(pt) = track_points.get(&feature_id) {
                            pts_3_vector.push(*pt);
                            pts_2_vector
                                .push(Point2d::new(point_feature.point.x, point_feature.point.y));
                        }
                    }

                    if pts_3_vector.len() < 6 {
                        log::info!("pts_3_vector.len() < 6: {}", pts_3_vector.len());
                        return false;
                    }
                    let k = Mat::from_slice_2d(&crate::config::K).unwrap();
                    let d = Mat::default();

                    if !opencv::calib3d::solve_pnp(
                        &pts_3_vector,
                        &pts_2_vector,
                        &k,
                        &d,
                        &mut rvec,
                        &mut tvec,
                        true,
                        opencv::calib3d::SOLVEPNP_ITERATIVE,
                    )
                    .unwrap()
                    {
                        log::info!("solve_pnp failed");
                        return false;
                    }
                    //
                    let rvec = global_cast::Vector3d::from(rvec).0;
                    let tvec = global_cast::Vector3d::from(tvec).0;
                    let quat = nalgebra::UnitQuaternion::from_axis_angle(
                        &nalgebra::UnitVector3::new_normalize(rvec),
                        rvec.norm(),
                    )
                    .inverse();
                    image_frame.rot_matrix =
                        quat.to_rotation_matrix() * self.imu_rot_to_cam.transpose();
                    image_frame.trans_vector = quat * -tvec;
                }
                // TODO visualInitialAlign

                self.visual_initial_align();
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
    fn process_image(&mut self, frame: &FeatureFrame, timestamp: u64) {
        self.image_window.timestamps[self.frame_count] = timestamp;
        self.image_window.images[self.frame_count] = frame.image.clone();

        // 检查视差
        self.marginalization_flag = if self.feature_manager.add_feature_check_parallax(
            self.frame_count,
            &frame.point_features,
            self.td,
        ) {
            MarginalizationFlag::MarginOld
        } else {
            MarginalizationFlag::MarginSecondNew
        };

        let mut image_frame = image_frame::ImageFrame::new(timestamp, &frame.point_features);
        image_frame.pre_integration = image_frame::IntegrationBase::default(); // FIXME:tmp_pre_integration
        self.t_image_frame_map.insert(timestamp, image_frame);

        // TODO ESTIMATE_EXTRINSIC == 2

        match self.solver_flag {
            SolverFlag::Initial => {
                if true {
                    let mut result = false;
                    if self.frame_count == WINDOW_SIZE {
                        if timestamp - self.initial_timestamp > 200_000_000 {
                            // 初始化
                            result = self.initial_structure();
                            self.initial_timestamp = timestamp;
                        }
                    }
                    if result {
                        self.optimization();
                        self.update_latest_states();
                        self.solver_flag = SolverFlag::NonLinear;
                        log::info!("initial success");
                    }
                    self.slide_window();
                }

                // ? 填充之前的数据 add: >=1
                if self.frame_count < WINDOW_SIZE {
                    self.frame_count += 1;
                    self.image_window.rot_mats[self.frame_count] =
                        self.image_window.rot_mats[self.frame_count - 1];
                    self.image_window.trans_vecs[self.frame_count] =
                        self.image_window.trans_vecs[self.frame_count - 1];
                    // TODO IMU
                }
            }
            SolverFlag::NonLinear => {
                // NonLinear
                if !USE_IMU {
                    self.feature_manager.init_frame_pose_by_pnp(
                        self.frame_count,
                        &mut self.image_window.rot_mats,
                        &mut self.image_window.trans_vecs,
                        &self.imu_rot_to_cam,
                        &self.imu_trans_to_cam,
                    );
                }
                self.feature_manager.triangulate(
                    &self.image_window.rot_mats,
                    &self.image_window.trans_vecs,
                    &self.imu_rot_to_cam,
                    &self.imu_trans_to_cam,
                );
                self.optimization();

                let remove_ids = self.outliers_rejection(); // [ ] outliersRejection 移除异常点
                self.feature_manager.remove_outlier(remove_ids);

                //  MULTIPLE_THREAD

                // [x] failureDetection
                // if self.failure_detection() {
                //     self.clear_state();
                //     self.set_parameters();
                //     return;
                // }

                self.slide_window();
                self.feature_manager.remove_failures();
                // [ ] key_poses;
                self.update_latest_states();
            }
        }
    }

    #[inline]
    fn imu_available(&self, t: u64) -> bool {
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
                    // 使用IMU等待IMU加载
                    if !USE_IMU || self.imu_available(cur_time) {
                        break;
                    } else {
                        log::info!("wait for imu data");

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

        self.prev_time = 0;
        self.cur_time = 0;

        self.input_image_cnt = 0;
        self.initial_timestamp = 0;

        self.frame_count = 0;

        self.image_window.clear();

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
