//! 估计器
//! 通过特征点和IMU数据估计相机位姿

mod feature_manager;
mod image_frame;
mod integration_base;
mod sfm;
mod windows;

use anyhow::Result;
use integration_base::IntegrationBase;
use std::collections::{HashMap, HashSet, VecDeque};
use windows::{EstimatorIMUWindows, EstimatorImageWindows};

use crate::feature_trakcer::FeatureFrame;
use crate::global_types::IMUData;
use crate::utility::Utility;
use crate::{config::*, global_cast};
use image_frame::ImageFrame;

use feature_manager::FeatureManager;

use opencv::core::{Mat, Point2d, Point3d, Vector};

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

/// EstimatorWindowsTrait
pub(crate) trait EstimatorWindowsTrait {
    fn forword(&mut self);
    fn clear(&mut self);
}

#[derive(Debug, Default)]
pub struct Estimator {
    /// 输入图像计数
    pub input_image_cnt: i32,

    pub prev_time: i64,
    pub cur_time: i64,
    /// td 校准时间
    pub td: i64,
    /// 初始化时间
    pub initial_timestamp: i64,

    // IMU buffer
    pub imu_buf: VecDeque<IMUData>,
    first_imu: bool,
    prev_imu: IMUData,

    /// 帧计数
    pub frame_count: usize,

    /* ? 窗口 是否可以合并成一个窗口, 不合并的问题：长度管理可能出问题 */
    image_window: EstimatorImageWindows,
    imu_window: EstimatorIMUWindows,
    /// 时间戳窗口
    // pub timestamps: VecDeque<i64>,
    /// 时间间隔窗口
    // pub diff_times: VecDeque<f64>,
    /// ? 图像窗口, 缓冲的必要性？
    pub images: VecDeque<Mat>,
    /// 旋转矩阵窗口
    // pub rot_mats: VecDeque<nalgebra::Rotation3<f64>>,
    /// 平移向量窗口
    // pub trans_vecs: VecDeque<nalgebra::Vector3<f64>>,

    /// dt buffer
    // pub dt_buf: VecDeque<f64>,
    // /// 速度向量窗口
    // pub vel_vecs: VecDeque<nalgebra::Vector3<f64>>,
    // /// 加速度窗口
    // pub acce_vecs: VecDeque<nalgebra::Vector3<f64>>,
    // /// 角速度窗口
    // pub gyro_vecs: VecDeque<nalgebra::Vector3<f64>>,
    // /// 加速度偏置窗口
    // pub bias_acces: VecDeque<nalgebra::Vector3<f64>>,
    // /// 角速度偏置窗口
    // pub bias_gyros: VecDeque<nalgebra::Vector3<f64>>,
    // /// pre_integrations
    // pub pre_integrations: VecDeque<PreIntegration>,

    /// body 到相机坐标系的变换
    pub imu_rot_to_cam: nalgebra::Rotation3<f64>, // ric
    pub imu_trans_to_cam: nalgebra::Vector3<f64>, // tic

    /// 时间戳和图像帧的映射
    pub t_image_frame_map: HashMap<i64, ImageFrame>,

    /// 每一帧的特征点缓冲：包括时间戳，图像，该图像所有特征点。
    feature_frame_buf: VecDeque<FeatureFrame>,
    /// 特征管理器，提取每一帧的特征点，按照时间顺序管理特征点。
    feature_manager: FeatureManager,

    // Flag
    marginalization_flag: MarginalizationFlag,
    solver_flag: SolverFlag,
    // pub key_poses: Vec<nalgebra::Vector3<f64>>,

    // time
    latest_time: i64,
    latest_vel: nalgebra::Vector3<f64>,
    latest_pos: nalgebra::Vector3<f64>,
    latest_quat: nalgebra::UnitQuaternion<f64>,

    latest_acce: nalgebra::Vector3<f64>,
    latest_gyro: nalgebra::Vector3<f64>,

    latest_bias_acce: nalgebra::Vector3<f64>,
    latest_bias_gyro: nalgebra::Vector3<f64>,
    latest_gravity: nalgebra::Vector3<f64>,
}

impl Estimator {
    #[allow(dead_code)]
    pub fn new() -> Self {
        let mut my = Estimator::default();
        my.clear_state();
        my
    }

    pub fn input_imu(
        &mut self,
        timestamp: i64,
        acce: &nalgebra::Vector3<f64>,
        gyro: &nalgebra::Vector3<f64>,
    ) -> Result<()> {
        self.imu_buf.push_back(IMUData {
            timestamp,
            acc: acce.clone(),
            gyro: gyro.clone(),
        });

        match self.solver_flag {
            SolverFlag::NonLinear => {
                // TODO processIMU
                // fastPredictIMU
                let acce = nalgebra::Vector3::new(acce[0], acce[1], acce[2]);
                let gyro = nalgebra::Vector3::new(gyro[0], gyro[1], gyro[2]);
                self.fast_predict_imu(timestamp, acce, gyro);
            }
            _ => {}
        }
        Ok(())
    }

    fn fast_predict_imu(
        &mut self,
        timestamp: i64,
        acce: nalgebra::Vector3<f64>,
        gyro: nalgebra::Vector3<f64>,
    ) {
        // TODO fast_predict_imu
        // ! 以下 dt 单位为秒
        let dt = (timestamp - self.latest_time) as f64 / 1e9;

        // 计算四元数变化量
        let un_gyro = (self.latest_gyro + gyro) * 0.5 - self.latest_bias_gyro;
        let delta_quat = Utility::delta_quat(un_gyro * dt);
        self.latest_quat = self.latest_quat * delta_quat;

        // 计算加速度
        let un_acc_0 =
            self.latest_quat * (self.latest_acce - self.latest_bias_acce) - self.latest_gravity;
        let un_arr_1 = self.latest_quat * (acce - self.latest_bias_acce) - self.latest_gravity;
        let un_acc = 0.5 * (un_acc_0 + un_arr_1);

        // 更新位置和速度
        self.latest_pos = self.latest_pos + self.latest_vel * dt + 0.5 * un_acc * dt * dt; // S = S0 + V0 * t + 0.5 * a * t^2
        self.latest_vel = self.latest_vel + un_acc * dt; // V = V0 + a * t

        // 记录上一次的加速度和角速度
        self.latest_acce = acce;
        self.latest_gyro = gyro;
        self.latest_time = timestamp;
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

        for i in 0..WINDOW_SIZE + 1 {
            if let Some(pre_integration) = self.imu_window.pre_integrations[i].as_mut() {
                pre_integration.repropagate(
                    nalgebra::Vector3::<f64>::zeros(),
                    self.imu_window.bias_gyros[i],
                );
            }
        }
        // TODO
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
    fn process_image(&mut self, frame: &FeatureFrame, timestamp: i64) {
        self.image_window.timestamps[self.frame_count] = timestamp;
        self.image_window.images[self.frame_count] = frame.image.clone();

        // 检查视差，决定窗口的边缘化策略
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
        image_frame.pre_integration = IntegrationBase::default(); // FIXME:tmp_pre_integration
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
                    // image_window
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
    fn imu_available(&self, t: i64) -> bool {
        return true;
    }

    fn wait_imu_available(&mut self, t: i64) {
        loop {
            if self.imu_available(t) {
                //
                break;
            } else {
                log::info!("wait for imu data");
                // FIXME sleep()
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    }

    fn get_imu_interval(&mut self, prev_time: i64, cur_time: i64) -> Option<Vec<IMUData>> {
        let mut imu_vecs = vec![];
        if !USE_IMU || self.imu_buf.is_empty() {
            return None;
        }
        if cur_time <= self.imu_buf.back().unwrap().timestamp {
            // 0 timestamp 1 acce
            while self.imu_buf.back().unwrap().timestamp < prev_time {
                self.imu_buf.pop_front();
            }

            while self.imu_buf.back().unwrap().timestamp < cur_time {
                imu_vecs.push(self.imu_buf.pop_front().unwrap());
            }
            imu_vecs.push(self.imu_buf.front().unwrap().clone());
            Some(imu_vecs)
        } else {
            log::info!("wait for imu data");
            None
        }
    }

    fn init_first_imu_pose(&mut self, imu_vecs: &Vec<IMUData>) {
        //
        log::info!("init_first_imu_pose");
        // init_first_pose_flag = true;
        let acce_average = imu_vecs
            .iter()
            .map(|it| it.acc)
            .sum::<nalgebra::Vector3<f64>>()
            / imu_vecs.len() as f64;

        log::info!("acce_average: {}", acce_average);

        // 计算重力向量夹角
        let g0 = acce_average.normalize();
        let g1 = nalgebra::Vector3::new(0.0, 0.0, 1.0);
        let rot0 = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(g0.cross(&g1)),
            g0.angle(&g1),
        );
        let yaw = rot0.euler_angles().2;
        let rot0 = nalgebra::Rotation3::from_euler_angles(0.0, 0.0, -yaw) * rot0;

        self.image_window.rot_mats[0] = rot0;
        log::info!("rot0: {}", rot0);
    }

    fn process_imu(&mut self, dt: i64, imu_unit: &IMUData) {
        // TODO process_imu
        if !self.first_imu {
            self.first_imu = true;
            self.prev_imu = imu_unit.clone();
        }

        // pre_integrations
        if self.imu_window.pre_integrations[self.frame_count].is_none() {
            self.imu_window.pre_integrations[self.frame_count].replace(IntegrationBase::new(
                &self.prev_imu,
                &self.imu_window.bias_acces[self.frame_count],
                &self.imu_window.bias_gyros[self.frame_count],
            ));
        }

        if self.frame_count != 0 {
            let pre_integrations = self.imu_window.pre_integrations[self.frame_count]
                .as_mut()
                .unwrap();
            pre_integrations.push_back(dt, imu_unit);
            // tmp_pre_integration
            self.imu_window.dt_buf[self.frame_count] = dt;
            self.imu_window.acce_vecs[self.frame_count] = imu_unit.acc;
            self.imu_window.gyro_vecs[self.frame_count] = imu_unit.gyro;

            // ! 以下 dt 单位为秒
            let dt = dt as f64 / 1e9;
            let j = self.frame_count;
            // 计算旋转矩阵
            let un_gyro =
                0.5 * (self.prev_imu.gyro + imu_unit.gyro) - self.imu_window.bias_gyros[j];
            self.image_window.rot_mats[j] = self.image_window.rot_mats[j]
                * Utility::delta_quat(un_gyro * dt).to_rotation_matrix();
            // 计算加速度
            let un_acce_0 = self.image_window.rot_mats[j]
                * (self.prev_imu.acc - self.imu_window.bias_acces[j])
                - self.latest_gravity;
            let un_acce_1 = self.image_window.rot_mats[j]
                * (imu_unit.acc - self.imu_window.bias_acces[j])
                - self.latest_gravity;
            let un_acce = 0.5 * (un_acce_0 + un_acce_1);
            // 更新位置
            self.image_window.trans_vecs[j] =
                self.image_window.vel_vecs[j] * dt + 0.5 * un_acce * dt * dt;
            // 更新速度
            self.image_window.vel_vecs[j] = self.image_window.vel_vecs[j] + un_acce * dt;
        }
        self.prev_imu = imu_unit.to_owned();
    }

    fn process_measurements(&mut self) {
        // TODO: process measurements
        loop {
            if !self.feature_frame_buf.is_empty() {
                let frame = self.feature_frame_buf.pop_front().unwrap();
                let cur_time = frame.timestamp + self.td; // 校准时间
                self.wait_imu_available(cur_time);

                let imu_vecs =
                    if let Some(imu_vecs) = self.get_imu_interval(self.prev_time, cur_time) {
                        imu_vecs
                    } else {
                        vec![]
                    };

                if USE_IMU {
                    // TODO: USE_IMU initFirstIMUPose and processIMU
                    let init_first_pose_flag = true;
                    if init_first_pose_flag {
                        //
                        self.init_first_imu_pose(&imu_vecs);
                    }

                    for (i, imu) in imu_vecs.iter().enumerate() {
                        // 计算dt
                        let dt = match i {
                            0 => imu.timestamp - self.prev_time,
                            _ => imu.timestamp - imu_vecs[i - 1].timestamp,
                        };
                        self.process_imu(dt, imu);
                    }
                }
                // TODO: process_image()

                self.process_image(&frame, frame.timestamp);
                self.prev_time = self.cur_time;

                log::info!(
                    "{} {}",
                    *self.image_window.rot_mats.last().to_owned().unwrap(),
                    self.image_window.trans_vecs.last().unwrap()
                );

                break;
            }
        }
    }

    /// clearState
    fn clear_state(&mut self) {
        self.imu_buf.clear();
        self.feature_frame_buf.clear();

        self.prev_time = 0;
        self.cur_time = 0;

        self.input_image_cnt = 0;
        self.initial_timestamp = 0;

        self.frame_count = 0;
        // 图像
        self.image_window.clear();
        self.imu_window.clear();

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

#[test]
fn test_norm() {
    let v = nalgebra::Vector3::new(1.0, 2.0, 3.0);

    let norm = v.norm();
    println!("norm: {}", norm);
    let v_norm = v.normalize();
    println!("v_norm: {}", v_norm);
}
