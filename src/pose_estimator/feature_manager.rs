use std::collections::{HashSet, VecDeque};

use nalgebra::{Rotation3, Vector3};
use opencv::{calib3d::FM_RANSAC, core::*};

use crate::{
    config::{FOCAL_LENGTH, MIN_PARALLAX},
    feature_trakcer::{PointFeature, PointFeatureMap},
};

use super::WINDOW_SIZE;

#[derive(Debug, Default)]
pub struct FeaturePerFrame(pub PointFeature, pub f64);

#[allow(dead_code)]
#[derive(Debug, Default)]
pub enum FeatureSolveFlag {
    /// 待求解
    #[default]
    WaitSolve,
    /// 求解成功
    SolveSuccess,
    /// 求解失败
    SolveFail,
}

#[derive(Debug, Default)]
pub struct FeaturePerId {
    /// 特征点的id
    pub feature_id: i32,
    /// 特征点的起始帧
    pub start_frame: i32,
    /// 该特征点的所有特征数据
    pub point_features: Vec<FeaturePerFrame>,
    /// 估计点的深度
    pub estimated_depth: f64,
    ///  0 haven't solve yet; 1 solve succ; 2 solve fail;
    pub solve_flag: FeatureSolveFlag,
}
impl FeaturePerId {
    #[allow(dead_code)]
    pub fn new(feature_id: i32, start_frame: i32) -> Self {
        Self {
            feature_id,
            start_frame,
            point_features: Vec::new(), // TODO point_features
            estimated_depth: -1.0,
            ..Default::default()
        }
    }

    pub fn end_frame_id(&self) -> i32 {
        self.start_frame + self.point_features.len() as i32 - 1
    }
}

#[derive(Debug, Default)]
pub struct FeatureManager {
    /// 所有特征点
    pub features: Vec<FeaturePerId>,

    /* 特征点 是否可使用局部变量？ */
    /// 新增的特征点数量
    pub new_feature_num: i32,
    /// 上一帧的特征点数量
    pub last_track_num: i32,
    /// 长期追踪特征点数量
    pub long_track_num: i32,
    /// 最新帧的平均视差
    pub last_average_parallax: f64,
}

impl FeatureManager {
    /// solvePoseByPnP
    fn solve_pose_by_pnp(
        &self,
        cam_rot_mat: &mut nalgebra::Rotation3<f64>,
        cam_trans_vec: &mut nalgebra::Vector3<f64>,
        pts_2d: &Vector<Point2f>,
        // pts_3d: &Vector<Point3f>,
    ) -> bool {
        let rot_initial = cam_rot_mat.inverse();
        let trans_initial = rot_initial * *cam_trans_vec;
        if pts_2d.len() < 4 {
            log::warn!("PnP needs at least 4 points");
            return false;
        }
        let _ = trans_initial;
        // TODO EPnP Algorithm https://github.com/cvlab-epfl/EPnPƒ
        // https://github.com/rust-cv/pnp
        // https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
        false
    }
    /// initFramePoseByPnP
    pub fn init_frame_pose_by_pnp(
        &self,
        frame_count: i32,
        rot_mats: &mut VecDeque<Rotation3<f64>>,
        trans_vec: &mut VecDeque<Vector3<f64>>,
        imu_rot_to_cam: &nalgebra::Rotation3<f64>,
        imu_trans_to_cam: &nalgebra::Vector3<f64>,
    ) {
        if frame_count > 0 {
            let mut pts_2d = Vector::<Point2f>::default();
            let mut pts_3d = Vector::<Vec3f>::default(); // !!  Vector::<Point3f> error

            for it_per_id in self.features.iter() {
                if it_per_id.estimated_depth > 0.0 {
                    // 该特征点追踪的次数 - 1
                    let index = (frame_count - it_per_id.start_frame) as usize;
                    // 说明该点一直追踪到了当前帧
                    if it_per_id.point_features.len() >= index + 1 {
                        // [x] let pts_in_cam
                        // ? 使用 imu_rot_to_cam 和 imu_trans_to_cam 估计出 pts_in_cam
                        let pts_in_cam = imu_rot_to_cam
                            * (it_per_id.point_features[0].0.point * it_per_id.estimated_depth)
                            + imu_trans_to_cam;

                        // [x] let pts_in_world
                        // 使用估计的 rot_mat 和 trans_vec 估计出 pts_in_world
                        let pts_in_world = rot_mats[it_per_id.start_frame as usize] * pts_in_cam
                            + trans_vec[it_per_id.start_frame as usize];

                        pts_2d.push(Point2f::new(
                            it_per_id.point_features[index].0.point.x as f32,
                            it_per_id.point_features[index].0.point.y as f32,
                        ));

                        pts_3d.push(Vec3f::from_array([
                            pts_in_world.x as f32,
                            pts_in_world.y as f32,
                            pts_in_world.z as f32,
                        ]));
                    }
                }
            }
            // camera R and t
            let mut cam_rot_mat = rot_mats[frame_count as usize - 1] * imu_rot_to_cam;
            let mut cam_trans_vec = rot_mats[frame_count as usize - 1] * imu_trans_to_cam
                + trans_vec[frame_count as usize - 1];
            // trans to w_T_cam: world to camera

            // [ ] solvePoseByPnP 根据PnP算法计算相机姿态。
            if self.solve_pose_by_pnp(&mut cam_rot_mat, &mut cam_trans_vec, &pts_2d) {
                // 这里计算出 cam_rot_mat 和 cam_trans_vec
                let rot_tmp = cam_rot_mat * imu_rot_to_cam.transpose();
                rot_mats.push_back(rot_tmp);
                trans_vec.push_back(-(rot_tmp * imu_trans_to_cam) + cam_trans_vec);
                log::debug!("Rot Mat: {:?}", rot_tmp);
            }
        }
    }

    /// ? 三角测量
    pub fn triangulate(&mut self) {
        // TODO triangulate
        self.features.iter_mut().for_each(|it_per_id| {
            //
            if it_per_id.estimated_depth > 0.0 {
                return;
            }
            if it_per_id.point_features.len() > 1 {
                // [ ] triangulate 计算
                return;
            }
            // ? 判断有问题
            let used_num = it_per_id.point_features.len();
            if used_num < 4 {
                return;
            }

            let imu_i = it_per_id.start_frame;
            let mut imu_j = imu_i - 1;

            for _it_pt_feat in it_per_id.point_features.iter() {
                imu_j += 1;
                let _ = imu_j;
                // ? 毫无意义的判断
            }

            // [ ] estimated_depth
            if it_per_id.estimated_depth < 0.1 {
                it_per_id.estimated_depth = 1.0;
            }
        });
    }

    /// 移除异常点
    /// removeOutlier
    pub fn remove_outlier(&mut self, frame_counts: HashSet<i32>) {
        self.features
            .retain(|it_per_id| !frame_counts.contains(&it_per_id.feature_id));
    }
    /// remove_failures
    pub fn remove_failures(&mut self) {
        self.features.retain(|it| match it.solve_flag {
            FeatureSolveFlag::SolveFail => false,
            _ => true,
        });
    }

    /// 获取两帧之间的共识帧
    pub fn get_corresponding_points(
        &self,
        frame_left: i32,
        frame_right: i32,
    ) -> (Vec<(Point2f, Point2f)>, f64) {
        let mut sum_parallax = 0f64;
        let corres = if frame_left > frame_right {
            vec![]
        } else {
            self.features
                .iter()
                .filter_map(|it| {
                    if it.start_frame <= frame_left && frame_right <= it.end_frame_id() {
                        let index_left = (frame_left - it.start_frame) as usize;
                        let index_right = (frame_right - it.start_frame) as usize;
                        // point match
                        let left_point = &it.point_features[index_left].0.point;
                        let right_point = &it.point_features[index_right].0.point;

                        let left = Point2f::new(left_point.x as f32, left_point.y as f32);
                        let right = Point2f::new(right_point.x as f32, right_point.y as f32);
                        let parallax = (left_point - right_point).norm();
                        sum_parallax += parallax;
                        // 计算视差
                        Some((left, right))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let average_parallax = if corres.len() > 0 {
            sum_parallax / corres.len() as f64
        } else {
            0.0
        };
        (corres, average_parallax * FOCAL_LENGTH)
    }

    // solveRelativeRT 根据共视点求解相对位姿
    pub fn solve_relative_rt(
        &self,
        corres: &Vec<(Point2f, Point2f)>,
    ) -> Option<(nalgebra::Rotation3<f64>, nalgebra::Vector3<f64>)> {
        // corres size > 20
        // let
        let ll = corres
            .iter()
            .map(|(l, _)| l.clone())
            .collect::<Vector<Point2f>>();
        let rr = corres
            .iter()
            .map(|(_, r)| r.clone())
            .collect::<Vector<Point2f>>();

        let mut mask = Vector::<u8>::default();
        let fund_mat = opencv::calib3d::find_fundamental_mat_1(
            &ll,
            &rr,
            FM_RANSAC,
            0.3 / FOCAL_LENGTH,
            0.99,
            &mut mask,
        )
        .unwrap();

        // 无需校准相机内参，已经校验过
        let cam_mat = Mat::eye(3, 3, CV_32F).unwrap().to_mat().unwrap();

        let mut rot = Mat::default();
        let mut trans = Mat::default();
        let inlier_cnt = opencv::calib3d::recover_pose_estimated(
            &fund_mat, &ll, &rr, &cam_mat, &mut rot, &mut trans, &mut mask,
        )
        .unwrap();

        let mut rot_mat = nalgebra::Matrix3::default();
        let mut trans_vec = nalgebra::Vector3::default();
        for i in 0..3 {
            for j in 0..3 {
                rot_mat[(i, j)] = *rot.at_2d::<f64>(i as i32, j as i32).unwrap();
            }
            trans_vec[i] = *trans.at::<f64>(i as i32).unwrap();
        }
        let rot = Rotation3::from_matrix(&rot_mat);
        let trans = -(rot * trans_vec);
        if inlier_cnt > 12 {
            Some((rot, trans))
        } else {
            None
        }
    }

    /// removeFront
    pub fn remove_new(&mut self, frame_count: i32) {
        self.features.iter_mut().for_each(|it| {
            if it.start_frame == frame_count {
                // 最新帧滑动成次新帧
                it.start_frame -= 1;
            } else {
                // 次新帧该点仍然被追踪
                if it.end_frame_id() >= frame_count - 1 {
                    let second_new = (frame_count - 1 - it.start_frame) as usize; // second new frame
                    it.point_features.remove(second_new as usize);
                }
            }
        });
        // remove empty features
        self.features.retain(|it| it.point_features.len() > 0);
    }
    /// remove first frame
    pub fn remove_old(&mut self) {
        self.features.iter_mut().for_each(|it| {
            if it.start_frame != 0 {
                it.start_frame -= 1;
            } else {
                it.point_features.remove(0);
            }
        });
        // remove empty features
        self.features.retain(|it| it.point_features.len() > 0);
    }

    /// Finish: compensatedParallax2 视差补偿
    fn compensated_parallax2(&self, it_per_id: &FeaturePerId, frame_count: i32) -> f64 {
        // i: 倒数第二帧 j：倒数第一帧
        let i = (frame_count - 2 - it_per_id.start_frame) as usize;
        let frame_i = &it_per_id.point_features[i];
        let frame_j = &it_per_id.point_features[i + 1];

        // p_j
        let p_j = frame_j.0.point;
        let (u_j, v_j) = (p_j[0] / p_j[2], p_j[1] / p_j[2]);

        // p_i
        let p_i = frame_i.0.point;
        let (u_i, v_i) = (p_i[0] / p_i[2], p_i[1] / p_i[2]);
        let (du, dv) = (u_i - u_j, v_i - v_j);

        // p_i_comp
        let p_i_comp = p_i.clone();
        let (u_i_comp, v_i_comp) = (p_i_comp[0] / p_i_comp[2], p_i_comp[1] / p_i_comp[2]);
        let (du_comp, dv_comp) = (u_i_comp - u_j, v_i_comp - v_j);

        let a = (du * du + dv * dv) as f64;
        let b = (du_comp * du_comp + dv_comp * dv_comp) as f64;
        a.min(b).sqrt().max(0.0f64)
    }
    /// Finish: 添加特征点检查视差
    pub fn add_feature_check_parallax(
        &mut self,
        frame_count: i32,                  // 当前帧数
        points_features: &PointFeatureMap, // 最新帧的所有特征点
        td: f64,
    ) -> bool {
        let mut parallax_sum = 0.0;
        let mut parallax_num = 0;

        self.new_feature_num = 0;
        self.last_track_num = 0;
        self.long_track_num = 0;

        points_features.iter().for_each(|(k, v)| {
            // [x] FeaturePerFrame

            // 特征点 id
            let feature_id = *k;
            // 创建一个带 td 的特征点
            let f_per_frame = FeaturePerFrame(v.clone(), td);
            // 从所有 features 中查找 feature_id
            let find_it = self
                .features
                .iter_mut()
                .find(|feature| feature.feature_id == feature_id);
            // 如何可以找到 feature_id
            if let Some(f_per_id) = find_it {
                // 最新帧的已存在追踪特征点数量加1，已经存在的特征点仍然存在的数量。
                self.last_track_num += 1;
                // 在已有的特征点追踪，添加一个新的 PointFeature
                f_per_id.point_features.push(f_per_frame);
                // 某个 id 对应的特征点的数量大于4，认为是长追踪特征点，追踪了5帧。
                if f_per_id.point_features.len() > 4 {
                    self.long_track_num += 1;
                }
            } else {
                // 创建一个新的特征点进行追踪
                let mut f_per_id = FeaturePerId::new(feature_id, frame_count);
                // 对该特征点添加特征数据
                f_per_id.point_features.push(f_per_frame);
                self.features.push(f_per_id);
                // 新增特征点数量加1
                self.new_feature_num += 1;
            }
        });

        // 条件：
        // 1. 帧数小于2
        // 2. 已经存在的特征点仍然存在的数量 < 20
        // 3. 长期追踪特征点数量（追踪次数大于等于5帧） < 40
        // 4. 新增特征点数量大于 已有的特征点数量一半
        if frame_count < 2
            || self.last_track_num < 20
            || self.long_track_num < 40
            || self.new_feature_num > self.last_track_num / 2
        {
            return true;
        };

        // 遍历所有特征点: 遍历所有特征点，计算平均视差
        self.features.iter().for_each(|it_per_id| {
            // 1. 某一个特征点追踪开始的帧号 < 当前帧号 - 2 ==> 说明该特征点已经被追踪了超过2帧
            // 2. 某一个特征点追踪开始的帧号 + 特征点追踪的帧数 - 1 >= 当前帧号 - 1 ==> 说明该特征点已经追踪到当前帧号的前一帧
            if it_per_id.start_frame <= frame_count - 2
                && it_per_id.end_frame_id() >= frame_count - 1
            {
                // [x] compensatedParallax2
                parallax_sum += self.compensated_parallax2(it_per_id, frame_count);
                parallax_num += 1;
            }
        });

        // 视差的个数
        if parallax_num == 0 {
            true
        } else {
            //  平均视差
            let average_parallax = parallax_sum / parallax_num as f64;
            self.last_average_parallax = average_parallax * FOCAL_LENGTH;
            average_parallax >= MIN_PARALLAX
        }
    }

    pub fn clear_state(&mut self) {
        self.features.clear();
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Rotation3, Vector3};

    #[test]
    fn test_rot2rvec() {
        let axis = Vector3::x_axis();
        let angle = 1.57;
        let b = Rotation3::from_axis_angle(&axis, angle);

        println!("{:?}", b);
    }
}
