use std::collections::HashSet;

use super::sfm;
use crate::{
    config::{FOCAL_LENGTH, INIT_DEPTH, MIN_PARALLAX},
    feature_trakcer::{PointFeature, PointFeatureMap},
};
use nalgebra::{Rotation3, Vector3};
use opencv::{calib3d::FM_RANSAC, core::*};

/// 特征点信息 和 时间偏置
#[derive(Debug, Default)]
pub struct FeaturePerFrame(pub PointFeature, pub u64);

/// 特征点求解状态
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

/// 特征点信息
/// feature_id: 特征点的id
/// point_features: 追踪的特征点在不同帧的信息。
/// estimated_depth: 估计的深度
/// solve_flag: 求解状态
#[derive(Debug, Default)]
pub struct FeaturePerId {
    /// 特征点的id
    pub feature_id: i32,
    /// 特征点的起始帧
    pub start_frame: usize,
    /// 该特征点的所有特征数据
    pub point_features: Vec<FeaturePerFrame>,
    /// 估计点的深度
    pub estimated_depth: f64,
    ///  0 haven't solve yet; 1 solve succ; 2 solve fail;
    pub solve_flag: FeatureSolveFlag,
}
impl FeaturePerId {
    #[allow(dead_code)]
    pub fn new(feature_id: i32, start_frame: usize) -> Self {
        Self {
            feature_id,
            start_frame,
            point_features: Vec::new(), // TODO point_features
            estimated_depth: -1.0,
            ..Default::default()
        }
    }

    pub fn end_frame_id(&self) -> usize {
        self.start_frame + self.point_features.len() - 1
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
        pts_2ds: &Vector<Point2d>,
        pts_3ds: &Vector<Point3d>,
    ) -> bool {
        let quat = nalgebra::UnitQuaternion::from_rotation_matrix(cam_rot_mat);
        let tvec = cam_trans_vec.clone();

        if let Some((quat, tvec)) = sfm::solve_pnp(pts_3ds, pts_2ds, &quat, &tvec) {
            *cam_rot_mat = quat.to_rotation_matrix();
            *cam_trans_vec = tvec;

            true
        } else {
            false
        }
    }

    /// initFramePoseByPnP
    pub fn init_frame_pose_by_pnp(
        &self,
        frame_count: usize,
        rot_mats: &mut [Rotation3<f64>],
        trans_vecs: &mut [Vector3<f64>],
        imu_rot_to_cam: &nalgebra::Rotation3<f64>,
        imu_trans_to_cam: &nalgebra::Vector3<f64>,
    ) {
        if frame_count > 0 {
            let mut pts_2d = Vector::<Point2d>::default();
            let mut pts_3d = Vector::<Point3d>::default();

            for it_per_id in self.features.iter() {
                if it_per_id.estimated_depth > 0.0 {
                    // 追踪到 frame_count 帧
                    if it_per_id.end_frame_id() >= frame_count {
                        let index = frame_count - it_per_id.start_frame;
                        // ? 使用 imu_rot_to_cam 和 imu_trans_to_cam 估计出 pts_in_cam
                        let pts_in_cam = imu_rot_to_cam
                            * (it_per_id.point_features[index].0.point * it_per_id.estimated_depth)
                            + imu_trans_to_cam;
                        let pts_in_world = rot_mats[it_per_id.start_frame] * pts_in_cam
                            + trans_vecs[it_per_id.start_frame];

                        let pt = it_per_id.point_features[index].0.point;
                        pts_2d.push(Point2d::new(pt.x, pt.y));
                        pts_3d.push(Point3d::new(pts_in_world.x, pts_in_world.y, pts_in_world.z));
                    }
                }
            }

            if pts_2d.len() < 4 {
                log::warn!("PnP needs at least 4 points: {} ", pts_2d.len());
                return;
            }

            let mut cam_rot = rot_mats[frame_count - 1] * imu_rot_to_cam;
            let mut cam_trans =
                rot_mats[frame_count - 1] * imu_trans_to_cam + trans_vecs[frame_count - 1];
            // FIXME solvePoseByPnP 根据PnP算法计算相机姿态。
            if self.solve_pose_by_pnp(&mut cam_rot, &mut cam_trans, &pts_2d, &pts_3d) {
                // 这里计算出 cam_rot_mat 和 cam_trans_vec
                let rot_tmp = cam_rot * imu_rot_to_cam.transpose();
                rot_mats[frame_count] = rot_tmp;
                trans_vecs[frame_count] = -(rot_tmp * imu_trans_to_cam) + cam_trans;

                // log::info!("Init frame pose by PnP: {}", frame_count);
                // log::info!("Rotation: {}", rot_mats[frame_count]);
                // log::info!("Translation: {}", trans_vecs[frame_count]);
            }
        }
    }

    /// 三角测量
    pub fn triangulate(
        &mut self,
        rot_mats: &[Rotation3<f64>],
        trans_vecs: &[Vector3<f64>],
        imu_rot_to_cam: &nalgebra::Rotation3<f64>,
        imu_trans_to_cam: &nalgebra::Vector3<f64>,
    ) {
        for it in self.features.iter_mut() {
            if it.estimated_depth > 0.0 {
                continue;
            }
            if it.point_features.len() > 1 {
                // [ ] 可以进行triangulate 计算
                // p1
                let i = it.start_frame;
                let rot_i = rot_mats[i] * imu_rot_to_cam;
                let tvec_i = rot_mats[i] * imu_trans_to_cam + trans_vecs[i];

                let rot_i = rot_i.transpose();
                let tvec_i = rot_i * -tvec_i;
                let pose_i = nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::from(tvec_i),
                    nalgebra::UnitQuaternion::from_rotation_matrix(&rot_i),
                );

                // p2
                let j = i + 1;
                let rot_j = rot_mats[j] * imu_rot_to_cam;
                let tvec_j = rot_mats[j] * imu_trans_to_cam + trans_vecs[j];

                let rot_j = rot_j.transpose();
                let tvec_j = rot_j * -tvec_j;
                let pose_j = nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::from(tvec_j),
                    nalgebra::UnitQuaternion::from_rotation_matrix(&rot_j),
                );

                let point0 = it.point_features[0].0.point;
                let point1 = it.point_features[1].0.point;
                let point_i = Point2d::new(point0.x, point0.y);
                let point_j = Point2d::new(point1.x, point1.y);

                if let Some(point3) = sfm::triangulate_point(
                    &point_i,
                    &point_j,
                    &pose_i.to_matrix(),
                    &pose_j.to_matrix(),
                ) {
                    let point3 = nalgebra::Vector3::new(point3.x, point3.y, point3.z);
                    let point3 = rot_i * point3 + tvec_i;
                    if point3.z > 0.0 {
                        it.estimated_depth = point3.z;
                        log::info!("Triangulate depth: {}", it.estimated_depth);
                    } else {
                        it.estimated_depth = INIT_DEPTH;
                    }
                }
            } else if it.estimated_depth < 0.1 {
                it.estimated_depth = INIT_DEPTH;
            }

            // //
            // if it.point_features.len() < 4 {
            //     continue;
            // }

            // let i = it.start_frame;
            // let mut j = i - 1;
            // let size = it.point_features.len();
            // let mut svd_a = nalgebra::MatrixXx4::from_fn(2 * size, |_, _| 0.0);
            // let mut svd_index = 0;

            // let tvec0 = trans_vecs[i] + rot_mats[i] * imu_trans_to_cam;
            // let rot0 = rot_mats[i] * imu_rot_to_cam;

            // for pt in it.point_features.iter() {
            //     j += 1;

            //     let tvec1 = trans_vecs[j] + rot_mats[j] * imu_trans_to_cam;
            //     let rot1 = rot_mats[j] * imu_rot_to_cam;
            //     let r = (rot0.transpose() * rot1).transpose();
            //     let t = r * -(rot0.transpose() * (tvec1 - tvec0));
            //     let pose = nalgebra::Isometry3::from_parts(
            //         nalgebra::Translation3::from(t),
            //         nalgebra::UnitQuaternion::from_rotation_matrix(&r),
            //     )
            //     .to_matrix();
            //     let f = pt.0.point.normalize();
            //     let row1 = f.x * pose.row(2) - f.z * pose.row(0);
            //     let row2 = f.y * pose.row(2) - f.z * pose.row(1);
            //     svd_a.set_row(svd_index, &row1);
            //     svd_a.set_row(svd_index + 1, &row2);
            //     svd_index += 2;
            // }

            // assert!(svd_index == svd_a.nrows());
            // let svd = nalgebra::SVD::new(svd_a, false, true);
            // let v = svd.v_t.unwrap().transpose();
            // it.estimated_depth = v.column(3)[3];
        }

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
        frame_left: usize,
        frame_right: usize,
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
    pub fn remove_new(&mut self, frame_count: usize) {
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
    fn compensated_parallax2(&self, it_per_id: &FeaturePerId, frame_count: usize) -> f64 {
        // i: 倒数第二帧 j：倒数第一帧
        let i = frame_count - 2 - it_per_id.start_frame;
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
        frame_count: usize,                // 当前帧数
        points_features: &PointFeatureMap, // 最新帧的所有特征点
        td: u64,
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
