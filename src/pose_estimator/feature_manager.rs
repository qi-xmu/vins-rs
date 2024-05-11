use opencv::core::*;

use crate::{
    config::{FOCAL_LENGTH, MIN_PARALLAX},
    feature_trakcer::{PointFeature, PointFeatureMap},
};

#[derive(Debug, Default)]
struct FeaturePerFrame(PointFeature, f64);

#[derive(Debug, Default)]
struct FeaturePerId {
    /// 特征点的id
    pub feature_id: i32,
    /// 特征点的起始帧
    pub start_frame: i32,
    /// 该特征点的所有特征数据
    pub point_features: Vec<FeaturePerFrame>,
    pub estimated_depth: f64,
}
impl FeaturePerId {
    pub fn new(feature_id: i32, start_frame: i32) -> Self {
        Self {
            feature_id,
            start_frame,
            point_features: Vec::new(), // TODO:point_features
            estimated_depth: -1.0,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct FeatureManager {
    features: Vec<FeaturePerId>,

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
        cam_rot_mat: &nalgebra::Matrix3<f64>,
        cam_trans_vec: &nalgebra::Vector3<f64>,
        pts_2d: &Vector<Point2f>,
        // pts_3d: &Vector<Point3f>,
    ) -> bool {
        //

        false
    }
    /// initFramePoseByPnP
    pub fn init_frame_pose_by_pnp(
        &self,
        frame_count: i32,
        rot_mats: &[nalgebra::Matrix3<f64>],
        trans_vec: &[nalgebra::Vector3<f64>],
        imu_rot_to_cam: &nalgebra::Matrix3<f64>,
        imu_trans_to_cam: &nalgebra::Vector3<f64>,
    ) {
        if frame_count > 0 {
            let mut pts_2d = Vector::<Point2f>::default();
            let mut pts_3d = Vector::<Vec3f>::default(); // !!  Vector::<Point3f> error

            for it_per_id in self.features.iter() {
                if it_per_id.estimated_depth > 0.0 {
                    // 该特征点追踪的次数 - 1
                    let index = frame_count - it_per_id.start_frame;
                    // 说明该点一直追踪到了当前帧
                    if it_per_id.point_features.len() >= (index + 1) as usize {
                        //
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
                            it_per_id.point_features[index as usize].0.point.x as f32,
                            it_per_id.point_features[index as usize].0.point.y as f32,
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
            let cam_rot_mat = rot_mats[frame_count as usize - 1] * imu_rot_to_cam;
            let cam_trans_vec = rot_mats[frame_count as usize - 1] * imu_trans_to_cam
                + trans_vec[frame_count as usize - 1];
            // trans to w_T_cam: world to camera

            // TODO:solvePoseByPnP
            if self.solve_pose_by_pnp(&cam_rot_mat, &cam_trans_vec, &pts_2d) {
                //
            }

            //
        }
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
            // 2. 某一个特征点追踪开始的帧号 + 特征点追踪的帧数 - 1 >= 当前帧号 - 1 ==> 说明该特征点已经追踪到当前帧号
            if it_per_id.start_frame <= frame_count - 2
                && it_per_id.start_frame + (it_per_id.point_features.len() as i32) >= frame_count
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
}
