use opencv::core::{Point2d, Point3d};

use crate::feature_trakcer::{PointFeature, PointFeatureMap};

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
        }
    }
}

#[derive(Debug, Default)]
pub struct FeatureManager {
    features: Vec<FeaturePerId>,
    //
}

impl FeatureManager {
    // initFramePoseByPnP
    fn init_frame_pose_by_pnp(
        &self,
        frame_count: i32,
        Rs: &Vec<i32>,
        ts: &Vec<PointFeature>,
        tic: &Vec<i32>,
        ric: &Vec<i32>,
    ) {
        //
        if frame_count > 0 {
            let mut pts_2d = Vec::<Point2d>::new();
            let mut pts_3d = Vec::<Point3d>::new();

            //
            for it_per_id in self.features.iter() {
                //
                if it_per_id.estimated_depth > 0.0 {
                    //
                    let index = frame_count - it_per_id.start_frame;
                    if it_per_id.point_features.len() >= (index + 1) as usize {
                        //
                        // TODO:let pts_in_cam
                        // TODO:let pts_in_world

                        pts_2d.push(Point2d::default());
                        pts_3d.push(Point3d::default());
                    }
                }
            }
            // camera R and t
            let R = nalgebra::Matrix3::<f64>::identity();
            let t = nalgebra::Vector3::<f64>::zeros();
            // trans to w_T_cam: world to camera

            // TODO:solvePoseByPnP
            if false {
                //
            }

            //
        }
    }
    /// compensatedParallax2 视差补偿
    fn compensated_parallax2(&self, it_per_id: &FeaturePerId, frame_count: i32) -> f64 {
        //
        let i = (frame_count - it_per_id.start_frame - 2) as usize;
        let frame_i = &it_per_id.point_features[i];
        let frame_j = &it_per_id.point_features[i + 1];

        // p_j
        let p_j = frame_j.0.point;
        let (u_j, v_j) = (p_j.0 / p_j.2, p_j.1 / p_j.2);

        // p_i
        let p_i = frame_i.0.point;
        let (u_i, v_i) = (p_i.0 / p_i.2, p_i.1 / p_i.2);
        let (du, dv) = (u_i - u_j, v_i - v_j);

        // p_i_comp
        let p_i_comp = p_i.clone();
        let (u_i_comp, v_i_comp) = (p_i_comp.0 / p_i_comp.2, p_i_comp.1 / p_i_comp.2);
        let (du_comp, dv_comp) = (u_i_comp - u_j, v_i_comp - v_j);

        let a = (du * du + dv * dv) as f64;
        let b = (du_comp * du_comp + dv_comp * dv_comp) as f64;
        a.min(b).sqrt().max(0.0f64)
    }
    /// 添加特征点检查视差
    pub fn add_feature_check_parallax(
        &mut self,
        frame_count: i32,
        points_map: &PointFeatureMap,
        td: f64,
    ) -> bool {
        //
        let mut parallax_sum = 0.0;
        let mut parallax_num = 0;

        let mut new_feature_num = 0;
        let mut last_track_num = 0;
        let mut long_track_num = 0; // 特征点数量大于4的数量

        points_map.iter().for_each(|(k, v)| {
            // TODO:FeaturePerFrame
            let f_per_frame = FeaturePerFrame(v.clone(), td);
            let feature_id = *k; // 特征点的id
            let find_it = self
                .features
                .iter_mut()
                .find(|feature| feature.feature_id == feature_id);
            if let Some(f_per_id) = find_it {
                // 已有的特征点追踪，添加一个新的 PointFeature
                f_per_id.point_features.push(f_per_frame);
                last_track_num += 1;
                // ?: 特征点数量大于4
                if f_per_id.point_features.len() > 4 {
                    long_track_num += 1;
                }
            } else {
                // 创建一个新的特征点进行追踪
                let mut f_per_id = FeaturePerId::new(feature_id, frame_count);
                // 添加特征点
                f_per_id.point_features.push(f_per_frame);
                self.features.push(f_per_id);
                new_feature_num += 1;
            }
        });

        if frame_count < 2
            || last_track_num < 20
            || long_track_num < 40
            || new_feature_num > last_track_num / 2
        {
            return true;
        };

        self.features.iter().for_each(|it_per_id| {
            if it_per_id.start_frame <= frame_count - 2
                && it_per_id.start_frame + (it_per_id.point_features.len() as i32) - 1
                    >= frame_count - 1
            {
                // TODO:compensatedParallax2
                parallax_sum += self.compensated_parallax2(it_per_id, frame_count);
                parallax_num += 1;
            }
        });

        false
    }
}
