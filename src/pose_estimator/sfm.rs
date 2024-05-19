use std::collections::HashMap;

use crate::global_cast;
use opencv::core::{Mat, Point2d, Point3d, Vector};
pub type SFMFeatureList = Vec<SFMFeature>; // feature_id : SFMFeature

#[derive(Debug, Clone, Default)]
pub struct SFMFeature {
    pub state: bool,
    pub feature_id: i32,
    pub start_frame: usize,
    pub point2s: Vector<Point2d>, // frame_number : point feature
    pub point3: Point3d,
}

impl SFMFeature {
    pub fn end_frame_id(&self) -> usize {
        self.start_frame + self.point2s.len() - 1
    }
    pub fn get_by_fid(&self, frame_id: usize) -> Option<Point2d> {
        if frame_id >= self.start_frame {
            let index = frame_id - self.start_frame;
            // 如果超出范围，返回None
            self.point2s.get(index).ok()
        } else {
            None
        }
    }
}

pub fn triangulate_point(
    point_i: &Point2d,
    point_j: &Point2d,
    pose_i: &nalgebra::Matrix4<f64>,
    pose_j: &nalgebra::Matrix4<f64>,
) -> Option<Point3d> {
    let row0 = point_i.x as f64 * pose_i.row(2) - pose_i.row(0);
    let row1 = point_i.y as f64 * pose_i.row(2) - pose_i.row(1);
    let row2 = point_j.x as f64 * pose_j.row(2) - pose_j.row(0);
    let row3 = point_j.y as f64 * pose_j.row(2) - pose_j.row(1);
    let design_matrix = nalgebra::Matrix4::<f64>::from_rows(&[row0, row1, row2, row3]);

    let v = nalgebra::SVD::new(design_matrix, false, true)
        .v_t
        .unwrap()
        .transpose();
    let tri_pt = v.column(3);
    let point3f = Point3d::new(
        &tri_pt[0] / &tri_pt[3],
        &tri_pt[1] / &tri_pt[3],
        &tri_pt[2] / &tri_pt[3],
    );
    Some(point3f)
}

// triangulateTwoFrames
fn triangulate_two_frames(
    i: usize,
    j: usize,
    pose_i: &nalgebra::Matrix4<f64>,
    pose_j: &nalgebra::Matrix4<f64>,
    features: &mut SFMFeatureList,
) {
    assert!(i != j);
    for it in features.iter_mut() {
        if it.state {
            continue;
        }
        let point_i = it.get_by_fid(i);
        let point_j = it.get_by_fid(j);
        if let (Some(point_i), Some(point_j)) = (point_i, point_j) {
            if let Some(point3) = triangulate_point(&point_i, &point_j, pose_i, pose_j) {
                it.point3 = point3;
                it.state = true;
            }
        }
    }
}

pub fn solve_pnp(
    object_points: &Vector<Point3d>,
    image_points: &Vector<Point2d>,
    quat: &nalgebra::UnitQuaternion<f64>,
    tvec: &nalgebra::Vector3<f64>,
) -> Option<(nalgebra::UnitQuaternion<f64>, nalgebra::Vector3<f64>)> {
    let k = Mat::from_slice_2d(&crate::config::K).unwrap();
    let d = Mat::default();

    let mut rvec: Mat = global_cast::Quaterniond(quat.clone()).into();
    let mut tvec: Mat = global_cast::Vector3d(tvec.clone()).into();

    if !opencv::calib3d::solve_pnp(
        object_points,
        image_points,
        &k,
        &d,
        &mut rvec,
        &mut tvec,
        true, // 是否使用rvec和tvec作为初始值
        opencv::calib3d::SOLVEPNP_ITERATIVE,
    )
    .unwrap()
    {
        None
    } else {
        let rvec = global_cast::Vector3d::from(rvec).0;
        let tvec = global_cast::Vector3d::from(tvec).0;
        Some((
            nalgebra::UnitQuaternion::from_axis_angle(
                &nalgebra::Unit::new_normalize(rvec),
                rvec.norm(),
            ),
            tvec,
        ))
    }
}

//
// solveFrameByPnP
fn solve_frame_by_pnp(
    i: usize,
    init_quat: &nalgebra::UnitQuaternion<f64>,
    init_trans: &nalgebra::Vector3<f64>,
    features: &mut SFMFeatureList,
) -> Option<(nalgebra::UnitQuaternion<f64>, nalgebra::Vector3<f64>)> {
    let mut matched = (Vector::<Point3d>::default(), Vector::<Point2d>::default());
    for it in features.iter_mut() {
        if !it.state {
            continue;
        }
        if let Some(point2) = it.get_by_fid(i) {
            matched.1.push(point2);
            matched.0.push(it.point3);
        }
    }
    if matched.0.len() < 15 {
        return None;
    }

    solve_pnp(&matched.0, &matched.1, init_quat, init_trans)
}

pub fn sfm_construct(
    start: usize,
    end: usize,
    rel_rot: nalgebra::Rotation3<f64>,
    rel_trans: nalgebra::Vector3<f64>,
    features: &mut SFMFeatureList,
) -> Option<(Vec<nalgebra::Isometry3<f64>>, HashMap<i32, Point3d>)> {
    let feature_number = features.len();
    let mut c_poses = vec![nalgebra::Isometry3::<f64>::identity(); end];
    c_poses[start] = nalgebra::Isometry3::<f64>::from_parts(
        nalgebra::Translation::default(),
        nalgebra::UnitQuaternion::identity(),
    );
    c_poses[end - 1] = nalgebra::Isometry3::<f64>::from_parts(
        nalgebra::Translation::from(-(rel_rot * rel_trans)),
        nalgebra::UnitQuaternion::from(rel_rot),
    );

    // 1: trangulate between l ----- frame_num - 1
    // 2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1;
    for i in start..end - 1 {
        if i > start {
            let init_quat = c_poses[i - 1].rotation;
            let init_trans = c_poses[i - 1].translation.vector;
            if let Some((quat, trans)) = solve_frame_by_pnp(i, &init_quat, &init_trans, features) {
                c_poses[i] = nalgebra::Isometry3::<f64>::from_parts(
                    nalgebra::Translation::from(trans),
                    quat,
                );
            } else {
                return None;
            }
        }
        triangulate_two_frames(
            i,
            end - 1,
            &c_poses[i].to_matrix(),
            &c_poses[end - 1].to_matrix(),
            features,
        );
    }

    // 3: triangulate l-----l+1 l+2 ... frame_num -2
    for i in start + 1..end - 1 {
        triangulate_two_frames(
            start,
            i,
            &c_poses[start].to_matrix(),
            &c_poses[i].to_matrix(),
            features,
        );
    }

    // 4: solve pnp l-1; triangulate l-1 ----- l
    //              l-2              l-2 ----- l
    for i in (0..start).rev() {
        if let Some((quat, trans)) = solve_frame_by_pnp(
            i,
            &c_poses[i + 1].rotation,
            &c_poses[i + 1].translation.vector,
            features,
        ) {
            c_poses[i] =
                nalgebra::Isometry3::<f64>::from_parts(nalgebra::Translation::from(trans), quat);
            triangulate_two_frames(
                i,
                start,
                &c_poses[i].to_matrix(),
                &c_poses[start].to_matrix(),
                features,
            )
        } else {
            return None;
        }
    }

    // 5: triangulate all other points
    for i in 0..feature_number {
        if features[i].state {
            continue;
        }
        if features[i].point2s.len() >= 2 {
            let frame_0 = features[i].start_frame;
            let frame_1 = features[i].end_frame_id();

            let point_0 = features[i].get_by_fid(frame_0).unwrap();
            let point_1 = features[i].get_by_fid(frame_1).unwrap();
            if let Some(point3) = triangulate_point(
                &point_0,
                &point_1,
                &c_poses[frame_0].to_matrix(),
                &c_poses[frame_1].to_matrix(),
            ) {
                features[i].point3 = point3;
            }
        }
    }
    // 展示结果
    // for i in 0..end {
    //     if features[i].state {
    //         print!("feature_id: {} ", features[i].feature_id);
    //         println!("point3: {:?}", features[i].point3);
    //     }
    // }

    // full BA

    // false
    let track_points: HashMap<i32, Point3d> = features
        .iter()
        .filter(|it| it.state)
        .map(|it| (it.feature_id, it.point3))
        .collect();
    Some((c_poses, track_points))
}

#[cfg(test)]
mod tests {
    use nalgebra::*;
    use opencv::core::{Mat, MatTraitConst};
    #[test]
    fn test_svd() {
        //
        let mat = matrix![1.0, 0.0 ,1.0; 0.0, 1.0, 1.0; 0.0, 0.0, 0.0];
        let svd = SVD::new(mat, true, true);
        let u = svd.u.unwrap(); // U
        let s = svd.singular_values; // ∑
        let v_t = svd.v_t.unwrap(); // V^T

        println!("u: {:?}", u);
        println!("s: {:?}", s);
        println!("v_t: {:?}", v_t);
    }

    #[test]
    fn test_transpose() {
        let mat = matrix![1.0, 0.0 ,1.0; 0.0, 1.0, 1.0; 0.0, 0.0, 0.0];
        let mat_t = mat.transpose();
        println!("mat: {}", mat);
        println!("mat_t: {}", mat_t);
    }

    #[test]
    fn test_k() {
        let k = Mat::from_slice_2d(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();

        println!("k: {:?}", k);
        for i in 0..3 {
            for j in 0..3 {
                print!("{} ", k.at_2d::<f64>(i, j).unwrap());
            }
            println!();
        }
    }

    #[test]
    fn test_transform() {
        //
        let mut m4 = nalgebra::Matrix4::<f64>::identity();
        let m3 = nalgebra::matrix![1.0, 2.0, 3.0; 4.0, 5.0, 6.0; 7.0, 8.0, 9.0];
        let v3 = nalgebra::Vector3::<f64>::new(1.0, 2.0, 3.0);

        // 替换m3 --> m4
        m4.fixed_view_mut::<3, 1>(0, 3).copy_from(&v3);
        m4.fixed_view_mut::<3, 3>(0, 0).copy_from(&m3);

        println!("m4: {}", m4);
    }
}
