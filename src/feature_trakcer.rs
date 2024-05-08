use std::collections::HashMap;

use crate::{camera::CameraTrait, config::*};
use opencv::{
    core::*,
    imgproc::{COLOR_GRAY2BGR, LINE_8},
    prelude::*,
};

//

#[derive(Debug, Default)]
pub struct FeatureTracker<Camera>
where
    Camera: CameraTrait,
{
    img_track: Mat,
    mask: Mat,
    prev_img: Mat,
    cur_img: Mat,
    n_pts: Vector<Point2f>,
    prev_pts: Vector<Point2f>,
    cur_pts: Vector<Point2f>,
    predict_pts: Vector<Point2f>,
    cur_time: f64,
    prev_time: f64,
    has_predicted: bool,

    // un
    prev_un_pts: Vector<Point2f>,
    cur_un_pts: Vector<Point2f>,
    // HashMap
    prev_un_pts_map: HashMap<i32, Point2f>,
    cur_un_pts_map: HashMap<i32, Point2f>,
    prev_left_pts_map: HashMap<i32, Point2f>,
    //
    //
    n_id: i32,
    ids: Vector<i32>,
    track_cnt: Vector<i32>,
    // image size
    row: i32,
    col: i32,
    // camera
    m_camera: Camera,
}

impl<Camera> FeatureTracker<Camera>
where
    Camera: CameraTrait,
{
    pub fn new() -> Self {
        Self {
            n_id: 0,
            has_predicted: false,
            ..Default::default()
        }
    }

    fn set_mask(&mut self) {
        let mask = Mat::zeros(self.row, self.col, CV_8UC1).unwrap();
        let mut cnt_pts_id = vec![];

        for i in 0..self.cur_pts.len() {
            cnt_pts_id.push((
                self.track_cnt.get(i).unwrap(),
                self.cur_pts.get(i).unwrap(),
                self.ids.get(i).unwrap(),
            ))
        }
        cnt_pts_id.sort_by(|x, y| x.0.cmp(&y.0));

        self.cur_pts.clear();
        self.ids.clear();
        self.track_cnt.clear();

        // for it in cnt_pts_id.iter() {
        //     mask.

        // //     // if (mask.)
        // }
    }

    /// 删除状态为0的点。
    fn reduce_vector_point(v: &Vector<Point2f>, status: &Vector<u8>) -> Vector<Point2f> {
        status
            .iter()
            .zip(v.iter())
            .filter(|(state, _)| *state != 0)
            .map(|(_, p)| p)
            .collect()

        // v.iter()
        //     .zip(status.iter())
        //     .filter(|(_, status)| *status != 0)
        //     .map(|(p, _)| p)
        //     .collect::<Vector<Point2f>>()
    }

    fn reduce_vector_i32(v: &Vector<i32>, status: &Vector<u8>) -> Vector<i32> {
        status
            .iter()
            .zip(v.iter())
            .filter(|(state, _)| *state != 0)
            .map(|(_, p)| p)
            .collect()
    }

    /// 判断点是否在图像边界内。
    fn in_border(&self, pt: &Point2f) -> bool {
        const BORDER_SIZE: i32 = 1;
        let img_x = pt.x.round() as i32;
        let img_y = pt.y.round() as i32;

        img_x >= BORDER_SIZE
            && img_x < self.col - BORDER_SIZE
            && img_y >= BORDER_SIZE
            && img_y < self.row - BORDER_SIZE
    }

    fn distance(a: &Point2f, b: &Point2f) -> f32 {
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        (dx * dx + dy * dy).sqrt()
    }

    fn undistorted_pts(&self, pts: &Vector<Point2f>, cam: &impl CameraTrait) -> Vector<Point2f> {
        let mut undistorted_pts = Vector::<Point2f>::new();
        for pt in pts.iter() {
            let x = pt.x;
            let y = pt.y;
            let p = Point2d::new(x as f64, y as f64);
            let mut p3d = Point3d::default();
            cam.lift_projective(&p, &mut p3d);

            undistorted_pts.push(Point2f::new(x, y));
        }
        undistorted_pts
    }

    fn pts_velocity(
        // &self,
        dt: f64,
        ids: &Vector<i32>,
        pts: &Vector<Point2f>,
        cur_id_pts: &mut HashMap<i32, Point2f>,
        prev_id_pts: &HashMap<i32, Point2f>,
    ) -> Vector<Point2f> {
        let mut pts_velocity = Vector::<Point2f>::new();
        cur_id_pts.clear();
        ids.iter().zip(pts.iter()).for_each(|(id, pt)| {
            cur_id_pts.insert(id, pt);
        });

        if !prev_id_pts.is_empty() {
            for i in 0..pts.len() {
                let v_pt = if let Some(it) = prev_id_pts.get(&ids.get(i).unwrap()) {
                    let v_x = (pts.get(i).unwrap().x - it.x) as f64 / dt;
                    let v_y = (pts.get(i).unwrap().y - it.y) as f64 / dt;
                    Point2f::new(v_x as f32, v_y as f32)
                } else {
                    Point2f::new(0.0, 0.0)
                };
                pts_velocity.push(v_pt);
            }
        } else {
            for _ in 0..pts.len() {
                pts_velocity.push(Point2f::new(0.0, 0.0));
            }
        }

        pts_velocity
    }

    fn draw_track(&self) -> Mat {
        let mut img_track = self.cur_img.clone();
        opencv::imgproc::cvt_color(&self.cur_img, &mut img_track, COLOR_GRAY2BGR, 0).unwrap();

        // 绘制特征点
        self.cur_pts
            .iter()
            .zip(self.track_cnt.iter())
            .for_each(|(pt, tc)| {
                // let len = std::cmp::min(1.0, 1.0 * tc / 20);
                let len = 255.min(tc * 255 / 20);
                opencv::imgproc::circle(
                    &mut img_track,
                    Point2i::new(pt.x as i32, pt.y as i32),
                    2,
                    Scalar::from((255 - len, 0, len)),
                    2,
                    LINE_8,
                    0,
                )
                .unwrap();
            });

        // 绘制对应位置
        self.ids
            .iter()
            .zip(self.cur_pts.iter())
            .for_each(|(id, pt)| {
                if let Some(map_it) = self.prev_left_pts_map.get(&id) {
                    opencv::imgproc::arrowed_line(
                        &mut img_track,
                        Point2i::new(pt.x as i32, pt.y as i32),
                        Point2i::new(map_it.x as i32, map_it.y as i32),
                        Scalar::from((0, 255, 0)),
                        1,
                        LINE_8,
                        0,
                        0.2,
                    )
                    .unwrap();
                }
            });
        img_track
    }

    pub fn track_image(&mut self, _cur_time: f64, img: &Mat) -> FeatureFrame {
        self.cur_time = _cur_time; // 当前时间
        self.cur_img = img.clone(); // 当前图像
        self.row = img.rows(); // 图像行数
        self.col = img.cols(); // 图像列数
        {
            // CLAHE 图像均衡化
            let mut clahe = opencv::imgproc::create_clahe(3.0, Size::new(8, 8)).unwrap();
            clahe.apply(&img, &mut self.cur_img).unwrap();
        }
        // 首先清空 cur_pts
        self.cur_pts.clear();

        if self.prev_pts.len() > 0 {
            let mut status = Vector::<u8>::new();
            let mut err = Vector::<f32>::new();
            if self.has_predicted {
                self.cur_pts = self.predict_pts.clone();
                let criteria = opencv::core::TermCriteria::new(
                    opencv::core::TermCriteria_EPS + opencv::core::TermCriteria_COUNT,
                    30,
                    0.01,
                )
                .unwrap();
                opencv::video::calc_optical_flow_pyr_lk(
                    &self.prev_img,
                    &self.cur_img,
                    &self.prev_pts,
                    &mut self.cur_pts,
                    &mut status,
                    &mut err,
                    Size::new(11, 11),
                    3,
                    criteria,
                    opencv::video::OPTFLOW_USE_INITIAL_FLOW,
                    1e-4,
                )
                .unwrap();
                let mut succ_num = 0;
                for s in status.iter() {
                    if s != 0 {
                        succ_num += 1;
                    }
                }
                if succ_num < 10 {
                    opencv::video::calc_optical_flow_pyr_lk(
                        &self.prev_img,
                        &self.cur_img,
                        &self.prev_pts,
                        &mut self.cur_pts,
                        &mut status,
                        &mut err,
                        Size::new(11, 11),
                        3,
                        TermCriteria::default().unwrap(),
                        0,
                        1e-4,
                    )
                    .unwrap();
                }
            } else {
                opencv::video::calc_optical_flow_pyr_lk(
                    &self.prev_img,
                    &self.cur_img,
                    &self.prev_pts,
                    &mut self.cur_pts,
                    &mut status,
                    &mut err,
                    Size::new(11, 11),
                    3,
                    TermCriteria::default().unwrap(),
                    0,
                    1e-4,
                )
                .unwrap();
            }
            //  其他
            for i in 0..self.cur_pts.len() {
                if status.get(i).unwrap() != 0 && false {
                    status.set(i, 0).unwrap();
                }
            }

            // TODO: in_border
            let status = status
                .iter()
                .zip(self.cur_pts.iter())
                .map(|(s, pt)| if s != 0 && !self.in_border(&pt) { 0 } else { s })
                .collect();

            // TODO: reduceVector

            self.prev_pts = Self::reduce_vector_point(&self.prev_pts, &status);
            self.cur_pts = Self::reduce_vector_point(&self.cur_pts, &status);
            self.ids = Self::reduce_vector_i32(&self.ids, &status);
            self.track_cnt = Self::reduce_vector_i32(&self.track_cnt, &status);
        }

        // 重新计算 track_cnt
        self.track_cnt = self
            .track_cnt
            .iter()
            .map(|x| x + 1)
            .collect::<Vector<i32>>();

        //
        if true {
            // self.set_mask();
            let n_max_cnt = MAX_CNT - self.cur_pts.len() as i32;
            if n_max_cnt > 0 {
                if self.mask.empty() {
                    println!("mask is empty");
                }
                if self.mask.typ() != CV_8UC1 {
                    println!("mask is not CV_8UC1");
                }
                // goodFeaturesToTrack --> n_pts
                opencv::imgproc::good_features_to_track(
                    &self.cur_img,
                    &mut self.n_pts,
                    n_max_cnt,
                    0.01,
                    MIN_DIST, // TODO
                    &self.mask,
                    3,
                    false,
                    0.04,
                )
                .unwrap();
            } else {
                self.n_pts.clear();
            }

            for p in self.n_pts.iter() {
                self.cur_pts.push(p);
                self.ids.push(self.n_id);
                self.n_id += 1;
                self.track_cnt.push(1);
            }
        }

        self.cur_un_pts = self.undistorted_pts(&self.cur_pts, &self.m_camera);
        // TODO: ptsVelocity
        Self::pts_velocity(
            self.cur_time - self.prev_time,
            &self.ids,
            &self.cur_pts,
            &mut self.cur_un_pts_map,
            &self.prev_un_pts_map,
        );
        // TODO: drawTrack
        self.img_track = self.draw_track();

        self.prev_img = self.cur_img.clone();
        self.prev_pts = self.cur_pts.clone();
        self.prev_un_pts = self.cur_un_pts.clone();
        self.prev_time = self.cur_time;
        self.has_predicted = false;

        self.prev_left_pts_map.clear();

        // HashMap
        self.ids.iter().zip(self.cur_pts.iter()).for_each(|(k, v)| {
            self.prev_left_pts_map.insert(k, v);
        });

        let feature_frame = FeatureFrame(0f64, 0f64, 0f64);
        return feature_frame;
    }

    pub fn get_track_image(&self) -> &Mat {
        &self.img_track
    }
}

pub struct FeatureFrame(f64, f64, f64);
