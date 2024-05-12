use crate::{camera::CameraTrait, config::*};
use opencv::{
    core::*,
    imgproc::{COLOR_GRAY2BGR, LINE_8},
    prelude::*,
};
use std::collections::HashMap;

/// 特征点跟踪器
/// 用于跟踪图像中的特征点。
/// 该结构体会在内部维护一个状态，用于跟踪特征点。
#[derive(Debug, Default)]
pub struct FeatureTracker<Camera>
where
    Camera: CameraTrait,
{
    /* 时间 */
    /// 上一帧的时间
    prev_time: f64,
    /// 当前帧的时间
    cur_time: f64,

    /* 图像 */
    /// 图像的行数
    row: i32,
    /// 图像的列数
    col: i32,
    /// 遮罩：用于屏蔽距离过近的特征点
    /// ? 是否可以使用局部变量？
    mask: Mat,
    /// 标记特征点的图像
    img_track: Mat,
    /// 上一帧图像
    prev_img: Mat,
    /// 当前帧图像
    cur_img: Mat,

    /* 特征点 */
    /// good_features_to_track 新增的特征点
    /// ? 是否可以使用局部变量？
    n_pts: Vector<Point2f>,
    /// 上一帧识别的特征点
    prev_pts: Vector<Point2f>,
    /// 当前帧识别的特征点
    cur_pts: Vector<Point2f>,
    /// 预测下一帧的特征点
    predict_pts: Vector<Point2f>,
    /// 上一帧 投影到归一化平面上的特征点
    prev_un_pts: Vector<Point2f>,
    /// 当前帧 投影到归一化平面上的特征点
    cur_un_pts: Vector<Point2f>,

    /* 状态 */
    /// 是否有预测点 predict_pts 是否可用。
    has_predicted: bool,

    /* Map */
    /// 上一帧 id 和特征点的映射
    prev_un_pts_map: HashMap<i32, Point2f>,
    /// 当前帧 id 和特征点的映射
    /// ? 是否可以使用局部变量？
    cur_un_pts_map: HashMap<i32, Point2f>,
    prev_left_pts_map: HashMap<i32, Point2f>,

    /* 计数 */
    /// 每一个新增的特征点分配一个新的 id，用于标记特征点
    id_cnt: i32,
    /// 特征点的 id
    feature_ids: Vector<i32>,
    /// track count: 记录每一个特征点被跟踪的次数。
    track_cnt: Vector<i32>,

    /// 相机 用于将特征点投影到归一化平面上
    camera: Camera,
}

impl<Camera> FeatureTracker<Camera>
where
    Camera: CameraTrait,
{
    /// 创建一个新的特征点跟踪器。
    /// # Arguments
    /// * `camera` - 相机模型，用于将特征点投影到归一化平面上。
    pub fn new_with_camera(camera: Camera) -> Self {
        Self {
            id_cnt: 0,
            camera: camera,
            has_predicted: false,
            ..Default::default()
        }
    }

    /// 创建一个默认的特征点跟踪器。
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            id_cnt: 0,
            has_predicted: false,
            ..Default::default()
        }
    }
    /// 根据距离过滤特征点
    fn set_mask(&mut self) {
        let mut mask =
            Mat::new_rows_cols_with_default(self.row, self.col, CV_8UC1, Scalar::from(255))
                .unwrap();

        let mut cnt_pts_id = vec![];
        for i in 0..self.cur_pts.len() {
            cnt_pts_id.push((
                self.track_cnt.get(i).unwrap(),
                self.cur_pts.get(i).unwrap(),
                self.feature_ids.get(i).unwrap(),
            ))
        }
        // 按照 track_cnt 降序排列 从多到少
        cnt_pts_id.sort_by(|x, y| y.0.cmp(&x.0));

        self.feature_ids.clear();
        self.track_cnt.clear();
        self.cur_pts.clear();

        // 屏蔽距离过近的特征点
        for it in cnt_pts_id.iter() {
            let it_pt = Point2i::new(it.1.x as i32, it.1.y as i32);
            let m: &u8 = mask.at_2d(it_pt.y, it_pt.x).unwrap();
            if *m == 255 {
                self.track_cnt.push(it.0);
                self.cur_pts.push(it.1);
                self.feature_ids.push(it.2);
                opencv::imgproc::circle(&mut mask, it_pt, MIN_DIST, Scalar::from(0), -1, LINE_8, 0)
                    .unwrap();
            }
        }
        // ? 设置 mask 后续没有使用
        self.mask = mask;
    }

    /// 过滤 得到status为true的集合
    #[inline]
    fn reduce_vector_point(v: &Vector<Point2f>, status: &Vector<bool>) -> Vector<Point2f> {
        status
            .iter()
            .zip(v.iter())
            .filter(|(state, _)| *state)
            .map(|(_, p)| p)
            .collect()
    }

    /// 过滤 得到status为true的集合
    #[inline]
    fn reduce_vector_i32(v: &Vector<i32>, status: &Vector<bool>) -> Vector<i32> {
        status
            .iter()
            .zip(v.iter())
            .filter(|(state, _)| *state)
            .map(|(_, p)| p)
            .collect()
    }

    /// 判断点是否在图像边界。
    #[inline]
    fn in_border(&self, pt: &Point2f) -> bool {
        let img_x = pt.x.round() as i32;
        let img_y = pt.y.round() as i32;

        img_x >= BORDER_SIZE
            && img_x < self.col - BORDER_SIZE
            && img_y >= BORDER_SIZE
            && img_y < self.row - BORDER_SIZE
    }

    /// 计算两个点之间的欧几里得距离。
    #[inline]
    fn distance(a: &Point2f, b: &Point2f) -> f64 {
        // 计算两点在 x 轴和 y 轴上的差值
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        // 计算并返回两点间的距离
        (dx * dx + dy * dy).sqrt() as f64
    }

    #[inline]
    fn undistorted_pts(&self, pts: &Vector<Point2f>, cam: &impl CameraTrait) -> Vector<Point2f> {
        let mut undistorted_pts = Vector::<Point2f>::new();
        for pt in pts.iter() {
            let x = pt.x;
            let y = pt.y;
            let p = Point2d::new(x as f64, y as f64);
            let p3d = cam.lift_projective(&p);
            let new_pt = Point2f::new((p3d.x / p3d.z) as f32, (p3d.y / p3d.z) as f32);
            undistorted_pts.push(new_pt);
        }
        undistorted_pts
    }

    /// 计算特征点的速度。
    /// # Arguments
    /// * `dt` - 两帧图像之间的时间间隔。
    /// * `ids` - 特征点的id。
    /// * `pts` - 特征点的图像坐标。
    /// * `cur_id_pts` - id 和当前帧特征点的映射。
    /// * `prev_id_pts` - id 和上一帧特征点的映射。
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
        // 将 id 和当前帧特征点的映射存储到 cur_id_pts 中
        ids.iter().zip(pts.iter()).for_each(|(id, pt)| {
            cur_id_pts.insert(id, pt);
        });

        if !prev_id_pts.is_empty() {
            // 上一帧有特征点，计算匹配特征点的速度
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
            // 上一帧没有特征点，速度为0
            for _ in 0..pts.len() {
                pts_velocity.push(Point2f::new(0.0, 0.0));
            }
        }

        pts_velocity
    }

    /// 绘制特征点。
    /// 特征点的颜色根据其被跟踪的次数来决定。
    /// 次数： 少 ----> 多
    /// 颜色： 蓝 ----> 红
    ///
    /// 绘制前后帧特征点之间的连接。（绿色）
    ///
    /// # Returns
    /// 返回绘制好的图像。
    #[inline]
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

        // 绘制特征点之间的连接
        self.feature_ids
            .iter()
            .zip(self.cur_pts.iter())
            .for_each(|(id, cur_pt)| {
                if let Some(prev_pt) = self.prev_left_pts_map.get(&id) {
                    opencv::imgproc::arrowed_line(
                        &mut img_track,
                        Point2i::new(prev_pt.x as i32, prev_pt.y as i32),
                        Point2i::new(cur_pt.x as i32, cur_pt.y as i32),
                        Scalar::from((0, 255, 0)), // green
                        1,
                        LINE_8,
                        0,
                        0.2,
                    )
                    .unwrap();
                }
            });
        img_track // 返回绘制好的图像
    }

    /// 跟踪图像中的特征点。
    /// # Arguments
    /// * `timestamp` - 当前时间。
    /// * `img` - 当前图像。
    ///
    /// # Returns
    /// 返回一个包含特征点的 [FeatureFrame] 结构体。
    ///
    /// # Example
    /// ```rust
    /// let mut feature_tracker = FeatureTracker::new();
    /// let img = imgcodecs::imread("path/to/image.jpg", imgcodecs::IMREAD_GRAYSCALE).unwrap();
    /// let feature_frame = feature_tracker.track_image(0.0, &img);
    /// ```
    /// # Note
    /// 该方法会在内部维护一个状态，用于跟踪特征点。
    /// 该方法会返回一个 `FeatureFrame` 结构体，其中包含了当前图像中的特征点。
    pub fn track_image(&mut self, timestamp: f64, img: &Mat) -> FeatureFrame {
        self.cur_time = timestamp; // 当前时间
        self.cur_img = img.clone(); // 当前图像
        self.row = img.rows(); // 图像行数
        self.col = img.cols(); // 图像列数
        if true {
            // CLAHE 图像均衡化
            let mut clahe = opencv::imgproc::create_clahe(3.0, Size::new(8, 8)).unwrap();
            clahe.apply(&img, &mut self.cur_img).unwrap();
        }
        // 首先清空 cur_pts
        self.cur_pts.clear();

        if self.prev_pts.len() > 0 {
            // 如果有上一帧的特征点，计算光流
            let mut status = Vector::<u8>::new();
            let mut err = Vector::<f32>::new();
            if self.has_predicted {
                // 如果有预测点，使用预测点作为初始点，可以适当的加快光流的计算速度。
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
                    Size::new(21, 21),
                    1,
                    criteria,
                    opencv::video::OPTFLOW_USE_INITIAL_FLOW,
                    1e-4,
                )
                .unwrap();
                // 计算成功的特征点数量
                let succ_num = status.iter().filter(|s| *s != 0).count();
                // 如果成功的特征点数量小于 10，重新计算光流
                if succ_num < 10 {
                    opencv::video::calc_optical_flow_pyr_lk(
                        &self.prev_img,
                        &self.cur_img,
                        &self.prev_pts,
                        &mut self.cur_pts,
                        &mut status,
                        &mut err,
                        Size::new(21, 21),
                        3,
                        TermCriteria::default().unwrap(),
                        opencv::video::DISOpticalFlow_PRESET_ULTRAFAST,
                        1e-4,
                    )
                    .unwrap();
                }
            } else {
                // 没有预测点，直接计算光流
                opencv::video::calc_optical_flow_pyr_lk(
                    &self.prev_img,
                    &self.cur_img,
                    &self.prev_pts,
                    &mut self.cur_pts,
                    &mut status,
                    &mut err,
                    Size::new(21, 21),
                    3,
                    TermCriteria::default().unwrap(),
                    opencv::video::DISOpticalFlow_PRESET_ULTRAFAST,
                    1e-4,
                )
                .unwrap();
            }

            // [x] reverse check --> Vector<u8>
            let status = if FLOW_BACK {
                let mut reverse_status = Vector::<u8>::new();
                let mut reverse_pts = self.prev_pts.clone();
                opencv::video::calc_optical_flow_pyr_lk(
                    &self.cur_img,
                    &self.prev_img,
                    &self.cur_pts,
                    &mut reverse_pts,
                    &mut reverse_status,
                    &mut err,
                    Size::new(11, 11),
                    1,
                    TermCriteria::default().unwrap(),
                    opencv::video::OPTFLOW_USE_INITIAL_FLOW,
                    1e-4,
                )
                .unwrap();

                for i in 0..status.len() {
                    let val = if status.get(i).unwrap() != 0
                        && reverse_status.get(i).unwrap() != 0
                        && Self::distance(
                            &self.prev_pts.get(i).unwrap(),
                            &reverse_pts.get(i).unwrap(),
                        ) <= 0.5
                    {
                        1
                    } else {
                        0
                    };
                    status.set(i, val).unwrap();
                }
                status
            } else {
                status
            };

            let status = status
                .iter()
                .zip(self.cur_pts.iter())
                .map(|(s, pt)| s != 0 && self.in_border(&pt))
                .collect();

            // [x] reduceVector
            self.prev_pts = Self::reduce_vector_point(&self.prev_pts, &status);
            self.cur_pts = Self::reduce_vector_point(&self.cur_pts, &status);
            self.feature_ids = Self::reduce_vector_i32(&self.feature_ids, &status);
            self.track_cnt = Self::reduce_vector_i32(&self.track_cnt, &status);
        }

        // 重新计算 track_cnts = track_cnts + 1
        self.track_cnt = self
            .track_cnt
            .iter()
            .map(|x| x + 1)
            .collect::<Vector<i32>>();

        // 计算预测点
        if true {
            // 设置图像遮罩
            self.set_mask();
            // 检查是否需要添加新的特征点
            let n_max_cnt = MAX_CNT - self.cur_pts.len() as i32;
            if n_max_cnt > 0 {
                log::debug!("n_max_cnt={}", n_max_cnt);
                opencv::imgproc::good_features_to_track(
                    &self.cur_img,
                    &mut self.n_pts,
                    n_max_cnt, // 最大识别的特征点数量
                    0.01,
                    MIN_DIST as f64,
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
                self.track_cnt.push(1);
                self.cur_pts.push(p);
                self.feature_ids.push(self.id_cnt);
                self.id_cnt += 1;
            }
        }
        // 计算特征点的归一化坐标
        self.cur_un_pts = self.undistorted_pts(&self.cur_pts, &self.camera);
        // 计算特征点在图像上的速度
        let pts_velocity = Self::pts_velocity(
            self.cur_time - self.prev_time,
            &self.feature_ids,
            &self.cur_pts,
            &mut self.cur_un_pts_map,
            &self.prev_un_pts_map,
        );
        // 绘制特征点
        self.img_track = self.draw_track();

        // 更新状态
        // ? 是否可以交换
        self.prev_img = self.cur_img.clone();
        self.prev_pts = self.cur_pts.clone();
        self.prev_un_pts = self.cur_un_pts.clone();
        self.prev_time = self.cur_time;
        self.has_predicted = false;

        // 更新 prev_left_pts_map
        self.prev_left_pts_map.clear();
        self.feature_ids
            .iter()
            .zip(self.cur_pts.iter())
            .for_each(|(feature_id, pt)| {
                self.prev_left_pts_map.insert(feature_id, pt);
            });

        let mut point_features = HashMap::<i32, PointFeature>::new();
        for i in 0..self.feature_ids.len() {
            let feautre_id = self.feature_ids.get(i).unwrap();

            let x = self.cur_un_pts.get(i).unwrap().x;
            let y = self.cur_un_pts.get(i).unwrap().y;
            let z = 1.0f64;
            let p_u = self.cur_pts.get(i).unwrap().x;
            let p_v = self.cur_pts.get(i).unwrap().y;
            let veloctiry_x = pts_velocity.get(i).unwrap().x;
            let veloctiry_y = pts_velocity.get(i).unwrap().y;
            let ff = PointFeature {
                camera_id: 0,
                point: nalgebra::Vector3::new(x as f64, y as f64, z as f64),
                uv: nalgebra::Vector2::new(p_u as f64, p_v as f64),
                velocity: nalgebra::Vector2::new(veloctiry_x as f64, veloctiry_y as f64),
            };

            point_features.insert(feautre_id, ff);
        }

        FeatureFrame {
            timestamp,
            point_features,
            image: self.cur_img.clone(),
        }
    }

    pub fn get_track_image(&self) -> &Mat {
        &self.img_track
    }
}

/// 特征点包含的特征信息
/// 包括特征点在归一化平面的坐标、图像坐标、速度以及相机id。
#[derive(Debug, Default, Clone, Copy)]
pub struct PointFeature {
    /// 相机id
    pub camera_id: u8,
    /// 投影到归一化平面的坐标
    pub point: nalgebra::Vector3<f64>,
    /// 图像坐标
    pub uv: nalgebra::Vector2<f64>,
    /// 速度
    pub velocity: nalgebra::Vector2<f64>,
}

pub type PointFeatureMap = HashMap<i32, PointFeature>;

/// 一个帧的所有特征信息
/// 包括帧的时间戳、特征点的所有特征信息以及图像。
#[derive(Debug, Default, Clone)]
pub struct FeatureFrame {
    /// 当前帧时间戳
    pub timestamp: f64,
    /// 当前帧中的特征点的所有特征信息，包括特征点在归一化平面的坐标、图像坐标、速度以及相机id。
    pub point_features: PointFeatureMap,
    /// 当前帧的图像
    pub image: Mat,
}

#[cfg(test)]
mod tests {
    // fn test_feature_tracker() {}
}
