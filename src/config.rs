#![allow(dead_code)]

/* Feature Tracker */
/// 特征点最大数量
pub const MAX_CNT: i32 = 150;
/// 特征点最小距离 单位：像素
pub const MIN_DIST: i32 = 30;
/// 反向光流，然后匹配正向光流，进行特征点筛选。
pub const FLOW_BACK: bool = true;
/// 是否使用多线程
pub const MULTIPLE_THREAD: bool = true;
/// 边界的宽度。边界的宽度是指在图像的边界上不会检测特征点。
pub const BORDER_SIZE: i32 = 5;

/* Estimator */
/// 是否使用IMU
pub const USE_IMU: bool = false;
/// 窗口大小
pub const WINDOW_SIZE: i32 = 10;
/// 相机焦距
pub const FOCAL_LENGTH: f64 = 460.0;
/// 窗口内最小视差
pub static MIN_PARALLAX: f64 = 0.5;
