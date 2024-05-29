#![allow(dead_code)]

/* 静态常量：无需更改 */
pub const K: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/* Feature Tracker */
/// 特征点最大数量
pub static MAX_CNT: i32 = 150;
/// 特征点最小距离 单位：像素
pub static MIN_DIST: i32 = 30;
/// 反向光流，然后匹配正向光流，进行特征点筛选。
pub static FLOW_BACK: bool = true;
/// 是否使用多线程
pub static MULTIPLE_THREAD: bool = true;
/// 边界的宽度。边界的宽度是指在图像的边界上不会检测特征点。
pub static BORDER_SIZE: i32 = 5;

/* Estimator */
/// 窗口大小
pub const WINDOW_SIZE: usize = 10;

/// 是否使用IMU
pub static USE_IMU: bool = false;
/// 相机焦距
pub static FOCAL_LENGTH: f64 = 460.0;
/// 窗口内最小视差
pub static MIN_PARALLAX: f64 = 0.5;
/// INIT_DEPTH
pub static INIT_DEPTH: f64 = 5.0;

/* 传感器误差 */
pub static ACC_N: f64 = 0.0;
pub static ACC_W: f64 = 0.0;
pub static GYR_N: f64 = 0.0;
pub static GYR_W: f64 = 0.0;
