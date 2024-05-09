#![allow(dead_code)]

pub const MAX_CNT: i32 = 150;
pub const MIN_DIST: i32 = 30;
/// 反向光流，然后匹配正向光流，进行特征点筛选。
pub const FLOW_BACK: bool = true;
/// 是否使用多线程
pub const MULTIPLE_THREAD: bool = true;

/// 是否使用IMU
pub const USE_IMU: bool = false;

/// 窗口大小
pub const WINDOW_SIZE: i32 = 10;
