# VINS in Rust

This is a Rust implementation of the VINS algorithm. The original implementation is in C++ and can be found [here](https://github.com/lturing/vins_fusion_pangolin)

这是一个VINS算法的Rust实现。原始实现是C++的。原始代码在[这里](https://github.com/lturing/vins_fusion_pangolin)

# 目标

- [x] EuRoC 数据集读取 (EuRoC dataset reader) 2024/05/12
- [x] Pinhole相机模型 (Pinhole camera model) 2024/05/09
- [x] 实现特征识别和追踪 (Feature detection and tracking) 2024/05/08
- [ ] 实现视觉里程计 (Visual odometry)
  - [ ] 初始化 (Initialization)
  - [ ] 三角化 (Triangulation)
  - [ ] EPnP (EPnP)
  - [ ] 滑动窗口边缘化 (Sliding window marginalization)
- [ ] IMU
  - [ ] 实现IMU预积分 (IMU preintegration)
- [ ] 多线程并行 (Multi-threading)
# Dependencies

- [opencv](https://docs.rs/opencv/latest/opencv/index.html)
