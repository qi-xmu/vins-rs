// struct Camera {
//     position: Vec3,
//     direction: Vec3,
//     up: Vec3,
//     right: Vec3,
//     fov: f32,
//     aspect_ratio: f32,
//     near: f32,
//     far: f32,
// }

mod pinhole_camera;
pub use pinhole_camera::{PinholeCamera, PinholeParameters};

use opencv::core::{Point2d, Point3d};

#[allow(dead_code)]
pub enum CameraType {
    /// - 通用的相机模型，适用于普通相机、广角相机和鱼眼相机。
    /// - 使用多项式函数来描述图像光心到投影点的距离和角度之间的关系。
    KannalaBrandt(),
    /// - Mei模型是一种鱼眼相机模型，用于描述鱼眼镜头的畸变特性。
    /// - 通常使用四个多项式来建模径向畸变和两个多项式来建模切向畸变。
    Mei(),
    /// - 最简单的相机模型，假设光线沿直线传播，不考虑镜头畸变。
    /// - 简单，只包含内参（焦距、主点等）。适用于轻度畸变或高质量的相机系统。
    Pinhole,
    /// - 这是Pinhole模型的扩展，考虑了镜头的畸变。
    /// - 除了内参外，还包括畸变参数（径向畸变、切向畸变等）。
    PinholeFull(),
    /// - 这是一种用于鱼眼相机的模型，特别适用于超广角和全景相机。
    /// - 使用多项式函数来描述径向畸变，并提供了一种有效的畸变校正方法。
    /// - 在需要高精度畸变校正的鱼眼相机应用中使用。
    Scaramuzza(),
}

/// 相机的trait
pub trait CameraTrait: Default {
    fn lift_projective(&self, p: &Point2d, p3d: &mut Point3d);
    fn get_camera_type(&self) -> CameraType;
    // fn read_parameters(&mut self);
}

/// 相机参数的trait
pub trait CameraParametersTrait: Default {
    const CAMERA_TYPE: &'static str;
    fn read_from_yaml(path: &str) -> Self;
    fn write_to_yaml(&self, path: &str) -> anyhow::Result<()>;
}
