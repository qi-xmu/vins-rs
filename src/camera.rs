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

use opencv::core::{Point2d, Point3d};

pub trait CameraTrait: Default {
    fn lift_projective(&self, p: &Point2d, p3d: &mut Point3d);
}

#[derive(Debug, Default)]
pub struct CameraParameters {
    m_poly: Vec<f64>,
    m_inv_poly: Vec<f64>,
    m_c: f64,
    m_d: f64,
    m_e: f64,
    m_center_x: f64,
    m_center_y: f64,
}

#[derive(Debug, Default)]
pub struct MyCamera {
    m_parameters: CameraParameters,
    // other
    m_inv_scale: f64,
}
impl MyCamera {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CameraTrait for MyCamera {
    fn lift_projective(&self, p: &Point2d, p3d: &mut Point3d) {
        // todo!("implement")
    }
}
