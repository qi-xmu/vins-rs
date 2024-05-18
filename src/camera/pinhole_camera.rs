use opencv::core::{FileNodeTraitConst, FileStorageTrait, FileStorageTraitConst};

use opencv::core::{Point2d, Point3d};

use super::{CameraParametersTrait, CameraTrait};

#[derive(Debug, Default)]
pub struct PinholeParameters {
    pub camera_name: String,
    // size
    pub image_width: i32,
    pub image_height: i32,
    // intrinsic
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    // distortion
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
}

impl CameraParametersTrait for PinholeParameters {
    const CAMERA_TYPE: &'static str = "PINHOLE";
    fn read_from_yaml(path: &str) -> Self {
        let fs = opencv::core::FileStorage::new(
            path,
            opencv::core::FileStorage_Mode::READ as i32,
            "utf-8",
        )
        .unwrap();

        if fs.is_opened().unwrap() {
            if let Ok(model_type) = fs.get("model_type") {
                //
                let model_type = model_type.to_string().unwrap();
                if model_type != Self::CAMERA_TYPE {
                    log::error!("model type is not pinhole");
                    return Default::default();
                }
            }
            // image
            let camera_name = fs.get("camera_name").unwrap().to_string().unwrap();
            let image_width = fs.get("image_width").unwrap().to_i32().unwrap();
            let image_height = fs.get("image_height").unwrap().to_i32().unwrap();
            // distortion
            let distortion_parameters = fs.get("distortion_parameters").unwrap();
            let k1 = distortion_parameters.get("k1").unwrap().to_f64().unwrap();
            let k2 = distortion_parameters.get("k2").unwrap().to_f64().unwrap();
            let p1 = distortion_parameters.get("p1").unwrap().to_f64().unwrap();
            let p2 = distortion_parameters.get("p2").unwrap().to_f64().unwrap();
            // intrinsic
            let projection_parameters = fs.get("projection_parameters").unwrap();
            let fx = projection_parameters.get("fx").unwrap().to_f64().unwrap();
            let fy = projection_parameters.get("fy").unwrap().to_f64().unwrap();
            let cx = projection_parameters.get("cx").unwrap().to_f64().unwrap();
            let cy = projection_parameters.get("cy").unwrap().to_f64().unwrap();

            Self {
                camera_name,
                image_width,
                image_height,
                fx,
                fy,
                cx,
                cy,
                k1,
                k2,
                p1,
                p2,
            }
        } else {
            Default::default()
        }
    }

    fn write_to_yaml(&self, path: &str) -> anyhow::Result<()> {
        let mut fs = opencv::core::FileStorage::new(
            path,
            opencv::core::FileStorage_Mode::WRITE as i32,
            "utf-8",
        )?;

        fs.write_str("model_type", Self::CAMERA_TYPE)?;
        fs.write_str("camera_name", &self.camera_name)?;
        fs.write_i32("image_width", self.image_width)?;
        fs.write_i32("image_height", self.image_height)?;
        // intrinsic
        fs.start_write_struct("distortion_parameters", opencv::core::FileNode_MAP, "")?;
        fs.write_f64("k1", self.k1)?;
        fs.write_f64("k2", self.k2)?;
        fs.write_f64("p1", self.p1)?;
        fs.write_f64("p2", self.p2)?;
        fs.end_write_struct()?;

        // distortion
        fs.start_write_struct("projection_parameters", opencv::core::FileNode_MAP, "")
            .unwrap();
        fs.write_f64("fx", self.fx)?;
        fs.write_f64("fy", self.fy)?;
        fs.write_f64("cx", self.cx)?;
        fs.write_f64("cy", self.cy)?;
        fs.end_write_struct()?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct PinholeCamera {
    pub parameters: PinholeParameters,
    pub has_distortion: bool,
    pub inv_k11: f64,
    pub inv_k13: f64,
    pub inv_k22: f64,
    pub inv_k23: f64,
}

impl PinholeCamera {
    pub fn new(camera_file: &str) -> Self {
        let parameters = PinholeParameters::read_from_yaml(camera_file);
        let inv_k11 = 1.0 / parameters.fx;
        let inv_k13 = -parameters.cx / parameters.fx;
        let inv_k22 = 1.0 / parameters.fy;
        let inv_k23 = -parameters.cy / parameters.fy;
        let has_distortion = parameters.k1 != 0.0
            || parameters.k2 != 0.0
            || parameters.p1 != 0.0
            || parameters.p2 != 0.0;

        Self {
            parameters,
            inv_k11,
            inv_k13,
            inv_k22,
            inv_k23,
            has_distortion,
        }
    }

    #[inline]
    fn distortion(&self, p_u: (f64, f64)) -> (f64, f64) {
        //
        let k1 = self.parameters.k1;
        let k2 = self.parameters.k2;
        let p1 = self.parameters.p1;
        let p2 = self.parameters.p2;

        //
        let mx2_u = p_u.0 * p_u.0;
        let my2_u = p_u.1 * p_u.1;
        let mxy_u = p_u.0 * p_u.1;
        let rho2_u = mx2_u + my2_u;
        let rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
        let dx_u = p_u.0 * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u);
        let dy_u = p_u.1 * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

        (dx_u, dy_u)
    }
}

impl CameraTrait for PinholeCamera {
    fn lift_projective(&self, p: &Point2d) -> Point3d {
        // 归一化坐标
        let mx_d = self.inv_k11 * p.x + self.inv_k13;
        let my_d = self.inv_k22 * p.y + self.inv_k23;

        let (mx_u, my_u) = if !self.has_distortion {
            (mx_d, my_d)
        } else {
            let mut mx_u = mx_d;
            let mut my_u = my_d;
            for _ in 0..8 {
                let d_u = self.distortion((mx_u, my_u));
                mx_u = mx_d - d_u.0;
                my_u = my_d - d_u.1;
            }
            (mx_u, my_u)
        };
        Point3d::new(mx_u, my_u, 1.0)
    }

    fn get_camera_type(&self) -> super::CameraType {
        super::CameraType::Pinhole
    }
}

#[test]
fn read_from_yaml() {
    let parameters = PinholeParameters::read_from_yaml("configs/cam0_pinhole.yaml");
    println!("{:?}", parameters);
    parameters
        .write_to_yaml("configs/cam0_pinhole_write.yaml")
        .unwrap();
}
