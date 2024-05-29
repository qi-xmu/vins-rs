pub struct Utility {}

impl Utility {
    #[inline]
    pub fn delta_quat(theta: nalgebra::Vector3<f64>) -> nalgebra::UnitQuaternion<f64> {
        let half_theta = theta / 2.0;
        let q = nalgebra::Quaternion::new(1.0, half_theta.x, half_theta.y, half_theta.z);
        nalgebra::UnitQuaternion::from_quaternion(q)
    }
}
