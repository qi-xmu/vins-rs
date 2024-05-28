#[derive(Debug, Clone, Default)]
pub struct IMUData {
    pub timestamp: u64,
    pub acc: nalgebra::Vector3<f64>,
    pub gyro: nalgebra::Vector3<f64>,
}
