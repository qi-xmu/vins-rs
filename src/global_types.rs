#![allow(dead_code)]

#[derive(Debug, Clone, Default)]
pub struct IMUData {
    pub timestamp: i64,
    pub acc: nalgebra::Vector3<f64>,
    pub gyro: nalgebra::Vector3<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Timestamp(i64);
impl Timestamp {
    pub fn as_sec(&self) -> f64 {
        self.0 as f64 / 1e9
    }
    /// dt
    pub fn duration_since(&self, other: &Timestamp) -> f64 {
        (self.0 - other.0) as f64 / 1e9
    }
}

#[test]
fn test_timestamp() {
    let t1 = Timestamp(1);
    let t2 = Timestamp(2);
    let t3 = t1.0 + t2.0;
    let t4 = t2.0 - t1.0;
    println!("t3 = {}, t4 = {}", t3, t4);
}
