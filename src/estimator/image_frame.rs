use crate::feature_trakcer::FeatureFrameMap;

#[derive(Debug, Default)]
pub struct ImageFrame {
    pub timestamp: f64,
    pub is_key_frame: bool,
    pub points: FeatureFrameMap,

    /// 旋转矩阵
    pub R: nalgebra::Matrix3<f64>,
    /// 平移向量
    pub t: nalgebra::Vector3<f64>,
}

impl ImageFrame {
    #[allow(dead_code)]
    pub fn new(timestamp: f64, points: FeatureFrameMap) -> Self {
        Self {
            timestamp,
            points,
            ..Default::default()
        }
    }
}
