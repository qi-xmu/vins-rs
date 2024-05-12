use crate::feature_trakcer::PointFeatureMap;

#[derive(Debug, Default)]
pub struct IntegrationBase();

// TODO:ImageFrame
#[derive(Debug, Default)]
pub struct ImageFrame {
    pub timestamp: f64,
    pub is_key_frame: bool,
    pub points: PointFeatureMap,

    pub pre_integration: IntegrationBase,

    /// 旋转矩阵
    pub rot_matrix: nalgebra::Rotation3<f64>,
    /// 平移向量
    pub trans_vector: nalgebra::Vector3<f64>,
}

impl ImageFrame {
    #[allow(dead_code)]
    pub fn new(timestamp: f64, points: &PointFeatureMap) -> Self {
        Self {
            timestamp,
            points: points.clone(),
            ..Default::default()
        }
    }
}
