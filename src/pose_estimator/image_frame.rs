use crate::feature_trakcer::PointFeatureMap;

use super::integration_base::IntegrationBase;

// TODO ImageFrame
#[derive(Debug, Default)]
pub struct ImageFrame {
    /// 时间戳
    pub timestamp: i64,
    /// 是否为关键帧
    pub is_key_frame: bool,
    /// 特征点
    pub points: PointFeatureMap,
    /// 预积分
    pub pre_integration: IntegrationBase,
    /// 旋转矩阵
    pub rot_matrix: nalgebra::Rotation3<f64>,
    /// 平移向量
    pub trans_vector: nalgebra::Vector3<f64>,
}

impl ImageFrame {
    #[allow(dead_code)]
    pub fn new(timestamp: i64, points: &PointFeatureMap) -> Self {
        Self {
            timestamp,
            points: points.clone(),
            ..Default::default()
        }
    }
}
