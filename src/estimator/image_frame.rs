use crate::feature_trakcer::FeatureFrameMap;

pub struct ImageFrame {
    pub timestamp: f64,
    pub is_key_frame: bool,
    pub points: FeatureFrameMap,

    pub R: nalgebra::Matrix3<f64>,
}
