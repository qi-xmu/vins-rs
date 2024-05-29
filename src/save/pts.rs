use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::feature_trakcer::FeatureFrame;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct FramePointsSave {
    timestamp: i64,
    pub point_features: HashMap<i32, ((i32, i32), (f64, f64, f64))>,
}

impl From<FeatureFrame> for FramePointsSave {
    fn from(frame_points_save: FeatureFrame) -> Self {
        let mut map = HashMap::new();
        for (k, v) in frame_points_save.point_features {
            map.insert(
                k,
                (
                    (v.uv.x as i32, v.uv.y as i32),
                    (v.point.x, v.point.y, v.point.z),
                ),
            );
        }

        Self {
            timestamp: frame_points_save.timestamp,
            point_features: map,
        }
    }
}
