use opencv::core::Mat;

use super::{integration_base::IntegrationBase, EstimatorWindowsTrait, WINDOW_SIZE};

/// 根据帧号维护呀一个窗口大小为 [WINDOW_SIZE] 的时间戳窗口
#[derive(Debug, Default)]
pub(crate) struct EstimatorImageWindows {
    pub timestamps: [i64; WINDOW_SIZE + 1],
    // pub diff_times: [i64; WINDOW_SIZE + 1],
    pub images: [Mat; WINDOW_SIZE + 1],
    pub rot_mats: [nalgebra::Rotation3<f64>; WINDOW_SIZE + 1],
    pub trans_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
    pub vel_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
}

#[derive(Debug, Default)]
pub(crate) struct EstimatorIMUWindows {
    pub dt_buf: [i64; WINDOW_SIZE + 1],
    pub acce_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
    pub gyro_vecs: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
    pub bias_acces: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],
    pub bias_gyros: [nalgebra::Vector3<f64>; WINDOW_SIZE + 1],

    pub pre_integrations: [Option<IntegrationBase>; WINDOW_SIZE + 1],
}
impl EstimatorWindowsTrait for EstimatorImageWindows {
    fn forword(&mut self) {
        for i in 0..WINDOW_SIZE {
            self.timestamps.swap(i, i + 1);
            self.images.swap(i, i + 1);
            self.rot_mats.swap(i, i + 1);
            self.trans_vecs.swap(i, i + 1);
            self.vel_vecs.swap(i, i + 1);
        }
        self.timestamps[WINDOW_SIZE] = self.timestamps[WINDOW_SIZE - 1];
        self.images[WINDOW_SIZE] = self.images[WINDOW_SIZE - 1].clone();
        self.rot_mats[WINDOW_SIZE] = self.rot_mats[WINDOW_SIZE - 1];
        self.trans_vecs[WINDOW_SIZE] = self.trans_vecs[WINDOW_SIZE - 1];
        self.vel_vecs[WINDOW_SIZE] = self.vel_vecs[WINDOW_SIZE - 1];
    }

    fn clear(&mut self) {
        self.timestamps = [0; WINDOW_SIZE + 1];
        // ? images
        self.rot_mats = [nalgebra::Rotation3::identity(); WINDOW_SIZE + 1];
        self.trans_vecs = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
        self.vel_vecs = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
    }
}
impl EstimatorWindowsTrait for EstimatorIMUWindows {
    fn forword(&mut self) {
        for i in 0..WINDOW_SIZE {
            self.dt_buf.swap(i, i + 1);
            self.acce_vecs.swap(i, i + 1);
            self.gyro_vecs.swap(i, i + 1);
            self.bias_acces.swap(i, i + 1);
            self.bias_gyros.swap(i, i + 1);
            self.pre_integrations.swap(i, i + 1);
        }
        self.dt_buf[WINDOW_SIZE] = self.dt_buf[WINDOW_SIZE - 1];
        self.acce_vecs[WINDOW_SIZE] = self.acce_vecs[WINDOW_SIZE - 1];
        self.gyro_vecs[WINDOW_SIZE] = self.gyro_vecs[WINDOW_SIZE - 1];
        self.bias_acces[WINDOW_SIZE] = self.bias_acces[WINDOW_SIZE - 1];
        self.bias_gyros[WINDOW_SIZE] = self.bias_gyros[WINDOW_SIZE - 1];
        self.pre_integrations[WINDOW_SIZE] = self.pre_integrations[WINDOW_SIZE - 1].clone();
    }

    fn clear(&mut self) {
        self.dt_buf = [0; WINDOW_SIZE + 1];
        self.acce_vecs = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
        self.gyro_vecs = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
        self.bias_acces = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
        self.bias_gyros = [nalgebra::Vector3::zeros(); WINDOW_SIZE + 1];
        const ARRAY_REPEAT_VALUE: std::option::Option<IntegrationBase> = None;
        self.pre_integrations = [ARRAY_REPEAT_VALUE; WINDOW_SIZE + 1];
    }
}
