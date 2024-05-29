use crate::{global_types::IMUData, utility::Utility};

use super::{ACC_N, ACC_W, GYR_N, GYR_W};

#[derive(Debug, Default, Clone)]
pub struct IntegrationBase {
    imu_unit_0: IMUData,
    imu_unit_1: IMUData,
    linearized_imu: IMUData,

    linearized_ba: nalgebra::Vector3<f64>,
    linearized_bg: nalgebra::Vector3<f64>,

    imu_buf: Vec<IMUData>,
    dt_buf: Vec<f64>, // 单位 s

    //
    jacobian: nalgebra::SMatrix<f64, 15, 15>,
    covariance: nalgebra::SMatrix<f64, 15, 15>,

    sum_dt: f64, // 单位 s
    delta_pos: nalgebra::Vector3<f64>,
    delta_vel: nalgebra::Vector3<f64>,
    delta_quat: nalgebra::UnitQuaternion<f64>,

    //
    noise: nalgebra::SMatrix<f64, 18, 18>,
}

impl IntegrationBase {
    pub fn new(
        imu_unit_0: &IMUData,
        linearized_ba: &nalgebra::Vector3<f64>,
        linearized_bg: &nalgebra::Vector3<f64>,
    ) -> Self {
        let mut noise = nalgebra::SMatrix::<f64, 18, 18>::zeros();
        let acc_n_2 = nalgebra::SMatrix::<f64, 3, 3>::identity() * ACC_N * ACC_N;
        let gyr_n_2 = nalgebra::SMatrix::<f64, 3, 3>::identity() * GYR_N * GYR_N;
        let acc_w_2 = nalgebra::SMatrix::<f64, 3, 3>::identity() * ACC_W * ACC_W;
        let gyr_w_2 = nalgebra::SMatrix::<f64, 3, 3>::identity() * GYR_W * GYR_W;
        noise.fixed_view_mut::<3, 3>(0, 0).copy_from(&acc_n_2);
        noise.fixed_view_mut::<3, 3>(3, 3).copy_from(&gyr_n_2);
        noise.fixed_view_mut::<3, 3>(6, 6).copy_from(&acc_n_2);
        noise.fixed_view_mut::<3, 3>(9, 9).copy_from(&gyr_n_2);
        noise.fixed_view_mut::<3, 3>(12, 12).copy_from(&acc_w_2);
        noise.fixed_view_mut::<3, 3>(15, 15).copy_from(&gyr_w_2);

        let imu_unit_0 = imu_unit_0.clone();
        let linearized_ba = linearized_ba.clone();
        let linearized_bg = linearized_bg.clone();

        Self {
            linearized_imu: imu_unit_0.clone(),
            imu_unit_0,
            linearized_ba,
            linearized_bg,
            noise,
            ..Default::default()
        }
    }

    pub fn push_back(&mut self, dt: i64, imu_unit: &IMUData) {
        //
        let dt = dt as f64 / 1e9;
        self.dt_buf.push(dt);
        self.imu_buf.push(imu_unit.clone());
        self.propagate(dt, imu_unit);
    }

    fn propagate(&mut self, dt: f64, imu_unit: &IMUData) {
        self.imu_unit_1 = imu_unit.clone();
        self.mid_point_integrate(dt, true);

        self.sum_dt += dt;
        self.imu_unit_0 = imu_unit.to_owned();
    }

    pub fn repropagate(
        &mut self,
        linearized_ba: nalgebra::Vector3<f64>,
        linearized_bg: nalgebra::Vector3<f64>,
    ) {
        //
        self.sum_dt = 0.0;
        self.imu_unit_0 = self.linearized_imu.clone();
        self.delta_pos = nalgebra::Vector3::zeros();
        self.delta_vel = nalgebra::Vector3::zeros();
        self.delta_quat = nalgebra::UnitQuaternion::identity();
        self.jacobian = nalgebra::SMatrix::<f64, 15, 15>::identity();
        self.covariance = nalgebra::SMatrix::<f64, 15, 15>::zeros();
        self.linearized_ba = linearized_ba;
        self.linearized_bg = linearized_bg;

        for i in 0..self.dt_buf.len() {
            // FIXME：有待优化复制
            self.propagate(self.dt_buf[i], &self.imu_buf[i].to_owned());
        }
    }

    fn mid_point_integrate(&mut self, dt: f64, update_jacobian: bool) {
        //
        let delta_quat = self.delta_quat;
        let acce_0 = self.imu_unit_0.acc;
        let gyro_0 = self.imu_unit_0.gyro;
        let acce_1 = self.imu_unit_1.acc;
        let gyro_1 = self.imu_unit_1.gyro;

        let linearized_ba = self.linearized_ba;
        let linearized_bg = self.linearized_bg;
        // 计算四元数
        let un_gyro_0 = 0.5 * (gyro_0 + gyro_1) - linearized_bg;
        let dq = Utility::delta_quat(un_gyro_0 * dt);
        let res_delta_quat = delta_quat * dq;
        // 计算加速度
        let un_acce_0 = delta_quat * (acce_0 - linearized_ba);
        let un_acce_1 = res_delta_quat * (acce_1 - linearized_ba);
        let un_acce = 0.5 * (un_acce_0 + un_acce_1);

        // 更改状态
        self.delta_quat = res_delta_quat;
        self.delta_pos = self.delta_pos + self.delta_vel * dt + 0.5 * un_acce * dt * dt;
        self.delta_vel = self.delta_vel + un_acce * dt;

        // TODO: 未修改？
        // self.linearized_ba = linearized_ba;
        // self.linearized_ba = linearized_bg;

        // 更新雅可比矩阵
        if update_jacobian {
            let w_x = 0.5 * (gyro_0 + gyro_1) - linearized_bg;
            let a_0_x = acce_0 - linearized_ba;
            let a_1_x = acce_1 - linearized_ba;
            let rot_w_x = vec2skew_symmetric_matrix(&w_x);
            let rot_a_0_x = vec2skew_symmetric_matrix(&a_0_x);
            let rot_a_1_x = vec2skew_symmetric_matrix(&a_1_x);

            let delta_quat_mat = delta_quat.to_rotation_matrix().matrix().to_owned();
            let res_delta_quat_mat = res_delta_quat.to_rotation_matrix().matrix().to_owned();

            // F.block<3, 3>(0, 0) = Matrix3d::Identity();
            // F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * _dt * _dt +
            //                       -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt * _dt;
            // F.block<3, 3>(0, 6) = MatrixXd::Identity(3,3) * _dt;
            // F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt * _dt;
            // F.block<3, 3>(0, 12) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * _dt * -_dt;
            // F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * _dt;
            // F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3,3) * _dt;
            // F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * _dt +
            //                       -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * _dt) * _dt;
            // F.block<3, 3>(6, 6) = Matrix3d::Identity();
            // F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * _dt;
            // F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * _dt * -_dt;
            // F.block<3, 3>(9, 9) = Matrix3d::Identity();
            // F.block<3, 3>(12, 12) = Matrix3d::Identity();
            #[allow(non_snake_case)]
            let mut F = nalgebra::SMatrix::<f64, 15, 15>::identity();
            F.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&nalgebra::SMatrix::<f64, 3, 3>::identity());
            F.fixed_view_mut::<3, 3>(0, 3).copy_from(
                &(delta_quat_mat * rot_a_0_x * dt * dt * -0.25
                    + res_delta_quat_mat
                        * rot_a_1_x
                        * (nalgebra::Matrix3::identity() - rot_w_x * dt)
                        * dt
                        * dt
                        * -0.25),
            );
            F.fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(nalgebra::Matrix3::identity() * dt));
            F.fixed_view_mut::<3, 3>(0, 9)
                .copy_from(&((delta_quat_mat + res_delta_quat_mat) * dt * dt * -0.25));
            F.fixed_view_mut::<3, 3>(0, 12)
                .copy_from(&(res_delta_quat_mat * rot_a_1_x * dt * dt * -dt * -0.25));

            F.fixed_view_mut::<3, 3>(3, 3)
                .copy_from(&(nalgebra::Matrix3::identity() - rot_w_x * dt));
            F.fixed_view_mut::<3, 3>(3, 12)
                .copy_from(&(nalgebra::Matrix3::identity() * -dt));
            F.fixed_view_mut::<3, 3>(6, 3).copy_from(
                &(delta_quat_mat * rot_a_0_x * dt * -0.5
                    + res_delta_quat_mat
                        * rot_a_1_x
                        * (nalgebra::Matrix3::identity() - rot_w_x * dt)
                        * dt
                        * -0.5),
            );
            F.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&nalgebra::Matrix3::identity());
            F.fixed_view_mut::<3, 3>(6, 9)
                .copy_from(&((delta_quat_mat + res_delta_quat_mat) * dt * -0.5));
            F.fixed_view_mut::<3, 3>(6, 12)
                .copy_from(&(res_delta_quat_mat * rot_a_1_x * dt * -dt * -0.5));

            F.fixed_view_mut::<3, 3>(9, 9)
                .copy_from(&nalgebra::Matrix3::identity());
            F.fixed_view_mut::<3, 3>(12, 12)
                .copy_from(&nalgebra::Matrix3::identity());

            // V.block<3, 3>(0, 0) =  0.25 * delta_q.toRotationMatrix() * _dt * _dt;
            // V.block<3, 3>(0, 3) =  0.25 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * _dt * 0.5 * _dt;
            // V.block<3, 3>(0, 6) =  0.25 * result_delta_q.toRotationMatrix() * _dt * _dt;
            // V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
            // V.block<3, 3>(3, 3) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            // V.block<3, 3>(3, 9) =  0.5 * MatrixXd::Identity(3,3) * _dt;
            // V.block<3, 3>(6, 0) =  0.5 * delta_q.toRotationMatrix() * _dt;
            // V.block<3, 3>(6, 3) =  0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x  * _dt * 0.5 * _dt;
            // V.block<3, 3>(6, 6) =  0.5 * result_delta_q.toRotationMatrix() * _dt;
            // V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
            // V.block<3, 3>(9, 12) = MatrixXd::Identity(3,3) * _dt;
            // V.block<3, 3>(12, 15) = MatrixXd::Identity(3,3) * _dt;
            #[allow(non_snake_case)]
            let mut V = nalgebra::SMatrix::<f64, 15, 18>::zeros();
            V.fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(delta_quat_mat * dt * dt * 0.25));
            let v03_33 = -res_delta_quat_mat * rot_a_1_x * dt * dt * 0.5 * dt * 0.25;
            V.fixed_view_mut::<3, 3>(0, 3).copy_from(&v03_33);
            V.fixed_view_mut::<3, 3>(0, 6)
                .copy_from(&(res_delta_quat_mat * dt * dt * 0.25));
            V.fixed_view_mut::<3, 3>(0, 9).copy_from(&v03_33);
            let v33_33 = nalgebra::Matrix3::identity() * dt * 0.5;
            V.fixed_view_mut::<3, 3>(3, 3).copy_from(&v33_33);
            V.fixed_view_mut::<3, 3>(3, 9).copy_from(&v33_33);
            V.fixed_view_mut::<3, 3>(6, 0)
                .copy_from(&(delta_quat_mat * dt * 0.5));
            let v63_33 = -res_delta_quat_mat * rot_a_1_x * dt * 0.5 * dt * 0.5;
            V.fixed_view_mut::<3, 3>(6, 3).copy_from(&v63_33);
            V.fixed_view_mut::<3, 3>(6, 6)
                .copy_from(&(res_delta_quat_mat * dt * 0.5));
            V.fixed_view_mut::<3, 3>(6, 9).copy_from(&v63_33);
            let v912_33 = nalgebra::Matrix3::identity() * dt;
            V.fixed_view_mut::<3, 3>(9, 12).copy_from(&v912_33);
            V.fixed_view_mut::<3, 3>(12, 15).copy_from(&v912_33);

            self.jacobian = F * self.jacobian;
            self.covariance = F * self.covariance * F.transpose() + V * self.noise * V.transpose();
        }
    }
}

fn vec2skew_symmetric_matrix(vec: &nalgebra::Vector3<f64>) -> nalgebra::SMatrix<f64, 3, 3> {
    nalgebra::SMatrix::<f64, 3, 3>::new(
        0.0, -vec[2], vec[1], vec[2], 0.0, -vec[0], -vec[1], vec[0], 0.0,
    )
}

#[test]
fn test_view_copy_from() {
    let mut mat = nalgebra::SMatrix::<f64, 3, 3>::identity();
    let mat2 = nalgebra::SMatrix::<f64, 3, 3>::from_fn(|x, y| (x + y) as f64);

    println!("mat = {}", mat);
    println!("mat2 = {}", mat2);

    mat.fixed_view_mut::<3, 3>(0, 0).copy_from(&mat2);

    println!("mat = {}", mat);
}
