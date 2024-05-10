use std::{io::BufRead, path::Path};

use opencv::prelude::MatTraitConstManual;

use super::DatasetTrait;

#[derive(Debug, Default)]
pub struct EuRoCDataset {
    pub cam0s: Vec<(f64, String)>,
    pub cam1s: Vec<(f64, String)>,
    pub imu0s: Vec<(f64, [f64; 6])>, // t, gyro and acce
}
impl EuRoCDataset {
    const EUROC_CAM0_PATH: &str = "mav0/cam0/";
    const EUROC_CAM1_PATH: &str = "mav0/cam1/";
    const EUROC_IMU0_PATH: &str = "mav0/imu0/";

    pub fn read_imu(path: &Path) -> Vec<(f64, [f64; 6])> {
        let csv_path = path.join("data.csv");
        let mut reader = csv::Reader::from_path(csv_path).unwrap();
        reader
            .records()
            .map(|record| {
                let record = record.unwrap();
                let timestamp = record[0].parse::<f64>().unwrap();
                let gyro_acce = [
                    record[1].parse::<f64>().unwrap(),
                    record[2].parse::<f64>().unwrap(),
                    record[3].parse::<f64>().unwrap(),
                    record[4].parse::<f64>().unwrap(),
                    record[5].parse::<f64>().unwrap(),
                    record[6].parse::<f64>().unwrap(),
                ];
                (timestamp, gyro_acce)
            })
            .collect()
    }
    pub fn read_cam(path: &Path) -> Vec<(f64, String)> {
        let csv_path = path.join("data.csv");
        let data_path = path.join("data");
        let mut reader = csv::Reader::from_path(csv_path).unwrap();
        reader
            .records()
            .map(|record| {
                let record = record.unwrap();
                let timestamp = record[0].parse::<f64>().unwrap();
                let name = record[1].to_string();
                let path = data_path.join(name).to_str().unwrap().to_string();
                (timestamp, path)
            })
            .collect()
    }

    pub fn new(path: &str) -> Self {
        let path = std::path::Path::new(path);
        let cam0_base_path = path.join(EuRoCDataset::EUROC_CAM0_PATH);
        let cam1_base_path = path.join(EuRoCDataset::EUROC_CAM1_PATH);
        let imu0_base_path = path.join(EuRoCDataset::EUROC_IMU0_PATH);
        // read cam0
        let cam0s = Self::read_cam(&cam0_base_path);
        // read cam1
        let cam1s = Self::read_cam(&cam1_base_path);
        // read imu0
        let imu0s = Self::read_imu(&imu0_base_path);
        Self {
            cam0s,
            cam1s,
            imu0s,
        }
    }
}
impl DatasetTrait for EuRoCDataset {
    //

    fn read_t_cam0_list(&self) -> &Vec<(f64, String)> {
        &self.cam0s
    }
}

#[cfg(test)]
mod tests {
    use crate::dataset::DatasetTrait as _;
    #[test]
    fn test_read_cam() {
        let cam_path = "/home/qi/V201/mav0/cam0";
        let cam_path = std::path::Path::new(cam_path);
        let _ = super::EuRoCDataset::read_cam(cam_path);
    }
    #[test]
    fn test_read_imu() {
        let imu_path = "/home/qi/V201/mav0/imu0";
        let imu_path = std::path::Path::new(imu_path);
        let _ = super::EuRoCDataset::read_imu(imu_path);
    }
    #[test]
    fn test_dataset() {
        let path = "/home/qi/V201";
        let dataset = super::EuRoCDataset::new(path);
        let _cam0 = dataset.read_t_cam0_list();
    }
}
