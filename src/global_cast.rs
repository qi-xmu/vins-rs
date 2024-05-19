//! 全局类型转换
//!
//! 用于将全局类型转换为其他类型
//!
//! 时间： 2024-05-18
//! 作者： qi-xmu
//! 版本： 0.1.0
//!

use nalgebra::*;
use opencv::core::{Mat, CV_64F};
use opencv::prelude::*;
use std::fmt::Display;

/// 组合 [Rotation3] 和 [Vector3] 为 [Isometry3]
#[test]
fn test_isometry3() {
    let rotation = Rotation3::identity();
    let translation = Vector3::new(1.0, 2.0, 3.0);
    let isometry = Isometry3::from_parts(
        Translation::from(translation),
        UnitQuaternion::from(rotation),
    );
    println!("isometry: {}", isometry.to_matrix());
}

#[derive(Debug, Clone, Default)]
pub struct Quaterniond(pub UnitQuaternion<f64>);

impl Display for Quaterniond {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Into<Mat> for Quaterniond {
    fn into(self) -> Mat {
        let rvec = if let Some(axis) = self.0.axis() {
            self.0.angle() * axis.into_inner()
        } else {
            Default::default()
        };
        Into::<Mat>::into(Vector3d(rvec))
    }
}

/// 实现 Display trait 用于打印
pub struct MatPrinter(pub Mat);

impl Display for MatPrinter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let rows = self.0.rows();
        let cols = self.0.cols();
        write!(f, "\n")?;
        for i in 0..rows {
            for j in 0..cols {
                write!(f, "{}, ", self.0.at_2d::<f64>(i, j).unwrap())?;
            }
            write!(f, "\n")?;
        }
        // write!(f, "({}, {})\n", rows, cols)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct Matrix3d(pub Matrix3<f64>);

/// 实现 Display trait 用于打印
impl Display for Matrix3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 将 [Mat] 转换为 [Matrix]
impl From<Mat> for Matrix3d {
    fn from(mat: Mat) -> Self {
        let rows = mat.rows();
        let cols = mat.cols();
        assert!(rows == 3 && cols == 3);
        let mut matrix = Matrix3::<f64>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                matrix[(i, j)] = *mat.at_2d::<f64>(i as i32, j as i32).unwrap();
            }
        }
        Matrix3d(matrix)
    }
}

/// 将 [Matrix] 转换为 [Mat]
impl Into<Mat> for Matrix3d {
    fn into(self) -> Mat {
        let mut mat = Mat::zeros_nd(&[3, 3], CV_64F).unwrap().to_mat().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                *mat.at_2d_mut::<f64>(i as i32, j as i32).unwrap() = self.0[(i, j)];
            }
        }
        mat
    }
}

#[test]
fn test_mat_to_matrix() {
    let mat = Mat::from_slice_2d(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    let matrix = Matrix3d::from(mat);
    println!("matrix: {}", matrix);

    let matrix = Matrix3d(Matrix3::identity());
    let mat: Mat = matrix.into();
    println!("mat: {}", MatPrinter(mat));
}

#[derive(Debug, Clone, Default)]
pub struct Rotation3d(pub Rotation3<f64>);
impl Display for Rotation3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Mat> for Rotation3d {
    fn from(mat: Mat) -> Self {
        let rows = mat.rows();
        let cols = mat.cols();
        assert!(rows == 3 && cols == 3);

        let mut matrix = Matrix3::<f64>::zeros();
        for i in 0..3 {
            for j in 0..3 {
                matrix[(i, j)] = *mat.at_2d::<f64>(i as i32, j as i32).unwrap();
            }
        }
        Rotation3d(Rotation3::from_matrix_unchecked(matrix))
    }
}

impl Into<Mat> for Rotation3d {
    fn into(self) -> Mat {
        let mut mat = Mat::zeros_nd(&[3, 3], CV_64F).unwrap().to_mat().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                *mat.at_2d_mut::<f64>(i as i32, j as i32).unwrap() = self.0[(i, j)];
            }
        }
        mat
    }
}

#[test]
fn test_mat_to_rotation() {
    let mat = Mat::from_slice_2d(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]).unwrap();
    let rotation = Rotation3d::from(mat);
    println!("rotation: {}", rotation);

    let rotation = Rotation3d(Rotation3::identity());
    let mat: Mat = rotation.into();
    println!("mat: {}", MatPrinter(mat));
}

#[derive(Debug, Clone, Default)]
pub struct Vector3d(pub Vector3<f64>);
impl Display for Vector3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Mat> for Vector3d {
    fn from(mat: Mat) -> Self {
        let rows = mat.rows();
        let cols = mat.cols();
        assert!(rows == 3 && cols == 1);
        let mut vector = Vector3::<f64>::zeros();
        for i in 0..3 {
            vector[i] = *mat.at_2d::<f64>(i as i32, 0).unwrap();
        }
        Vector3d(vector)
    }
}

impl Into<Mat> for Vector3d {
    fn into(self) -> Mat {
        let mut mat = Mat::zeros_nd(&[3, 1], CV_64F).unwrap().to_mat().unwrap();
        for i in 0..3 {
            *mat.at_2d_mut::<f64>(i as i32, 0).unwrap() = self.0[i];
        }
        mat
    }
}

#[test]
fn test_mat_to_vector() {
    let mat = Mat::from_slice_2d(&[[1.0], [0.0], [0.0]]).unwrap();
    let vector = Vector3d::from(mat);
    println!("vector: {}", vector);

    let vector = Vector3d(Vector3::new(1.0, 0.0, 0.0));
    let mat: Mat = vector.into();
    println!("mat: {}", MatPrinter(mat));
}
