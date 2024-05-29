//! 数据集处理
//!
//! EuRoC Dataset  https://paperswithcode.com/dataset/euroc-mav
mod euroc;

pub type DefaultDataset = euroc::EuRoCDataset;
pub trait DatasetTrait {
    /// 读取图像列表
    /// 返回时间戳和图像路径
    fn read_t_cam0_list(&self) -> &Vec<(i64, String)>;
}
