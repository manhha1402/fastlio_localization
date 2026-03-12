# FAST_LIO_Relocation

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">简体中文</a>
</p>

![License](https://img.shields.io/badge/license-GPL--3.0-blue)

# FAST-LIO 固定先验地图定位扩展版本

# 项目简介
FAST_LIO_Relocation 是在 FAST-LIO 基础上实现的固定地图定位系统。
系统加载先验 PCD 地图，并在该地图上运行 FAST-LIO **原生** scan-to-map + EKF 估计流程，同时增加定位鲁棒性控制机制。

具体的开发流程可以看issue，当此项目stars达到80星时，将NDT加入的版本开源，并且更新ROS2版本，如果你喜欢这个项目的话，请您点点star~

# 核心原理
1 从在线建图切换为固定地图定位
系统提供两种模式：
MODE_MAPPING  
MODE_LOCALIZATION

在定位模式下：

- 加载 PCD 先验地图
- 进行 voxel 下采样
- 构建 ikd-tree
- 禁止地图在线增删

系统从增量建图系统转变为固定地图定位系统。

2 复用 FAST-LIO 原生估计框架
系统**复用** FAST-LIO 原有流程：
- IMU 传播
- 点云去畸变
- 最近邻搜索
- 局部平面拟合
- 点到平面残差
- iterated EKF 更新

3 匹配质量评估
通过以下指标判断定位质量：
- effct_feat_num
- res_mean_last
- accept_lidar_update

匹配质量分为：
acceptable match  
good match

4 四状态定位状态机
UNLOCKED → TRACKING → LOCKED → LOST

1. UNLOCKED  
初始化阶段。
2. TRACKING  
定位可用但不稳定。
3. LOCKED  
定位稳定可靠。
4. LOST  
定位失效，系统冻结。

鲁棒定位机制
第一层：更新后验门控
当系统处于 LOCKED 状态时，如果 EKF 更新导致位姿跳变超过阈值，则拒绝该更新。
第二层：可靠位姿回退

当系统进入 LOST 状态：
- 回退到 last_locked_state
- 输出位姿冻结

# 系统流程

LiDAR + IMU  
↓  
IMU传播 + 点云去畸变  
↓  
点云下采样  
↓  
邻点搜索  
↓  
平面拟合  
↓  
残差构建  
↓  
iterated EKF 更新  
↓  
匹配质量评估  
↓  
状态机  
↓  
门控 / 回退 / 冻结

# 主要功能
## 建图模式

- 原始 FAST-LIO 建图

## 定位模式
- 先验地图加载
- voxel 下采样
- 固定地图定位

定位鲁棒性
- 匹配质量评估
- 状态机
- 位姿门控
- 位姿回退
- RViz 状态可视化

# 依赖
Ubuntu 20.04  
ROS Noetic  
PCL  
Eigen
jsk_rviz_plugins

# 安装
git clone https://github.com/JoCatW/FAST_LIO_Relocation.git

# 编译
catkin_make

# 商业使用
本项目仅限用于学术研究或非商业行为。
商业使用需要联系作者授权。

# 联系
jocatww@gmail.com

# License
GPL-3.0
