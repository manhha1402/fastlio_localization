# FAST_LIO_Relocation

<p align="center">
  <a href="README.md">English</a> |
  <a href="README_CN.md">简体中文</a>
</p>

![License](https://img.shields.io/badge/license-GPL--3.0-blue)

A robust fixed-map localization extension for FAST-LIO
FAST_LIO_Relocation extends FAST-LIO from online mapping to prior-map-based localization.

# Overview
FAST_LIO_Relocation transforms FAST-LIO into a fixed-map LiDAR localization system.
Instead of continuously growing the map, the system loads a prior PCD map and performs localization against it while preserving FAST-LIO's original scan-to-map estimation pipeline.

For details on the development process, please refer to the issues. When this project reaches 80 stars, the version incorporating NDT will be open-sourced, and the ROS2 version will be updated. If you like this project, please give it a star!

Core Principle
1 Fixed-map localization instead of incremental mapping

Two modes are supported:
MODE_MAPPING  
MODE_LOCALIZATION

In localization mode:

- a prior PCD map is loaded
- the map is voxel-downsampled
- the ikd-tree is built once
- map insertion and deletion are disabled

2 Reuse FAST-LIO scan-to-map + EKF backbone

FAST_LIO_Relocation **reuses** the original FAST-LIO estimation pipeline:

- IMU propagation
- LiDAR motion undistortion
- nearest neighbor search
- plane fitting
- point-to-plane residual
- iterated EKF update

3 Match-quality-driven localization

Localization quality is evaluated using:

- effective feature count (effct_feat_num)
- mean residual (res_mean_last)
- update acceptance (accept_lidar_update)

Two match levels:

acceptable match  
good match

4 Localization state machine

UNLOCKED → TRACKING → LOCKED → LOST

1. UNLOCKED  
Initialization phase.
2. TRACKING  
Localization running but not stable.
3. LOCKED  
Localization reliable.
4. LOST  
Localization failure detected. System freezes at last reliable pose.

Robust Localization Mechanism
Layer 1: Pose gating
Large pose jumps after EKF update are rejected when localization is LOCKED.
Layer 2: Last reliable pose rollback

When LOST:
- system rolls back to last_locked_state
- pose output freezes

# System Pipeline

LiDAR + IMU  
↓  
IMU propagation + undistortion  
↓  
Downsample scan  
↓  
Nearest neighbor search  
↓  
Plane fitting  
↓  
Residual construction  
↓  
Iterated EKF update  
↓  
Match quality evaluation  
↓  
State machine  
↓  
Pose gating / rollback



# Features
## Mapping Mode
- original FAST-LIO mapping
## Localization Mode
- prior PCD map loading
- voxel filtering
- fixed-map localization

Robust Relocation

- match quality evaluation
- state machine
- pose gating
- pose rollback
- RViz overlay

Dependencies

Ubuntu 20.04  
ROS Noetic  
PCL  
Eigen
jsk_rviz_plugins

## Installation

git clone https://github.com/JoCatW/FAST_LIO_Relocation.git

## Build

catkin_make



# Commercial Use
Academic research only.
Commercial use requires permission.
Contact
jocatww@gmail.com

# License
GPL-3.0
