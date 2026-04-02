// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include "IMU_Processing.hpp"
#include "preprocess.h"
#include <Eigen/Core>
#include <algorithm>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <deque>
#include <fastlio_localization/ikd-Tree/ikd_Tree.h>
#include <fastlio_localization/ros2_time.hpp>
#include <fastlio_localization/so3_math.h>
#include <fstream>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <math.h>
#include <mutex>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <omp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <string>
#include <tf2_ros/transform_broadcaster.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vm_navigation_msgs/msg/localization_status.hpp>
static rclcpp::Node::SharedPtr g_node;
static std::shared_ptr<tf2_ros::TransformBroadcaster> g_tf_broadcaster;
static rclcpp::Publisher<vm_navigation_msgs::msg::LocalizationStatus>::SharedPtr
    g_pub_localization_status;

#define LOG_ERROR(...) RCLCPP_ERROR(g_node->get_logger(), __VA_ARGS__)
#define LOG_WARN(...) RCLCPP_WARN(g_node->get_logger(), __VA_ARGS__)
#define LOG_WARN_THROTTLE(ms, ...)                                             \
  RCLCPP_WARN_THROTTLE(g_node->get_logger(), *g_node->get_clock(),             \
                       static_cast<int64_t>(ms) * 1000000LL, __VA_ARGS__)
#define LOG_INFO(...) RCLCPP_INFO(g_node->get_logger(), __VA_ARGS__)
#define LOG_INFO_STREAM(...)                                                   \
  RCLCPP_INFO_STREAM(g_node->get_logger(), __VA_ARGS__)
#define LOG_ERROR_STREAM(...)                                                  \
  RCLCPP_ERROR_STREAM(g_node->get_logger(), __VA_ARGS__)
#define LOG_WARN_STREAM(...)                                                   \
  RCLCPP_WARN_STREAM(g_node->get_logger(), __VA_ARGS__)

#define INIT_TIME (0.1)
#define LASER_POINT_COV (0.001)
#define MAXN (720000)
#define PUBFRAME_PERIOD (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0,
       kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN],
    s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN],
    s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0,
    kdtree_delete_counter = 0;
bool runtime_pos_log = false, pcd_save_en = false, time_sync_en = false,
     extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

std::mutex mtx_buffer;
std::condition_variable sig_buffer;

std::mutex mtx_initial_pose;
geometry_msgs::msg::PoseWithCovarianceStamped pending_initial_pose_msg;
bool pending_initial_pose = false;

std::string root_dir = ROOT_DIR;
std::string map_file_path, lid_topic, imu_topic;
const std::string map_frame_id = "3dmap";
const std::string camera_init_frame_id = "camera_init";
/* ================= FAST_LIO_Relocation modification ================= */
/* Run mode:
   MODE_MAPPING        : original FAST-LIO mapping
   MODE_LOCALIZATION   : localize against an existing PCD map */
enum RunMode { MODE_MAPPING = 0, MODE_LOCALIZATION = 1 };

/* Current run mode */
int run_mode = MODE_MAPPING;

/* True after prior map has been loaded successfully */
bool prior_map_loaded = false;

/* True after ikdtree has been built from the prior map */
bool prior_map_tree_built = false;

/* Loaded prior map point clouds */
PointCloudXYZI::Ptr prior_map_raw(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_map_ds(new PointCloudXYZI());

/* Voxel size for downsampling the prior map */
double map_voxel_size = 0.5;

/* ================= FAST_LIO_Relocation robustness modification
 * ================= */
/* Localization tracking state enum */
enum LocalizationTrackingState {
  TRACKING_UNLOCKED = 0,
  TRACKING_TRACKING = 1,
  TRACKING_LOCKED = 2,
  TRACKING_LOST = 3
};

/* Current localization tracking state */
int tracking_state = TRACKING_UNLOCKED;

int acceptable_match_streak = 0;
int good_match_streak = 0;
int bad_match_streak = 0;

double min_time_before_lock_sec = 2.0;

bool has_last_locked_state = false;
state_ikfom last_locked_state;

bool has_last_tracking_state = false;
state_ikfom last_tracking_state;

/* Whether the current scan's EKF update was accepted */
bool accept_lidar_update = true;

/* Current-frame match quality (good / not good) */
bool current_match_good = false;

/* Match-quality thresholds */
int min_effective_points_for_good = 30;
double max_residual_for_good = 0.20;
int min_effective_points_for_tracking = 15;
double max_residual_for_tracking = 0.40;
/* Streak thresholds (consecutive good/bad frames) */
int good_match_streak_to_lock = 5;
int bad_match_streak_to_lost = 5;

/* Post-update gating threshold */
/* Max position jump after update (only applied in LOCKED state) */
double max_position_jump_for_update_locked = 1.0; // meter

/* Prior-map publishing (rate / one-shot) */
bool prior_map_published_once = false;
int prior_map_pub_counter = 0;
int prior_map_pub_interval = 50;

/* (Optional) fixed RViz overlay for localization status */
// ros::Publisher pubLocalizationOverlayText;
/* ================= FAST_LIO_Relocation robustness modification
 * ================= */

/* ================= FAST_LIO_Relocation modification ================= */
double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0,
       filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0,
       lidar_end_time = 0, first_lidar_time = 0.0;
int effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0,
    laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool point_selected_surf[100000] = {0};
bool lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

std::vector<std::vector<int>> pointSearchInd_surf;
std::vector<BoxPointType> cub_needrm;
std::vector<PointVector> Nearest_Points;
std::vector<double> extrinT(3, 0.0);
std::vector<double> extrinR(9, 0.0);
std::deque<double> time_buffer;
std::deque<PointCloudXYZI::Ptr> lidar_buffer;
std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu;

void SigHandle(int sig) {
  flg_exit = true;
  RCLCPP_WARN(rclcpp::get_logger("laserMapping"), "catch sig %d", sig);
  sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp) {
  V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
  fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
  fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2)); // Angle
  fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1),
          state_point.pos(2));                // Pos
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0); // omega
  fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1),
          state_point.vel(2));                // Vel
  fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0); // Acc
  fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1),
          state_point.bg(2)); // Bias_g
  fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1),
          state_point.ba(2)); // Bias_a
  fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1],
          state_point.grav[2]); // Bias_a
  fprintf(fp, "\r\n");
  fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po,
                            state_ikfom &s) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body +
                                  state_point.offset_T_L_I) +
               state_point.pos);

  po->x = p_global(0);
  po->y = p_global(1);
  po->z = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar +
                 state_point.offset_T_L_I);

  po->x = p_body_imu(0);
  po->y = p_body_imu(1);
  po->z = p_body_imu(2);
  po->intensity = pi->intensity;
}

void points_cache_collect() {
  PointVector points_history;
  ikdtree.acquire_removed_points(points_history);
  // for (int i = 0; i < points_history.size(); i++)
  // _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment() {
  cub_needrm.clear();
  kdtree_delete_counter = 0;
  kdtree_delete_time = 0.0;
  pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
  V3D pos_LiD = pos_lid;
  if (!Localmap_Initialized) {
    for (int i = 0; i < 3; i++) {
      LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
      LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
    }
    Localmap_Initialized = true;
    return;
  }
  float dist_to_map_edge[3][2];
  bool need_move = false;
  for (int i = 0; i < 3; i++) {
    dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
    dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
        dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
      need_move = true;
  }
  if (!need_move)
    return;
  BoxPointType New_LocalMap_Points, tmp_boxpoints;
  New_LocalMap_Points = LocalMap_Points;
  float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                       double(DET_RANGE * (MOV_THRESHOLD - 1)));
  for (int i = 0; i < 3; i++) {
    tmp_boxpoints = LocalMap_Points;
    if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] -= mov_dist;
      New_LocalMap_Points.vertex_min[i] -= mov_dist;
      tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
      New_LocalMap_Points.vertex_max[i] += mov_dist;
      New_LocalMap_Points.vertex_min[i] += mov_dist;
      tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
      cub_needrm.push_back(tmp_boxpoints);
    }
  }
  LocalMap_Points = New_LocalMap_Points;

  points_cache_collect();
  double delete_begin = omp_get_wtime();
  if (cub_needrm.size() > 0)
    kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
  kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  mtx_buffer.lock();
  scan_count++;
  double preprocess_start_time = omp_get_wtime();
  const double ts = fastlio_ros2::stamp_to_sec(msg->header.stamp);
  if (ts < last_timestamp_lidar) {
    LOG_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(ts);
  last_timestamp_lidar = ts;
  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool timediff_set_flg = false;
void livox_pcl_cbk(
    const livox_ros_driver2::msg::CustomMsg::ConstSharedPtr &msg) {
  mtx_buffer.lock();
  double preprocess_start_time = omp_get_wtime();
  scan_count++;
  const double ts = fastlio_ros2::stamp_to_sec(msg->header.stamp);
  if (ts < last_timestamp_lidar) {
    LOG_ERROR("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = ts;

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 &&
      !imu_buffer.empty() && !lidar_buffer.empty()) {
    printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",
           last_timestamp_imu, last_timestamp_lidar);
  }

  if (time_sync_en && !timediff_set_flg &&
      abs(last_timestamp_lidar - last_timestamp_imu) > 1 &&
      !imu_buffer.empty()) {
    timediff_set_flg = true;
    timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
    printf("Self sync IMU and LiDAR, time diff is %.10lf \n",
           timediff_lidar_wrt_imu);
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);

  s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr &msg_in) {
  publish_count++;
  auto msg = std::make_shared<sensor_msgs::msg::Imu>(*msg_in);
  const double t_in = fastlio_ros2::stamp_to_sec(msg_in->header.stamp);
  fastlio_ros2::set_stamp_from_sec(msg->header.stamp,
                                   t_in - time_diff_lidar_to_imu);
  if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en) {
    fastlio_ros2::set_stamp_from_sec(msg->header.stamp,
                                     timediff_lidar_wrt_imu + t_in);
  }

  const double timestamp = fastlio_ros2::stamp_to_sec(msg->header.stamp);

  mtx_buffer.lock();

  if (timestamp < last_timestamp_imu) {
    LOG_WARN("imu loop back, clear buffer");
    imu_buffer.clear();
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
  mtx_buffer.unlock();
  sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int scan_num = 0;
bool sync_packages(MeasureGroup &meas) {
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed) {
    meas.lidar = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();

    if (meas.lidar->points.size() <= 1) // time too little
    {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
      LOG_WARN("Too few input point cloud!\n");
    } else if (meas.lidar->points.back().curvature / double(1000) <
               0.5 * lidar_mean_scantime) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    } else {
      scan_num++;
      lidar_end_time = meas.lidar_beg_time +
                       meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime +=
          (meas.lidar->points.back().curvature / double(1000) -
           lidar_mean_scantime) /
          scan_num;
    }
    //if (lidar_type == MARSIM)
    //  lidar_end_time = meas.lidar_beg_time;

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time) {
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time =
      fastlio_ros2::stamp_to_sec(imu_buffer.front()->header.stamp);
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
    imu_time = fastlio_ros2::stamp_to_sec(imu_buffer.front()->header.stamp);
    if (imu_time > lidar_end_time)
      break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed = false;
  return true;
}

inline bool allow_map_update() {
  /* In localization the map is fixed (no add/remove); only mapping updates the
   * map. */
  return run_mode == MODE_MAPPING;
}

inline bool is_mapping_mode() { return run_mode == MODE_MAPPING; }

inline bool is_localization_mode() { return run_mode == MODE_LOCALIZATION; }

std::string get_tracking_state_name() {
  if (tracking_state == TRACKING_UNLOCKED)
    return "UNLOCKED";
  if (tracking_state == TRACKING_TRACKING)
    return "TRACKING";
  if (tracking_state == TRACKING_LOCKED)
    return "LOCKED";
  if (tracking_state == TRACKING_LOST)
    return "LOST";
  return "UNKNOWN";
}

void save_last_tracking_state() {
  last_tracking_state = state_point;
  has_last_tracking_state = true;
}

void save_last_locked_state() {
  last_locked_state = state_point;
  has_last_locked_state = true;
}

/** When LOST: freeze pose at last acceptable match (tracked), else last locked
 *  good pose, else keep current EKF state. */
void apply_lost_hold_pose() {
  if (has_last_tracking_state) {
    kf.change_x(last_tracking_state);
    state_point = last_tracking_state;
  } else if (has_last_locked_state) {
    kf.change_x(last_locked_state);
    state_point = last_locked_state;
  } else {
    state_point = kf.get_x();
  }
}

void refresh_pose_cache_from_state();

/** RViz2 "2D Pose Estimate" /initialpose: body pose in map (same as /Odometry). */
static void initial_pose_callback(
    const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
  if (!is_localization_mode()) {
    LOG_WARN_THROTTLE(5000,
                      "/initialpose ignored: not in localization mode");
    return;
  }
  /* LocalizationStatus publishes UNLOCKED as "LOST" (only TRACKING→DEGRADED,
   * LOCKED→GOOD). User must be able to set pose while UNLOCKED (startup /
   * never locked). Reject only when already LOCKED. */
  // if (tracking_state == TRACKING_LOCKED) {
  //   LOG_WARN_THROTTLE(
  //       3000,
  //       "/initialpose ignored: localization is LOCKED (GOOD); use when "
  //       "UNLOCKED, DEGRADED, or LOST");
  //   return;
  // }
  if (!msg->header.frame_id.empty() && msg->header.frame_id != map_frame_id) {
    LOG_WARN(
        "/initialpose frame_id is '%s' (expected 'map'); applying anyway",
        msg->header.frame_id.c_str()); 
  }
  std::lock_guard<std::mutex> lk(mtx_initial_pose);
  pending_initial_pose_msg = *msg;
  pending_initial_pose = true;
  LOG_WARN(
      "[FAST_LIO_Relocation] Queued /initialpose for relocalization on next "
      "cycle");
}

static void try_apply_pending_initial_pose() {
  geometry_msgs::msg::PoseWithCovarianceStamped msg_copy;
  bool have = false;
  {
    std::lock_guard<std::mutex> lk(mtx_initial_pose);
    if (!pending_initial_pose) {
      return;
    }
    msg_copy = pending_initial_pose_msg;
    pending_initial_pose = false;
    have = true;
  }
  if (!have || !is_localization_mode()) {
    return;
  }
  // if (tracking_state == TRACKING_LOCKED) {
  //   LOG_WARN_THROTTLE(2000,
  //                     "Discarded queued /initialpose: state is LOCKED (GOOD)");
  //   return;
  // }

  const auto &p = msg_copy.pose.pose.position;
  const auto &o = msg_copy.pose.pose.orientation;
  Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
  if (q.squaredNorm() < 1e-18) {
    LOG_WARN("Ignored /initialpose: invalid quaternion");
    return;
  }
  q.normalize();

  state_point = kf.get_x();
  const double z_prev = state_point.pos(2);
  state_point.pos << p.x, p.y, p.z;
  /* RViz "2D Pose Estimate" usually sends z = 0. Keeping prior z keeps the scan
   * on the same vertical slice as the prior map for point-to-plane matching. */
  if (std::abs(p.z) < 1e-3) {
    state_point.pos(2) = z_prev;
  }
  state_point.rot = SO3(q.w(), q.x(), q.y(), q.z());
  state_point.vel << 0.0, 0.0, 0.0;
  kf.change_x(state_point);

  /* change_x() does not touch P. After a long LOCKED run P is tiny, Kalman gain
   * suppresses lidar corrections — scan stays misaligned despite a good click.
   * Inflate pose block so iterated updates can pull pose toward the map. */
  {
    auto P = kf.get_P();
    constexpr double infl_pos = 25.0; // ~5 m std on x,y,z
    constexpr double infl_rot = 0.35; // ~0.59 rad std on roll/pitch/yaw axes
    for (int d = 0; d < 3; ++d) {
      P(d, d) += infl_pos;
      P(d + 3, d + 3) += infl_rot;
    }
    kf.change_P(P);
  }

  tracking_state = TRACKING_UNLOCKED;
  acceptable_match_streak = 0;
  good_match_streak = 0;
  bad_match_streak = 0;
  save_last_tracking_state();
  save_last_locked_state();

  refresh_pose_cache_from_state();
  LOG_WARN("[FAST_LIO_Relocation] Applied /initialpose: EKF pose reset, "
           "tracking → UNLOCKED for relocalization");
}

bool is_match_acceptable() {
  /* "Acceptable": match can sustain TRACKING but may not be strong enough for
   * LOCKED. */

  if (!accept_lidar_update) {
    return false;
  }

  if (effct_feat_num < min_effective_points_for_tracking) {
    return false;
  }

  if (res_mean_last > max_residual_for_tracking) {
    return false;
  }

  return true;
}

bool is_match_good() {
  /* "Good": high-quality match; can enter or hold LOCKED. */

  if (!accept_lidar_update) {
    return false;
  }

  if (effct_feat_num < min_effective_points_for_good) {
    return false;
  }

  if (res_mean_last > max_residual_for_good) {
    return false;
  }

  return true;
}
void update_tracking_state_machine() {
  /* Four-state machine: UNLOCKED -> TRACKING -> LOCKED -> TRACKING -> LOST.
     In LOST we do not auto-recover; wait for external relocalization. */

  bool match_acceptable = is_match_acceptable();
  bool match_good = is_match_good();

  current_match_good = match_good;

  if (match_acceptable) {
    acceptable_match_streak++;
    bad_match_streak = 0;
  } else {
    acceptable_match_streak = 0;
    bad_match_streak++;
  }

  if (match_good) {
    good_match_streak++;
  } else {
    good_match_streak = 0;
  }

  double time_since_start = Measures.lidar_beg_time - first_lidar_time;

  if (tracking_state == TRACKING_UNLOCKED) {
    if (match_acceptable && acceptable_match_streak >= 3) {
      tracking_state = TRACKING_TRACKING;
      save_last_tracking_state();
      LOG_INFO("[FAST_LIO_Relocation] Tracking state switched to TRACKING.");
    }
    return;
  }

  if (tracking_state == TRACKING_TRACKING) {
    if (match_acceptable) {
      save_last_tracking_state();
    }

    if (time_since_start >= min_time_before_lock_sec && match_good &&
        good_match_streak >= good_match_streak_to_lock) {
      tracking_state = TRACKING_LOCKED;
      save_last_tracking_state();
      save_last_locked_state();
      LOG_INFO("[FAST_LIO_Relocation] Tracking state switched to LOCKED.");
      return;
    }

    if (!match_acceptable && bad_match_streak >= bad_match_streak_to_lost) {
      tracking_state = TRACKING_LOST;
      LOG_WARN("[FAST_LIO_Relocation] Tracking state switched to LOST.");
      return;
    }

    return;
  }
  if (tracking_state == TRACKING_LOCKED) {
    if (match_acceptable) {
      save_last_tracking_state();
    }

    if (match_good) {
      save_last_locked_state();
    }

    if (!match_good) {
      tracking_state = TRACKING_TRACKING;
      good_match_streak = 0;
      LOG_WARN("[FAST_LIO_Relocation] Tracking state downgraded to TRACKING.");
      return;
    }

    return;
  }

  if (tracking_state == TRACKING_LOST) {
    /* LOST: stay frozen until external relocalization; no auto transition out.
     */
    return;
  }
}

/** Publish vm_navigation_msgs/LocalizationStatus from FAST-LIO tracking state.
 *  TRACKING_LOCKED → GOOD, TRACKING_TRACKING → DEGRADED, else → LOST. */
static void publish_localization_status() {
  if (!g_pub_localization_status || !g_node) {
    return;
  }
  vm_navigation_msgs::msg::LocalizationStatus msg;
  msg.header.stamp = g_node->now();
  msg.header.frame_id = map_frame_id;
  if (tracking_state == TRACKING_LOCKED) {
    msg.status = "GOOD";
  } else if (tracking_state == TRACKING_TRACKING) {
    msg.status = "DEGRADED";
  } else {
    msg.status = "LOST";
  }
  msg.inlier_rmse = res_mean_last;
  msg.fitness_score =
      current_match_good
          ? std::max(0.0, 1.0 - std::min(1.0, res_mean_last))
          : 0.0;
  msg.pose_distance_error = 0.0;
  g_pub_localization_status->publish(msg);
}

bool check_pose_update_reasonable(const state_ikfom &state_before,
                                  const state_ikfom &state_after) {
  /* Position gating only in LOCKED. UNLOCKED / TRACKING / LOST allow larger
   * corrections to converge. */

  if (tracking_state != TRACKING_LOCKED) {
    return true;
  }

  double delta_pos = (state_after.pos - state_before.pos).norm();

  if (delta_pos > max_position_jump_for_update_locked) {
    LOG_WARN_STREAM(
        "[FAST_LIO_Relocation] Reject update: delta_pos too large = "
        << delta_pos
        << ", threshold = " << max_position_jump_for_update_locked);
    return false;
  }

  return true;
}

static std::string map_file_extension_lower(const std::string &path) {
  const auto dot = path.find_last_of('.');
  if (dot == std::string::npos || dot + 1 >= path.size()) {
    return {};
  }
  std::string ext = path.substr(dot + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return ext;
}

static void
pointcloud_xyz_to_pointtype(const pcl::PointCloud<pcl::PointXYZ> &src,
                            PointCloudXYZI::Ptr dst) {
  dst->resize(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    PointType &d = dst->points[i];
    d.x = src.points[i].x;
    d.y = src.points[i].y;
    d.z = src.points[i].z;
    d.intensity = 0.f;
    d.normal_x = 0.f;
    d.normal_y = 0.f;
    d.normal_z = 0.f;
    d.curvature = 0.f;
  }
  dst->width = src.width;
  dst->height = src.height;
  dst->is_dense = src.is_dense;
}

bool load_prior_map_from_pcd(const std::string &pcd_path,
                             PointCloudXYZI::Ptr cloud_raw,
                             PointCloudXYZI::Ptr cloud_ds) {
  /* Load prior map from disk, voxel-downsample (localization mode only). */

  const std::string pkg_path =
      ament_index_cpp::get_package_share_directory("fastlio_localization");
  const std::string full_path = pkg_path + "/map/" + pcd_path;
  const std::string ext = map_file_extension_lower(full_path);
  bool loaded = false;

  /*
   * Mesh-style PLY (e.g. VCGLIB: x,y,z + nx,ny,nz + rgba, no intensity) must
   * not be read as PointXYZINormal — property order/names do not match and xyz
   * end up corrupted (RViz: no cloud).
   */
  if (ext == "ply") {
    pcl::PointCloud<pcl::PointXYZ> ply_xyz;
    if (pcl::io::loadPLYFile(full_path, ply_xyz) >= 0 && !ply_xyz.empty()) {
      pointcloud_xyz_to_pointtype(ply_xyz, cloud_raw);
      loaded = true;
    }
    if (!loaded) {
      LOG_WARN("[FAST_LIO_Relocation] PLY load as xyz failed or empty; trying "
               "PointXYZINormal.");
      if (pcl::io::loadPLYFile<PointType>(full_path, *cloud_raw) < 0) {
        LOG_ERROR_STREAM(
            "[FAST_LIO_Relocation] Failed to load map: " << full_path);
        return false;
      }
      loaded = true;
    }
  } else {
    if (pcl::io::loadPCDFile<PointType>(full_path, *cloud_raw) < 0) {
      LOG_ERROR_STREAM(
          "[FAST_LIO_Relocation] Failed to load map: " << full_path);
      return false;
    }
  }

  if (cloud_raw->empty()) {
    LOG_ERROR("[FAST_LIO_Relocation] Loaded map is empty.");
    return false;
  }

  {
    std::vector<int> nan_idx;
    pcl::removeNaNFromPointCloud(*cloud_raw, *cloud_raw, nan_idx);
  }
  if (cloud_raw->empty()) {
    LOG_ERROR(
        "[FAST_LIO_Relocation] Map contains only NaN points after sanitizing.");
    return false;
  }

  {
    float min_x = cloud_raw->points[0].x, max_x = min_x;
    float min_y = cloud_raw->points[0].y, max_y = min_y;
    float min_z = cloud_raw->points[0].z, max_z = min_z;
    for (const auto &p : cloud_raw->points) {
      min_x = std::min(min_x, p.x);
      max_x = std::max(max_x, p.x);
      min_y = std::min(min_y, p.y);
      max_y = std::max(max_y, p.y);
      min_z = std::min(min_z, p.z);
      max_z = std::max(max_z, p.z);
    }
    LOG_INFO_STREAM("[FAST_LIO_Relocation] Prior map AABB (raw) x["
                    << min_x << ", " << max_x << "] y[" << min_y << ", "
                    << max_y << "] z[" << min_z << ", " << max_z << "]");
  }

  pcl::VoxelGrid<PointType> prior_filter;
  prior_filter.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);
  prior_filter.setInputCloud(cloud_raw);
  prior_filter.filter(*cloud_ds);

  if (cloud_ds->empty()) {
    LOG_ERROR("[FAST_LIO_Relocation] Downsampled map is empty.");
    return false;
  }

  LOG_INFO_STREAM("[FAST_LIO_Relocation] Prior map loaded. Raw size = "
                  << cloud_raw->size()
                  << ", downsampled size = " << cloud_ds->size());

  return true;
}

bool init_localization_map_from_prior() {
  /* Localization: build ikdtree from PCD map once. */

  if (prior_map_tree_built) {
    LOG_WARN("[FAST_LIO_Relocation] Prior map already initialized.");
    return true;
  }

  if (!load_prior_map_from_pcd(map_file_path, prior_map_raw, prior_map_ds)) {
    return false;
  }

  ikdtree.set_downsample_param(map_voxel_size);
  ikdtree.Build(prior_map_ds->points);

  prior_map_loaded = true;
  prior_map_tree_built = true;

  LOG_INFO("[FAST_LIO_Relocation] ikdtree initialized from prior map.");

  return true;
}

bool init_map_from_first_scan() {
  /* Mapping: original FAST-LIO — initialize map from the first scan. */

  if (feats_down_size <= 5) {
    LOG_WARN("Too few points to initialize map.");
    return false;
  }

  ikdtree.set_downsample_param(filter_size_map_min);

  feats_down_world->resize(feats_down_size);

  for (int i = 0; i < feats_down_size; i++) {
    pointBodyToWorld(&(feats_down_body->points[i]),
                     &(feats_down_world->points[i]));
  }

  ikdtree.Build(feats_down_world->points);

  LOG_INFO_STREAM(
      "[FAST_LIO_Relocation] Map initialized from first scan. Size = "
      << feats_down_world->size());

  return true;
}

int process_increments = 0;
void map_incremental() {
  PointVector PointToAdd;
  PointVector PointNoNeedDownsample;
  PointToAdd.reserve(feats_down_size);
  PointNoNeedDownsample.reserve(feats_down_size);
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]),
                     &(feats_down_world->points[i]));
    /* decide if need add to map */
    if (!Nearest_Points[i].empty() && flg_EKF_inited) {
      const PointVector &points_near = Nearest_Points[i];
      bool need_add = true;
      PointType downsample_result, mid_point;
      mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) *
                        filter_size_map_min +
                    0.5 * filter_size_map_min;
      float dist = calc_dist(feats_down_world->points[i], mid_point);
      if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
          fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
        PointNoNeedDownsample.push_back(feats_down_world->points[i]);
        continue;
      }
      for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
        if (points_near.size() < NUM_MATCH_POINTS)
          break;
        if (calc_dist(points_near[readd_i], mid_point) < dist) {
          need_add = false;
          break;
        }
      }
      if (need_add)
        PointToAdd.push_back(feats_down_world->points[i]);
    } else {
      PointToAdd.push_back(feats_down_world->points[i]);
    }
  }

  double st_time = omp_get_wtime();
  add_point_size = ikdtree.Add_Points(PointToAdd, true);
  ikdtree.Add_Points(PointNoNeedDownsample, false);
  add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
  kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

/** Use lidar time when available; otherwise node clock so TF and PointCloud2
 * stamps stay consistent for RViz. */
static void set_publish_stamp(builtin_interfaces::msg::Time &t) {
  if (lidar_end_time > 0.0) {
    fastlio_ros2::set_stamp_from_sec(t, lidar_end_time);
  } else if (g_node) {
    const rclcpp::Time now = g_node->get_clock()->now();
    const int64_t ns = now.nanoseconds();
    t.sec = static_cast<int32_t>(ns / 1000000000LL);
    t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  } else {
    fastlio_ros2::set_stamp_from_sec(t, 0.0);
  }
}

void publish_frame_world(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudFull) {
  if (scan_pub_en) {
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort
                                                       : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                          &laserCloudWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    set_publish_stamp(laserCloudmsg.header.stamp);
    /* World-frame cloud: frame_id is camera_init (mapping) or map
     * (localization). */
    std::string world_frame = is_localization_mode() ? map_frame_id : camera_init_frame_id;
    laserCloudmsg.header.frame_id = world_frame;
    pubLaserCloudFull->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
  }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. noted that pcd save will influence the real-time performences **/
  if (pcd_save_en) {
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
      RGBpointBodyToWorld(&feats_undistort->points[i],
                          &laserCloudWorld->points[i]);
    }
    *pcl_wait_save += *laserCloudWorld;

    static int scan_wait_num = 0;
    scan_wait_num++;
    if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 &&
        scan_wait_num >= pcd_save_interval) {
      pcd_index++;
      std::string out_path(std::string(std::string(ROOT_DIR) + "PLY/scans_") +
                           std::to_string(pcd_index) + std::string(".ply"));
      if (pcl::io::savePLYFileBinary(out_path, *pcl_wait_save) != 0) {
        LOG_ERROR("Failed to write PLY: %s", out_path.c_str());
      } else {
        std::cout << "current scan saved to " << out_path << std::endl;
      }
      pcl_wait_save->clear();
      scan_wait_num = 0;
    }
  }
}

void publish_frame_body(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudFull_body) {
  int size = feats_undistort->points.size();
  PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

  for (int i = 0; i < size; i++) {
    RGBpointBodyLidarToIMU(&feats_undistort->points[i],
                           &laserCloudIMUBody->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
  set_publish_stamp(laserCloudmsg.header.stamp);
  laserCloudmsg.header.frame_id = "body";
  pubLaserCloudFull_body->publish(laserCloudmsg);
  publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudEffect) {
  PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(effct_feat_num, 1));
  for (int i = 0; i < effct_feat_num; i++) {
    RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
  }

  sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  set_publish_stamp(laserCloudFullRes3.header.stamp);

  std::string world_frame = is_localization_mode() ? map_frame_id : camera_init_frame_id;
  laserCloudFullRes3.header.frame_id = world_frame;

  pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudMap) {
  sensor_msgs::msg::PointCloud2 laserCloudMap;
  pcl::toROSMsg(*featsFromMap, laserCloudMap);
  set_publish_stamp(laserCloudMap.header.stamp);

  std::string world_frame = is_localization_mode() ? map_frame_id : camera_init_frame_id;
  laserCloudMap.header.frame_id = world_frame;

  pubLaserCloudMap->publish(laserCloudMap);
}

void publish_prior_map(
    const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr
        &pubLaserCloudMap) {
  if (!prior_map_loaded || prior_map_ds->empty()) {
    return;
  }

  /* RViz2: xyz+intensity; voxel-downsample again for lighter /Laser_map traffic
   * (ikdtree keeps full prior_map_ds). */
  pcl::PointCloud<pcl::PointXYZI>::Ptr viz(new pcl::PointCloud<pcl::PointXYZI>);
  viz->resize(prior_map_ds->size());
  for (size_t i = 0; i < prior_map_ds->size(); ++i) {
    viz->points[i].x = prior_map_ds->points[i].x;
    viz->points[i].y = prior_map_ds->points[i].y;
    viz->points[i].z = prior_map_ds->points[i].z;
    viz->points[i].intensity = prior_map_ds->points[i].intensity;
  }
  viz->width = static_cast<uint32_t>(viz->size());
  viz->height = 1;
  viz->is_dense = false;

  const float pub_leaf =
      static_cast<float>(std::max(map_voxel_size * 2.0, 0.2));
  pcl::VoxelGrid<pcl::PointXYZI> vg_pub;
  vg_pub.setLeafSize(pub_leaf, pub_leaf, pub_leaf);
  vg_pub.setInputCloud(viz);
  pcl::PointCloud<pcl::PointXYZI> viz_pub;
  vg_pub.filter(viz_pub);

  sensor_msgs::msg::PointCloud2 laserCloudMap;
  pcl::toROSMsg(viz_pub, laserCloudMap);

  set_publish_stamp(laserCloudMap.header.stamp);

  /* Same parent frame as odometry / registered cloud in localization (map). */
  laserCloudMap.header.frame_id =
      is_localization_mode() ? map_frame_id : camera_init_frame_id;
  pubLaserCloudMap->publish(laserCloudMap);
}

// void publish_localization_status_overlay(const ros::Publisher & pubOverlay)
// {
//     /* OverlayText: show localization status at a fixed screen position in
//     RViz. */

//     if (!is_localization_mode())
//     {
//         return;
//     }

//     jsk_rviz_plugins::OverlayText text;
//     text.action = jsk_rviz_plugins::OverlayText::ADD;

//     text.width = 420;
//     text.height = 240;

//     /* Text box position in RViz (smaller left/top -> closer to top-left). */
//     text.left = 20;
//     text.top = 20;

//     text.text_size = 16;
//     text.line_width = 2;
//     text.font = "DejaVu Sans Mono";

//     /* Background color */
//     text.bg_color.r = 0.0;
//     text.bg_color.g = 0.0;
//     text.bg_color.b = 0.0;
//     text.bg_color.a = 0.6;

//     /* Foreground color by state */
//     if (tracking_state == TRACKING_LOCKED)
//     {
//         text.fg_color.r = 0.0;
//         text.fg_color.g = 1.0;
//         text.fg_color.b = 0.0;
//         text.fg_color.a = 1.0;
//     }
//     else if (tracking_state == TRACKING_TRACKING)
//     {
//         text.fg_color.r = 0.0;
//         text.fg_color.g = 0.8;
//         text.fg_color.b = 1.0;
//         text.fg_color.a = 1.0;
//     }
//     else if (tracking_state == TRACKING_UNLOCKED)
//     {
//         text.fg_color.r = 1.0;
//         text.fg_color.g = 1.0;
//         text.fg_color.b = 0.0;
//         text.fg_color.a = 1.0;
//     }
//     else
//     {
//         text.fg_color.r = 1.0;
//         text.fg_color.g = 0.0;
//         text.fg_color.b = 0.0;
//         text.fg_color.a = 1.0;
//     }

//     std::ostringstream oss;
//     oss << "FAST_LIO_Relocation\n";
//     oss << "Mode: LOCALIZATION\n";
//     oss << "State: " << get_tracking_state_name() << "\n";
//     oss << "Effective points: " << effct_feat_num << "\n";
//     oss << "Mean residual: " << std::fixed << std::setprecision(3) <<
//     res_mean_last << "\n"; oss << "Good streak: " << good_match_streak <<
//     "\n"; oss << "Bad streak: " << bad_match_streak << "\n"; oss <<
//     "Acceptable streak: " << acceptable_match_streak << "\n"; oss << "Update
//     accepted: " << (accept_lidar_update ? "YES" : "NO")<< "\n"; oss << "Has
//     last tracking: " << (has_last_tracking_state ? "YES" : "NO") << "\n"; oss
//     << "Has last locked: " << (has_last_locked_state ? "YES" : "NO") << "\n";

//     text.text = oss.str();

//     pubOverlay.publish(text);
// }

void refresh_pose_cache_from_state() {
  euler_cur = SO3ToEuler(state_point.rot);
  pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

  geoQuat.x = state_point.rot.coeffs()[0];
  geoQuat.y = state_point.rot.coeffs()[1];
  geoQuat.z = state_point.rot.coeffs()[2];
  geoQuat.w = state_point.rot.coeffs()[3];
}

template <typename T> void set_posestamp(T &out) {
  out.pose.position.x = state_point.pos(0);
  out.pose.position.y = state_point.pos(1);
  out.pose.position.z = state_point.pos(2);
  out.pose.orientation.x = geoQuat.x;
  out.pose.orientation.y = geoQuat.y;
  out.pose.orientation.z = geoQuat.z;
  out.pose.orientation.w = geoQuat.w;
}

void publish_odometry(
    const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr
        &pubOdomAftMapped) {
  /* Odometry parent frame: camera_init (mapping) or map (localization, pose in
   * prior map). */
  std::string world_frame = is_localization_mode() ? map_frame_id : camera_init_frame_id;

  odomAftMapped.header.frame_id = world_frame;
  odomAftMapped.child_frame_id = "body";
  set_publish_stamp(odomAftMapped.header.stamp);

  set_posestamp(odomAftMapped.pose);

  auto P = kf.get_P();
  for (int i = 0; i < 6; i++) {
    int k = i < 3 ? i + 3 : i - 3;
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 0)] = P(k, 3);
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 1)] = P(k, 4);
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 2)] = P(k, 5);
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 3)] = P(k, 0);
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 4)] = P(k, 1);
    odomAftMapped.pose.covariance[static_cast<size_t>(i * 6 + 5)] = P(k, 2);
  }

  pubOdomAftMapped->publish(odomAftMapped);

  if (g_tf_broadcaster) {
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = odomAftMapped.header.stamp;
    tf_msg.header.frame_id = world_frame;
    tf_msg.child_frame_id = "body";
    tf_msg.transform.translation.x = odomAftMapped.pose.pose.position.x;
    tf_msg.transform.translation.y = odomAftMapped.pose.pose.position.y;
    tf_msg.transform.translation.z = odomAftMapped.pose.pose.position.z;
    tf_msg.transform.rotation = odomAftMapped.pose.pose.orientation;
    g_tf_broadcaster->sendTransform(tf_msg);
  }
}

void publish_path(
    const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath) {
  /* Path uses the same parent frame as odometry: camera_init or map. */
  std::string world_frame = is_localization_mode() ? map_frame_id : camera_init_frame_id;

  set_posestamp(msg_body_pose);
  set_publish_stamp(msg_body_pose.header.stamp);
  msg_body_pose.header.frame_id = world_frame;
  path.header.frame_id = world_frame;

  /*** if path is too large, the rvis will crash ***/
  static int jjj = 0;
  jjj++;
  if (jjj % 10 == 0) {
    path.poses.push_back(msg_body_pose);
    pubPath->publish(path);
  }
}

void h_share_model(state_ikfom &s,
                   esekfom::dyn_share_datastruct<double> &ekfom_data) {
  double match_start = omp_get_wtime();
  laserCloudOri->clear();
  corr_normvect->clear();
  total_residual = 0.0;

/** closest surface search and residual computation **/
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < feats_down_size; i++) {
    PointType &point_body = feats_down_body->points[i];
    PointType &point_world = feats_down_world->points[i];

    /* transform to world frame */
    V3D p_body(point_body.x, point_body.y, point_body.z);
    V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);
    point_world.x = p_global(0);
    point_world.y = p_global(1);
    point_world.z = p_global(2);
    point_world.intensity = point_body.intensity;

    std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

    auto &points_near = Nearest_Points[i];

    if (ekfom_data.converge) {
      /** Find the closest surfaces in the map **/
      ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near,
                             pointSearchSqDis);
      point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false
                               : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5
                                   ? false
                                   : true;
    }

    if (!point_selected_surf[i])
      continue;

    VF(4) pabcd;
    point_selected_surf[i] = false;
    if (esti_plane(pabcd, points_near, 0.1f)) {
      float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                  pabcd(2) * point_world.z + pabcd(3);
      float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

      if (s > 0.9) {
        point_selected_surf[i] = true;
        normvec->points[i].x = pabcd(0);
        normvec->points[i].y = pabcd(1);
        normvec->points[i].z = pabcd(2);
        normvec->points[i].intensity = pd2;
        res_last[i] = abs(pd2);
      }
    }
  }

  effct_feat_num = 0;

  for (int i = 0; i < feats_down_size; i++) {
    if (point_selected_surf[i]) {
      laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
      corr_normvect->points[effct_feat_num] = normvec->points[i];
      total_residual += res_last[i];
      effct_feat_num++;
    }
  }

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    LOG_WARN_THROTTLE(1000, "[FAST_LIO_Relocation] No Effective Points!");
    return;
  }

  res_mean_last = total_residual / effct_feat_num;
  match_time += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); // 23
  ekfom_data.h.resize(effct_feat_num);

  for (int i = 0; i < effct_feat_num; i++) {
    const PointType &laser_p = laserCloudOri->points[i];
    V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
    M3D point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    const PointType &norm_p = corr_normvect->points[i];
    V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() *
            C); // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z,
          VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -norm_p.intensity;
  }
  solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  g_node = std::make_shared<rclcpp::Node>("laserMapping");
  g_tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(g_node);

  p_imu.reset(new ImuProcess());
  bool gravity_align_en = true;

  auto node = g_node;

  path_en = node->declare_parameter<bool>("publish.path_en", true);
  scan_pub_en = node->declare_parameter<bool>("publish.scan_publish_en", true);
  dense_pub_en =
      node->declare_parameter<bool>("publish.dense_publish_en", true);
  scan_body_pub_en =
      node->declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
  NUM_MAX_ITERATIONS = node->declare_parameter<int>("max_iteration", 4);
  map_file_path = node->declare_parameter<std::string>("map_file_path", "");
  lid_topic =
      node->declare_parameter<std::string>("common.lid_topic", "/livox/lidar");
  imu_topic =
      node->declare_parameter<std::string>("common.imu_topic", "/livox/imu");
  time_sync_en = node->declare_parameter<bool>("common.time_sync_en", false);
  time_diff_lidar_to_imu =
      node->declare_parameter<double>("common.time_offset_lidar_to_imu", 0.0);
  filter_size_corner_min =
      node->declare_parameter<double>("filter_size_corner", 0.5);
  filter_size_surf_min =
      node->declare_parameter<double>("filter_size_surf", 0.5);
  filter_size_map_min = node->declare_parameter<double>("filter_size_map", 0.5);
  cube_len = node->declare_parameter<double>("cube_side_length", 200.0);
  gravity_align_en =
      node->declare_parameter<bool>("mapping.gravity_align_en", true);
  DET_RANGE = node->declare_parameter<float>("mapping.det_range", 300.f);
  fov_deg = node->declare_parameter<double>("mapping.fov_degree", 180.0);
  gyr_cov = node->declare_parameter<double>("mapping.gyr_cov", 0.1);
  acc_cov = node->declare_parameter<double>("mapping.acc_cov", 0.1);
  b_gyr_cov = node->declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
  b_acc_cov = node->declare_parameter<double>("mapping.b_acc_cov", 0.0001);
  p_pre->blind = node->declare_parameter<double>("preprocess.blind", 0.01);
  lidar_type = node->declare_parameter<int>("preprocess.lidar_type", LIVOX);
  const bool lidar_qos_sensor_data =
      node->declare_parameter<bool>("preprocess.lidar_qos_sensor_data", false);
  p_pre->N_SCANS = node->declare_parameter<int>("preprocess.scan_line", 16);
  p_pre->time_unit =
      node->declare_parameter<int>("preprocess.timestamp_unit", US);
  p_pre->SCAN_RATE = node->declare_parameter<int>("preprocess.scan_rate", 10);
  p_pre->point_filter_num = node->declare_parameter<int>("point_filter_num", 2);
  p_pre->feature_enabled =
      node->declare_parameter<bool>("feature_extract_enable", false);
  runtime_pos_log =
      node->declare_parameter<bool>("runtime_pos_log_enable", false);
  extrinsic_est_en =
      node->declare_parameter<bool>("mapping.extrinsic_est_en", true);
  pcd_save_en = node->declare_parameter<bool>("pcd_save.pcd_save_en", false);
  pcd_save_interval = node->declare_parameter<int>("pcd_save.interval", -1);
  extrinT = node->declare_parameter<std::vector<double>>("mapping.extrinsic_T",
                                                         std::vector<double>());
  extrinR = node->declare_parameter<std::vector<double>>("mapping.extrinsic_R",
                                                         std::vector<double>());

  /* ================= FAST_LIO_Relocation modification ================= */
  run_mode = node->declare_parameter<int>("run_mode", 0);
  map_voxel_size = node->declare_parameter<double>("map_voxel_size", 0.5);
  /* ================= FAST_LIO_Relocation modification ================= */

  /* ================= FAST_LIO_Relocation robustness modification
   * ================= */
  min_effective_points_for_tracking = node->declare_parameter<int>(
      "localization.min_effective_points_for_tracking", 15);
  max_residual_for_tracking = node->declare_parameter<double>(
      "localization.max_residual_for_tracking", 0.40);
  min_time_before_lock_sec = node->declare_parameter<double>(
      "localization.min_time_before_lock_sec", 2.0);
  min_effective_points_for_good = node->declare_parameter<int>(
      "localization.min_effective_points_for_good", 30);
  max_residual_for_good = node->declare_parameter<double>(
      "localization.max_residual_for_good", 0.20);
  good_match_streak_to_lock =
      node->declare_parameter<int>("localization.good_match_streak_to_lock", 5);
  bad_match_streak_to_lost =
      node->declare_parameter<int>("localization.bad_match_streak_to_lost", 5);
  max_position_jump_for_update_locked = node->declare_parameter<double>(
      "localization.max_position_jump_for_update_locked", 1.0);
  prior_map_pub_interval =
      node->declare_parameter<int>("localization.prior_map_pub_interval", 50);
  /* ================= FAST_LIO_Relocation robustness modification
   * ================= */

  /* ================= FAST_LIO_Relocation modification ================= */
  /* Validate run_mode */

  if (run_mode != MODE_MAPPING && run_mode != MODE_LOCALIZATION) {
    LOG_WARN(
        "[FAST_LIO_Relocation] Invalid run_mode, fallback to MODE_MAPPING.");
    run_mode = MODE_MAPPING;
  }

  if (is_localization_mode()) {
    LOG_INFO("[FAST_LIO_Relocation] Running in LOCALIZATION mode.");

    tracking_state = TRACKING_UNLOCKED;
    acceptable_match_streak = 0;
    good_match_streak = 0;
    bad_match_streak = 0;
    accept_lidar_update = true;
    prior_map_published_once = false;
    prior_map_pub_counter = 0;
  } else {
    LOG_INFO("[FAST_LIO_Relocation] Running in MAPPING mode.");
  }
  /* ================= FAST_LIO_Relocation modification ================= */

  p_imu->set_gravity_align_enable(gravity_align_en);
  p_pre->lidar_type = lidar_type;
  std::cout << "p_pre->lidar_type " << p_pre->lidar_type << std::endl;

  path.header.stamp = g_node->get_clock()->now();
  path.header.frame_id = is_localization_mode() ? map_frame_id : camera_init_frame_id;

  /*** variables definition ***/
  int frame_num = 0;
  double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0,
         aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

  FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
  HALF_FOV_COS = cos((FOV_DEG)*0.5 * PI_M / 180.0);

  _featsArray.reset(new PointCloudXYZI());

  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));
  downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min,
                                 filter_size_surf_min);
  downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min,
                                filter_size_map_min);
  memset(point_selected_surf, true, sizeof(point_selected_surf));
  memset(res_last, -1000.0f, sizeof(res_last));

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
  p_imu->lidar_type = lidar_type;
  double epsi[23] = {0.001};
  fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS,
                    epsi);

  /*** debug record ***/
  FILE *fp;
  std::string pos_log_dir = root_dir + "/Log/pos_log.txt";
  fp = fopen(pos_log_dir.c_str(), "w");

  std::ofstream fout_pre, fout_out, fout_dbg;
  fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
  fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
  fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"), std::ios::out);
  if (fout_pre && fout_out)
    std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
  else
    std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

  /*** ROS 2 subscribers / publishers ***/
  const rclcpp::QoS qos_lidar(rclcpp::KeepLast(2000));
  const rclcpp::QoS qos_imu(rclcpp::KeepLast(2000));
  const rclcpp::QoS qos_pub(rclcpp::KeepLast(2000));
  /* Prior map is published at startup; RViz must subscribe with Transient Local
     durability (see rviz/livox*.rviz) or volatile subscribers miss the first
     message and stay empty. */
  const rclcpp::QoS qos_map =
      rclcpp::QoS(rclcpp::KeepLast(1)).transient_local().reliable();

  rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_livox;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl_std;
  if (p_pre->lidar_type == LIVOX) {
    sub_livox = node->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        lid_topic, qos_lidar, livox_pcl_cbk);
  } else {
    /* SIM / many simulators publish PointCloud2 with BEST_EFFORT; a RELIABLE
     * subscriber never matches and receives no data (DDS incompatibility). */
    rclcpp::QoS qos_pc(rclcpp::KeepLast(2000));
    qos_pc.durability_volatile();
    if (lidar_type == SIM || lidar_qos_sensor_data) {
      qos_pc.best_effort();
    } else {
      qos_pc.reliable();
    }
    sub_pcl_std = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        lid_topic, qos_pc, standard_pcl_cbk);
  }
  auto sub_imu = node->create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, qos_imu, imu_cbk);

  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
      sub_initial_pose;
  if (is_localization_mode()) {
    sub_initial_pose =
        node->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 50,
            [](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
              initial_pose_callback(msg);
            });
  }

  auto pubLaserCloudFull =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered",
                                                            qos_pub);
  auto pubLaserCloudFull_body =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "/cloud_registered_body", qos_pub);
  auto pubLaserCloudEffect =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected",
                                                            qos_pub);
  auto pubLaserCloudMap = node->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/Laser_map", qos_map);
  auto pubOdomAftMapped =
      node->create_publisher<nav_msgs::msg::Odometry>("/Odometry", qos_pub);
  auto pubPath = node->create_publisher<nav_msgs::msg::Path>("/path", qos_pub);
  if (is_localization_mode()) {
    g_pub_localization_status =
        node->create_publisher<vm_navigation_msgs::msg::LocalizationStatus>(
            "/localization_status", rclcpp::QoS(10).reliable());
  }
  // pubLocalizationOverlayText = nh.advertise<jsk_rviz_plugins::OverlayText>
  //     ("/localization_status_overlay", 10);
  /* ================= FAST_LIO_Relocation modification ================= */
  /* Localization: load prior map before main loop */

  if (is_localization_mode()) {
    if (!init_localization_map_from_prior()) {
      LOG_ERROR("[FAST_LIO_Relocation] Failed to initialize prior map.");
      rclcpp::shutdown();
      return -1;
    }

    /* Register frame map_frame_id in tf2 and align stamps before first /Laser_map
     * (RViz needs map->body). */
    state_point = kf.get_x();
    refresh_pose_cache_from_state();
    publish_odometry(pubOdomAftMapped);

    publish_prior_map(pubLaserCloudMap);
    prior_map_published_once = true;
    LOG_INFO("[FAST_LIO_Relocation] Prior map published as latched topic "
             "/Laser_map.");
  }
  /* ================= FAST_LIO_Relocation modification ================= */
  //------------------------------------------------------------------------------------------------------
  signal(SIGINT, SigHandle);
  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(g_node);
  rclcpp::WallRate rate(5000.0);
  bool status = rclcpp::ok();
  (void)sub_imu;
  (void)sub_livox;
  (void)sub_pcl_std;
  (void)sub_initial_pose;
  while (status) {
    if (flg_exit)
      break;
    exec.spin_some();
    try_apply_pending_initial_pose();
    if (sync_packages(Measures)) {
      if (flg_first_scan) {
        first_lidar_time = Measures.lidar_beg_time;
        p_imu->first_lidar_time = first_lidar_time;
        flg_first_scan = false;
        continue;
      }

      double t0, t1, t3, t5;

      match_time = 0;
      kdtree_search_time = 0.0;
      solve_time = 0;
      solve_const_H_time = 0;
      t0 = omp_get_wtime();

      if (is_localization_mode() && tracking_state == TRACKING_LOST) {
        /* Undistort current scan, then publish in map using last tracked pose
         * (last_tracking_state), else last_locked_state, matching odometry. */
        p_imu->Process(Measures, kf, feats_undistort);
        state_point = kf.get_x();
        pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
        if (feats_undistort->empty() || (feats_undistort == NULL)) {
          LOG_WARN("No point, skip this scan!\n");
          continue;
        }
        downSizeFilterSurf.setInputCloud(feats_undistort);
        downSizeFilterSurf.filter(*feats_down_body);

        apply_lost_hold_pose();
        refresh_pose_cache_from_state();

        publish_odometry(pubOdomAftMapped);
        if (path_en)
          publish_path(pubPath);
        if (scan_pub_en || pcd_save_en)
          publish_frame_world(pubLaserCloudFull);
        if (scan_pub_en && scan_body_pub_en)
          publish_frame_body(pubLaserCloudFull_body);
        publish_localization_status();

        status = rclcpp::ok();
        rate.sleep();
        continue;
      }

      p_imu->Process(Measures, kf, feats_undistort);
      state_point = kf.get_x();
      pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

      if (feats_undistort->empty() || (feats_undistort == NULL)) {
        LOG_WARN("No point, skip this scan!\n");
        continue;
      }

      flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME
                           ? false
                           : true;
      /*** Segment the map in lidar FOV ***/
      /* ================= FAST_LIO_Relocation modification ================= */
      /* Localization: do not remove map points in FOV segment */

      if (is_mapping_mode()) {
        lasermap_fov_segment();
      }
      /* ================= FAST_LIO_Relocation modification ================= */

      /*** downsample the feature points in a scan ***/
      downSizeFilterSurf.setInputCloud(feats_undistort);
      downSizeFilterSurf.filter(*feats_down_body);
      t1 = omp_get_wtime();
      feats_down_size = feats_down_body->points.size();

      /*** initialize the map kdtree ***/
      /* ================= FAST_LIO_Relocation modification ================= */
      /* ikdtree initialization */

      if (ikdtree.Root_Node == nullptr) {
        if (is_localization_mode()) {
          LOG_ERROR("[FAST_LIO_Relocation] ikdtree should already be "
                    "initialized from prior map.");
        } else {
          init_map_from_first_scan();
        }
        continue;
      }
      /* ================= FAST_LIO_Relocation modification ================= */
      kdtree_size_st = ikdtree.size();



      /*** ICP and iterated Kalman filter update ***/
      if (feats_down_size < 5) {
        LOG_WARN("No point, skip this scan!\n");
        continue;
      }

      normvec->resize(feats_down_size);
      feats_down_world->resize(feats_down_size);

      V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
      fout_pre << std::setw(20) << Measures.lidar_beg_time - first_lidar_time
               << " " << euler_cur.transpose() << " "
               << state_point.pos.transpose() << " " << ext_euler.transpose()
               << " " << state_point.offset_T_L_I.transpose() << " "
               << state_point.vel.transpose() << " "
               << state_point.bg.transpose() << " "
               << state_point.ba.transpose() << " " << state_point.grav
               << std::endl;

      if (0) // If you need to see map point, change to "if(1)"
      {
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        featsFromMap->clear();
        featsFromMap->points = ikdtree.PCL_Storage;
      }

      pointSearchInd_surf.resize(feats_down_size);
      Nearest_Points.resize(feats_down_size);

      /*** iterated state estimation ***/
      double t_update_start = omp_get_wtime();
      double solve_H_time = 0;

      /* Save pre-update state for post-update gating */
      state_ikfom state_before_update = kf.get_x();
      auto P_before_update = kf.get_P();
      kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

      state_ikfom state_after_update = kf.get_x();

      /* Reject unreasonable pose jump; roll back to pre-update if needed */
      if (is_localization_mode() &&
          !check_pose_update_reasonable(state_before_update,
                                        state_after_update)) {
        kf.change_x(state_before_update);
        kf.change_P(P_before_update);
        state_point = state_before_update;
        accept_lidar_update = false;
      } else {
        state_point = state_after_update;
        accept_lidar_update = true;
      }

      refresh_pose_cache_from_state();
      /* Localization: advance tracking state machine */
      if (is_localization_mode()) {
        update_tracking_state_machine();

        if (tracking_state == TRACKING_LOST) {
          LOG_WARN_THROTTLE(1000, "[FAST_LIO_Relocation] Tracking LOST.");

          apply_lost_hold_pose();
          refresh_pose_cache_from_state();

          publish_odometry(pubOdomAftMapped);
          if (path_en)
            publish_path(pubPath);
          if (scan_pub_en || pcd_save_en)
            publish_frame_world(pubLaserCloudFull);
          if (scan_pub_en && scan_body_pub_en)
            publish_frame_body(pubLaserCloudFull_body);
          publish_localization_status();

          status = rclcpp::ok();
          rate.sleep();
          continue;
        }
      }

      double t_update_end = omp_get_wtime();

      /******* Publish odometry *******/
      publish_odometry(pubOdomAftMapped);

      /*** add the feature points to map kdtree ***/
      t3 = omp_get_wtime();

      /* ================= FAST_LIO_Relocation modification ================= */
      /* Localization: no new map points */

      if (allow_map_update()) {
        map_incremental();
      } else {
        kdtree_incremental_time = 0.0;
        add_point_size = 0;
      }
      /* ================= FAST_LIO_Relocation modification ================= */

      t5 = omp_get_wtime();

      /******* Publish points *******/
      if (path_en)
        publish_path(pubPath);
      if (scan_pub_en || pcd_save_en)
        publish_frame_world(pubLaserCloudFull);
      if (scan_pub_en && scan_body_pub_en)
        publish_frame_body(pubLaserCloudFull_body);
      if (is_localization_mode()) {
        publish_localization_status();
      }
      // publish_effect_world(pubLaserCloudEffect);

      /* Localization: publish prior map once, then at low rate */
      if (is_localization_mode()) {
        prior_map_pub_counter++;

        if (!prior_map_published_once ||
            prior_map_pub_counter >= prior_map_pub_interval) {
          publish_prior_map(pubLaserCloudMap);
          prior_map_published_once = true;
          prior_map_pub_counter = 0;
        }
      } else {
        // publish_map(pubLaserCloudMap);
      }

      /*** Debug variables ***/
      if (runtime_pos_log) {
        frame_num++;
        kdtree_size_end = ikdtree.size();
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num +
                          (t5 - t0) / frame_num;
        aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num +
                        (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num +
                          (match_time) / frame_num;
        aver_time_incre = aver_time_incre * (frame_num - 1) / frame_num +
                          (kdtree_incremental_time) / frame_num;
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num +
                          (solve_time + solve_H_time) / frame_num;
        aver_time_const_H_time =
            aver_time_const_H_time * (frame_num - 1) / frame_num +
            solve_time / frame_num;
        T1[time_log_counter] = Measures.lidar_beg_time;
        s_plot[time_log_counter] = t5 - t0;
        s_plot2[time_log_counter] = feats_undistort->points.size();
        s_plot3[time_log_counter] = kdtree_incremental_time;
        s_plot4[time_log_counter] = kdtree_search_time;
        s_plot5[time_log_counter] = kdtree_delete_counter;
        s_plot6[time_log_counter] = kdtree_delete_time;
        s_plot7[time_log_counter] = kdtree_size_st;
        s_plot8[time_log_counter] = kdtree_size_end;
        s_plot9[time_log_counter] = aver_time_consu;
        s_plot10[time_log_counter] = add_point_size;
        time_log_counter++;
        const char *mode_name =
            is_localization_mode() ? "localization" : "mapping";
        printf("[ %s ]: time: IMU + Map + Input Downsample: %0.6f ave match: "
               "%0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave "
               "total: %0.6f icp: %0.6f construct H: %0.6f \n",
               mode_name, t1 - t0, aver_time_match, aver_time_solve, t3 - t1,
               t5 - t3, aver_time_consu, aver_time_icp, aver_time_const_H_time);
        ext_euler = SO3ToEuler(state_point.offset_R_L_I);
        fout_out << std::setw(20) << Measures.lidar_beg_time - first_lidar_time
                 << " " << euler_cur.transpose() << " "
                 << state_point.pos.transpose() << " " << ext_euler.transpose()
                 << " " << state_point.offset_T_L_I.transpose() << " "
                 << state_point.vel.transpose() << " "
                 << state_point.bg.transpose() << " "
                 << state_point.ba.transpose() << " " << state_point.grav << " "
                 << feats_undistort->points.size() << std::endl;
        dump_lio_state_to_log(fp);
      }
    }

    status = rclcpp::ok();
    rate.sleep();
  }

  /**************** save map ****************/
  /* 1. make sure you have enough memories
  /* 2. pcd save will largely influence the real-time performences **/
  if (pcl_wait_save->size() > 0 && pcd_save_en) {
    const std::string file_name = std::string("scans.ply");
    const std::string out_path(std::string(std::string(ROOT_DIR) + "PLY/") +
                               file_name);
    if (pcl::io::savePLYFileBinary(out_path, *pcl_wait_save) != 0) {
      LOG_ERROR("Failed to write PLY: %s", out_path.c_str());
    } else {
      std::cout << "current scan saved to " << out_path << std::endl;
    }
  }

  fout_out.close();
  fout_pre.close();

  if (runtime_pos_log) {
    std::vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6,
        s_vec7;
    FILE *fp2;
    std::string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
    fp2 = fopen(log_dir.c_str(), "w");
    fprintf(fp2, "time_stamp, total time, scan point size, incremental time, "
                 "search time, delete size, delete time, tree size st, tree "
                 "size end, add point size, preprocess time\n");
    for (int i = 0; i < time_log_counter; i++) {
      fprintf(fp2, "%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",
              T1[i], s_plot[i], int(s_plot2[i]), s_plot3[i], s_plot4[i],
              int(s_plot5[i]), s_plot6[i], int(s_plot7[i]), int(s_plot8[i]),
              int(s_plot10[i]), s_plot11[i]);
      t.push_back(T1[i]);
      s_vec.push_back(s_plot9[i]);
      s_vec2.push_back(s_plot3[i] + s_plot6[i]);
      s_vec3.push_back(s_plot4[i]);
      s_vec5.push_back(s_plot[i]);
    }
    fclose(fp2);
  }

  rclcpp::shutdown();
  return 0;
}
