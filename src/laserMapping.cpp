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
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include <jsk_rviz_plugins/OverlayText.h>
#include <sstream>
#include <iomanip>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

/* ================= FAST_LIO_Relocation modification ================= */
/* 运行模式定义
   MODE_MAPPING        : 原始 FAST-LIO 建图模式
   MODE_LOCALIZATION   : 使用已有 PCD 地图进行定位 */
enum RunMode
{
    MODE_MAPPING = 0,
    MODE_LOCALIZATION = 1
};

/* 当前运行模式 */
int run_mode = MODE_MAPPING;

/* 是否已经成功加载先验地图 */
bool prior_map_loaded = false;

/* 是否已经使用先验地图初始化 ikdtree */
bool prior_map_tree_built = false;

/* 存储读取的 PCD 地图 */
PointCloudXYZI::Ptr prior_map_raw(new PointCloudXYZI());
PointCloudXYZI::Ptr prior_map_ds(new PointCloudXYZI());

/* PCD 地图体素降采样大小 */
double map_voxel_size = 0.5;

/* ================= FAST_LIO_Relocation robustness modification ================= */
/* 定位跟踪状态定义 */
enum LocalizationTrackingState
{
    TRACKING_UNLOCKED = 0,
    TRACKING_TRACKING = 1,
    TRACKING_LOCKED   = 2,
    TRACKING_LOST     = 3
};

/* 当前定位跟踪状态 */
int tracking_state = TRACKING_UNLOCKED;

int acceptable_match_streak = 0;
int good_match_streak = 0;
int bad_match_streak = 0;

double min_time_before_lock_sec = 2.0;

bool has_last_locked_state = false;
state_ikfom last_locked_state;

bool has_last_tracking_state = false;
state_ikfom last_tracking_state;


/* 当前帧激光更新是否被接受 */
bool accept_lidar_update = true;

/* 当前帧匹配质量 */
bool current_match_good = false;

/* 匹配质量门限 */
int min_effective_points_for_good = 30;
double max_residual_for_good = 0.20;
int min_effective_points_for_tracking = 15;
double max_residual_for_tracking = 0.40;
/* 连续计数门限 */
int good_match_streak_to_lock = 5;
int bad_match_streak_to_lost = 5;

/* 更新后验门控阈值 */
/* 更新后验位置门控阈值（仅在 LOCKED 状态下启用） */
double max_position_jump_for_update_locked = 1.0;   // meter

/* 先验地图发布控制 */
bool prior_map_published_once = false;
int prior_map_pub_counter = 0;
int prior_map_pub_interval = 50;

/* RViz 屏幕固定文字状态发布 */
ros::Publisher pubLocalizationOverlayText;
/* ================= FAST_LIO_Relocation robustness modification ================= */


/* ================= FAST_LIO_Relocation modification ================= */
double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

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

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu;

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();
    
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

inline bool allow_map_update()
{
    /* 中文注释：
       在重定位模式下地图是固定的，不允许新增或删除点
       只有 mapping 模式允许更新地图 */
    return run_mode == MODE_MAPPING;
}

inline bool is_mapping_mode()
{
    return run_mode == MODE_MAPPING;
}

inline bool is_localization_mode()
{
    return run_mode == MODE_LOCALIZATION;
}

std::string get_tracking_state_name()
{
    if (tracking_state == TRACKING_UNLOCKED) return "UNLOCKED";
    if (tracking_state == TRACKING_TRACKING) return "TRACKING";
    if (tracking_state == TRACKING_LOCKED)   return "LOCKED";
    if (tracking_state == TRACKING_LOST)     return "LOST";
    return "UNKNOWN";
}

void save_last_tracking_state()
{
    last_tracking_state = state_point;
    has_last_tracking_state = true;
}

void save_last_locked_state()
{
    last_locked_state = state_point;
    has_last_locked_state = true;
}

bool is_match_acceptable()
{
    /* 中文注释：
       acceptable 表示还能维持 TRACKING，
       但不一定足够好到进入 LOCKED */

    if (!accept_lidar_update)
    {
        return false;
    }

    if (effct_feat_num < min_effective_points_for_tracking)
    {
        return false;
    }

    if (res_mean_last > max_residual_for_tracking)
    {
        return false;
    }

    return true;
}

bool is_match_good()
{
    /* 中文注释：
       good 表示质量较高，可用于进入或维持 LOCKED */

    if (!accept_lidar_update)
    {
        return false;
    }

    if (effct_feat_num < min_effective_points_for_good)
    {
        return false;
    }

    if (res_mean_last > max_residual_for_good)
    {
        return false;
    }

    return true;
}
void update_tracking_state_machine()
{
    /* 中文注释：
       四态状态机：
       UNLOCKED -> TRACKING -> LOCKED -> TRACKING -> LOST
       LOST 状态下不自动恢复，等待外部重定位模块 */

    bool match_acceptable = is_match_acceptable();
    bool match_good = is_match_good();

    current_match_good = match_good;

    if (match_acceptable)
    {
        acceptable_match_streak++;
        bad_match_streak = 0;
    }
    else
    {
        acceptable_match_streak = 0;
        bad_match_streak++;
    }

    if (match_good)
    {
        good_match_streak++;
    }
    else
    {
        good_match_streak = 0;
    }

    double time_since_start = Measures.lidar_beg_time - first_lidar_time;

    if (tracking_state == TRACKING_UNLOCKED)
    {
        if (match_acceptable && acceptable_match_streak >= 3)
        {
            tracking_state = TRACKING_TRACKING;
            save_last_tracking_state();
            ROS_INFO("[FAST_LIO_Relocation] Tracking state switched to TRACKING.");
        }
        return;
    }

    if (tracking_state == TRACKING_TRACKING)
    {
        if (match_acceptable)
        {
            save_last_tracking_state();
        }

        if (time_since_start >= min_time_before_lock_sec &&
            match_good &&
            good_match_streak >= good_match_streak_to_lock)
        {
            tracking_state = TRACKING_LOCKED;
            save_last_tracking_state();
            save_last_locked_state();
            ROS_INFO("[FAST_LIO_Relocation] Tracking state switched to LOCKED.");
            return;
        }

        if (!match_acceptable &&
            bad_match_streak >= bad_match_streak_to_lost)
        {
            tracking_state = TRACKING_LOST;
            ROS_WARN("[FAST_LIO_Relocation] Tracking state switched to LOST.");
            return;
        }

        return;
    }
    if (tracking_state == TRACKING_LOCKED)
    {
        if (match_acceptable)
        {
            save_last_tracking_state();
        }

        if (match_good)
        {
            save_last_locked_state();
        }

        if (!match_good)
        {
            tracking_state = TRACKING_TRACKING;
            good_match_streak = 0;
            ROS_WARN("[FAST_LIO_Relocation] Tracking state downgraded to TRACKING.");
            return;
        }

        return;
    }

    if (tracking_state == TRACKING_LOST)
    {
        /* 中文注释：
           LOST 状态下保持冻结，等待外部重定位，不自动回到其他状态 */
        return;
    }
}



bool check_pose_update_reasonable(const state_ikfom &state_before,
                                  const state_ikfom &state_after)
{
    /* 中文注释：
       仅在 LOCKED 状态下做位置门控。
       UNLOCKED / TRACKING / LOST 阶段允许较大修正，以便收敛。 */

    if (tracking_state != TRACKING_LOCKED)
    {
        return true;
    }

    double delta_pos = (state_after.pos - state_before.pos).norm();

    if (delta_pos > max_position_jump_for_update_locked)
    {
        ROS_WARN_STREAM("[FAST_LIO_Relocation] Reject update: delta_pos too large = "
                        << delta_pos << ", threshold = "
                        << max_position_jump_for_update_locked);
        return false;
    }

    return true;
}

bool load_prior_map_from_pcd(const std::string &pcd_path,
                             PointCloudXYZI::Ptr cloud_raw,
                             PointCloudXYZI::Ptr cloud_ds)
{
    /* 中文注释：
       从磁盘读取 PCD 地图
       并进行体素降采样
       该函数只在重定位模式下使用 */

    if (pcd_path.empty())
    {
        ROS_ERROR("[FAST_LIO_Relocation] map_file_path is empty.");
        return false;
    }

    if (pcl::io::loadPCDFile<PointType>(pcd_path, *cloud_raw) < 0)
    {
        ROS_ERROR_STREAM("[FAST_LIO_Relocation] Failed to load map: " << pcd_path);
        return false;
    }

    if (cloud_raw->empty())
    {
        ROS_ERROR("[FAST_LIO_Relocation] Loaded map is empty.");
        return false;
    }

    pcl::VoxelGrid<PointType> prior_filter;
    prior_filter.setLeafSize(map_voxel_size, map_voxel_size, map_voxel_size);
    prior_filter.setInputCloud(cloud_raw);
    prior_filter.filter(*cloud_ds);

    if (cloud_ds->empty())
    {
        ROS_ERROR("[FAST_LIO_Relocation] Downsampled map is empty.");
        return false;
    }

    ROS_INFO_STREAM("[FAST_LIO_Relocation] Prior map loaded. Raw size = "
                    << cloud_raw->size()
                    << ", downsampled size = "
                    << cloud_ds->size());

    return true;
}

bool init_localization_map_from_prior()
{
    /* 中文注释：
       在定位模式下
       从 PCD 地图构建 ikdtree
       该过程只执行一次 */

    if (prior_map_tree_built)
    {
        ROS_WARN("[FAST_LIO_Relocation] Prior map already initialized.");
        return true;
    }

    if (!load_prior_map_from_pcd(map_file_path, prior_map_raw, prior_map_ds))
    {
        return false;
    }

    ikdtree.set_downsample_param(map_voxel_size);
    ikdtree.Build(prior_map_ds->points);

    prior_map_loaded = true;
    prior_map_tree_built = true;

    ROS_INFO("[FAST_LIO_Relocation] ikdtree initialized from prior map.");

    return true;
}

bool init_map_from_first_scan()
{
    /* 中文注释：
       原始 FAST-LIO 的行为
       在 mapping 模式下
       使用第一帧点云初始化地图 */

    if (feats_down_size <= 5)
    {
        ROS_WARN("Too few points to initialize map.");
        return false;
    }

    ikdtree.set_downsample_param(filter_size_map_min);

    feats_down_world->resize(feats_down_size);

    for (int i = 0; i < feats_down_size; i++)
    {
        pointBodyToWorld(&(feats_down_body->points[i]),
                         &(feats_down_world->points[i]));
    }

    ikdtree.Build(feats_down_world->points);

    ROS_INFO_STREAM("[FAST_LIO_Relocation] Map initialized from first scan. Size = "
                    << feats_down_world->size());

    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
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
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        /* 
        世界系下点云发布到对应的全局坐标系，
        mapping 模式为 camera_init，
        relocalization 模式为 map。 */
        std::string world_frame = is_localization_mode() ? "map" : "camera_init";
        laserCloudmsg.header.frame_id = world_frame;
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld(
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i],
                            &laserCloudWorld->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);

    std::string world_frame = is_localization_mode() ? "map" : "camera_init";
    laserCloudFullRes3.header.frame_id = world_frame;

    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);

    std::string world_frame = is_localization_mode() ? "map" : "camera_init";
    laserCloudMap.header.frame_id = world_frame;

    pubLaserCloudMap.publish(laserCloudMap);
}

void publish_prior_map(const ros::Publisher & pubLaserCloudMap)
{
    if (!prior_map_loaded || prior_map_ds->empty())
    {
        return;
    }

    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*prior_map_ds, laserCloudMap);

    laserCloudMap.header.stamp = (lidar_end_time > 0.0) ?
                                 ros::Time().fromSec(lidar_end_time) :
                                 ros::Time::now();

    laserCloudMap.header.frame_id = "map";
    pubLaserCloudMap.publish(laserCloudMap);
}

void publish_localization_status_overlay(const ros::Publisher & pubOverlay)
{
    /* 中文注释：
       使用 OverlayText 在 RViz 屏幕固定位置显示定位状态 */

    if (!is_localization_mode())
    {
        return;
    }

    jsk_rviz_plugins::OverlayText text;
    text.action = jsk_rviz_plugins::OverlayText::ADD;

    text.width = 420;
    text.height = 240;

    /* 中文注释：
       这里控制文字框在 RViz 屏幕中的位置
       left/top 越小越靠左上角 */
    text.left = 20;
    text.top = 20;

    text.text_size = 16;
    text.line_width = 2;
    text.font = "DejaVu Sans Mono";

    /* 背景颜色 */
    text.bg_color.r = 0.0;
    text.bg_color.g = 0.0;
    text.bg_color.b = 0.0;
    text.bg_color.a = 0.6;

    /* 前景颜色按状态变化 */
    if (tracking_state == TRACKING_LOCKED)
    {
        text.fg_color.r = 0.0;
        text.fg_color.g = 1.0;
        text.fg_color.b = 0.0;
        text.fg_color.a = 1.0;
    }
    else if (tracking_state == TRACKING_TRACKING)
    {
        text.fg_color.r = 0.0;
        text.fg_color.g = 0.8;
        text.fg_color.b = 1.0;
        text.fg_color.a = 1.0;
    }
    else if (tracking_state == TRACKING_UNLOCKED)
    {
        text.fg_color.r = 1.0;
        text.fg_color.g = 1.0;
        text.fg_color.b = 0.0;
        text.fg_color.a = 1.0;
    }
    else
    {
        text.fg_color.r = 1.0;
        text.fg_color.g = 0.0;
        text.fg_color.b = 0.0;
        text.fg_color.a = 1.0;
    }

    std::ostringstream oss;
    oss << "FAST_LIO_Relocation\n";
    oss << "Mode: LOCALIZATION\n";
    oss << "State: " << get_tracking_state_name() << "\n";
    oss << "Effective points: " << effct_feat_num << "\n";
    oss << "Mean residual: " << std::fixed << std::setprecision(3) << res_mean_last << "\n";
    oss << "Good streak: " << good_match_streak << "\n";
    oss << "Bad streak: " << bad_match_streak << "\n";
    oss << "Acceptable streak: " << acceptable_match_streak << "\n";
    oss << "Update accepted: " << (accept_lidar_update ? "YES" : "NO")<< "\n";
    oss << "Has last tracking: " << (has_last_tracking_state ? "YES" : "NO") << "\n";
    oss << "Has last locked: " << (has_last_locked_state ? "YES" : "NO") << "\n";

    text.text = oss.str();

    pubOverlay.publish(text);
}

void refresh_pose_cache_from_state()
{
    euler_cur = SO3ToEuler(state_point.rot);
    pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

    geoQuat.x = state_point.rot.coeffs()[0];
    geoQuat.y = state_point.rot.coeffs()[1];
    geoQuat.z = state_point.rot.coeffs()[2];
    geoQuat.w = state_point.rot.coeffs()[3];
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    /* 
       mapping 模式下，父坐标系为 camera_init；
       relocalization 模式下，父坐标系为 map，
       表示当前位姿是在先验 PCD 地图坐标系下估计得到的。 */
    std::string world_frame = is_localization_mode() ? "map" : "camera_init";

    odomAftMapped.header.frame_id = world_frame;
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);

    set_posestamp(odomAftMapped.pose);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    pubOdomAftMapped.publish(odomAftMapped);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x,
                                    odomAftMapped.pose.pose.position.y,
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation(q);

    br.sendTransform(tf::StampedTransform(transform,
                                          odomAftMapped.header.stamp,
                                          world_frame,
                                          "body"));
}

void publish_path(const ros::Publisher pubPath)
{
    /* ：
       path 的父坐标系需要和 odometry 保持一致，
       mapping 模式下为 camera_init，
       relocalization 模式下为 map。 */
    std::string world_frame = is_localization_mode() ? "map" : "camera_init";

    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = world_frame;
    path.header.frame_id = world_frame;

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
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

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN_THROTTLE(1.0, "[FAST_LIO_Relocation] No Effective Points!");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;
    p_imu.reset(new ImuProcess());
    bool gravity_align_en = true;
    
    
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<bool>("mapping/gravity_align_en", gravity_align_en, true);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);
    nh.param<int>("preprocess/lidar_type", lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    /* ================= FAST_LIO_Relocation modification ================= */
    /* 读取运行模式参数 */
    nh.param<int>("run_mode", run_mode, 0);

    /* 读取地图体素大小 */
    nh.param<double>("map_voxel_size", map_voxel_size, 0.5);
    /* ================= FAST_LIO_Relocation modification ================= */

    /* ================= FAST_LIO_Relocation robustness modification ================= */
    nh.param<int>("localization/min_effective_points_for_tracking",
              min_effective_points_for_tracking, 15);
    nh.param<double>("localization/max_residual_for_tracking",
                    max_residual_for_tracking, 0.40);
    nh.param<double>("localization/min_time_before_lock_sec",
                 min_time_before_lock_sec, 2.0);
    nh.param<int>("localization/min_effective_points_for_good", min_effective_points_for_good, 30);
    nh.param<double>("localization/max_residual_for_good", max_residual_for_good, 0.20);
    nh.param<int>("localization/good_match_streak_to_lock", good_match_streak_to_lock, 5);
    nh.param<int>("localization/bad_match_streak_to_lost", bad_match_streak_to_lost, 5);
    nh.param<double>("localization/max_position_jump_for_update_locked",
                 max_position_jump_for_update_locked, 1.0);
    nh.param<int>("localization/prior_map_pub_interval", prior_map_pub_interval, 50);
    /* ================= FAST_LIO_Relocation robustness modification ================= */

    /* ================= FAST_LIO_Relocation modification ================= */
    /* 判断当前运行模式 */

    if (run_mode != MODE_MAPPING && run_mode != MODE_LOCALIZATION)
    {
        ROS_WARN("[FAST_LIO_Relocation] Invalid run_mode, fallback to MODE_MAPPING.");
        run_mode = MODE_MAPPING;
    }

    if (is_localization_mode())
    {
        ROS_INFO("[FAST_LIO_Relocation] Running in LOCALIZATION mode.");

        tracking_state = TRACKING_UNLOCKED;
        acceptable_match_streak = 0;
        good_match_streak = 0;
        bad_match_streak = 0;
        accept_lidar_update = true;
        prior_map_published_once = false;
        prior_map_pub_counter = 0;
    }
    else
    {
        ROS_INFO("[FAST_LIO_Relocation] Running in MAPPING mode.");
    }
    /* ================= FAST_LIO_Relocation modification ================= */

    p_imu->set_gravity_align_enable(gravity_align_en);
    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
        ("/Laser_map", 1, true);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);
    pubLocalizationOverlayText = nh.advertise<jsk_rviz_plugins::OverlayText>
        ("/localization_status_overlay", 10);
    /* ================= FAST_LIO_Relocation modification ================= */
    /* 如果是定位模式，在进入主循环前加载先验地图 */

    if (is_localization_mode())
    {
        if (!init_localization_map_from_prior())
        {
            ROS_ERROR("[FAST_LIO_Relocation] Failed to initialize prior map.");
            return -1;
        }

        publish_prior_map(pubLaserCloudMap);
        prior_map_published_once = true;
        ROS_INFO("[FAST_LIO_Relocation] Prior map published as latched topic /Laser_map.");
    }
    /* ================= FAST_LIO_Relocation modification ================= */
//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            if (is_localization_mode() && tracking_state == TRACKING_LOST)
            {
                if (has_last_locked_state)
                {
                    kf.change_x(last_locked_state);
                    state_point = last_locked_state;
                }
                else
                {
                    state_point = kf.get_x();
                }

                refresh_pose_cache_from_state();

                publish_odometry(pubOdomAftMapped);
                if (path_en) publish_path(pubPath);
                publish_localization_status_overlay(pubLocalizationOverlayText);

                status = ros::ok();
                rate.sleep();
                continue;
            }

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            /* ================= FAST_LIO_Relocation modification ================= */
            /* 在定位模式下不允许删除地图点 */

            if (is_mapping_mode())
            {
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
            /* ikdtree 初始化逻辑 */

            if (ikdtree.Root_Node == nullptr)
            {
                if (is_localization_mode())
                {
                    ROS_ERROR("[FAST_LIO_Relocation] ikdtree should already be initialized from prior map.");
                }
                else
                {
                    init_map_from_first_scan();
                }
                continue;
            }
            /* ================= FAST_LIO_Relocation modification ================= */
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;

            /* 中文注释：
            保存更新前状态，用于更新后验门控 */
            state_ikfom state_before_update = kf.get_x();
            auto P_before_update = kf.get_P();
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);

            state_ikfom state_after_update = kf.get_x();

            /* 中文注释：
            对优化后的位姿跳变量做合理性检查
            若当前帧更新不合理，则回退到更新前状态 */
            if (is_localization_mode() &&
                !check_pose_update_reasonable(state_before_update, state_after_update))
            {
                kf.change_x(state_before_update);
                kf.change_P(P_before_update);
                state_point = state_before_update;
                accept_lidar_update = false;
            }
            else
            {
                state_point = state_after_update;
                accept_lidar_update = true;
            }

            refresh_pose_cache_from_state();
            /* 中文注释：
            localization 模式下更新定位状态机 */
            if (is_localization_mode())
            {
                update_tracking_state_machine();

                if (tracking_state == TRACKING_LOST)
                {
                    ROS_WARN_THROTTLE(1.0, "[FAST_LIO_Relocation] Tracking LOST.");

                    if (has_last_locked_state)
                    {
                        kf.change_x(last_locked_state);
                        state_point = last_locked_state;
                    }

                    refresh_pose_cache_from_state();

                    publish_odometry(pubOdomAftMapped);
                    if (path_en) publish_path(pubPath);
                    publish_localization_status_overlay(pubLocalizationOverlayText);

                    status = ros::ok();
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
            /* 在定位模式下禁止新增地图点 */

            if (allow_map_update())
            {
                map_incremental();
            }
            else
            {
                kdtree_incremental_time = 0.0;
                add_point_size = 0;
            }
            /* ================= FAST_LIO_Relocation modification ================= */

            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (is_localization_mode())
            {
                publish_localization_status_overlay(pubLocalizationOverlayText);
            }
            // publish_effect_world(pubLaserCloudEffect);

            /* 
            定位模式下先验地图首次发布一次，后续低频补发 */
            if (is_localization_mode())
            {
                prior_map_pub_counter++;

                if (!prior_map_published_once || prior_map_pub_counter >= prior_map_pub_interval)
                {
                    publish_prior_map(pubLaserCloudMap);
                    prior_map_published_once = true;
                    prior_map_pub_counter = 0;
                }
            }
            else
            {
                // publish_map(pubLaserCloudMap);
            }

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
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
                time_log_counter ++;
                const char* mode_name = is_localization_mode() ? "localization" : "mapping";
                printf("[ %s ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",
                    mode_name, t1-t0, aver_time_match, aver_time_solve, t3-t1, t5-t3,
                    aver_time_consu, aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
