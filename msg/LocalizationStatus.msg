# Localization status from FAST-LIO tracking.
# Header with timestamp and frame information
std_msgs/Header header

# Status classification: GOOD, DEGRADED, or LOST
string status

# Registration fitness score (0.0 to 1.0, higher is better)
float64 fitness_score

# Inlier RMSE from registration (lower is better)
float64 inlier_rmse

# Pose distance error compared to odometry prediction (lower is better)
float64 pose_distance_error
