#ifndef FASTLIO_LOCALIZATION_ROS2_TIME_HPP_
#define FASTLIO_LOCALIZATION_ROS2_TIME_HPP_

#include <cmath>
#include <cstdint>

#include <builtin_interfaces/msg/time.hpp>

namespace fastlio_ros2
{

inline double stamp_to_sec(const builtin_interfaces::msg::Time & t)
{
  return static_cast<double>(t.sec) + static_cast<double>(t.nanosec) * 1e-9;
}

inline void set_stamp_from_sec(builtin_interfaces::msg::Time & t, double sec)
{
  t.sec = static_cast<int32_t>(std::floor(sec));
  const double frac = sec - static_cast<double>(t.sec);
  t.nanosec = static_cast<uint32_t>(frac * 1e9);
}

}  // namespace fastlio_ros2

#endif
