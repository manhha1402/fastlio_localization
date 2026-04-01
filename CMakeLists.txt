cmake_minimum_required(VERSION 3.8)
project(fastlio_localization)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -fexceptions)

add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
  include(ProcessorCount)
  ProcessorCount(N)
  message(STATUS "Processor number: ${N}")
  if(N GREATER 4)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=3)
    message(STATUS "core for MP: 3")
  elseif(N GREATER 3)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=2)
    message(STATUS "core for MP: 2")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(ament_cmake REQUIRED)
find_package(OpenMP QUIET)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vm_navigation_msgs REQUIRED)


add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/fastlio_localization/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp)

target_include_directories(fastlio_mapping PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PCL_INCLUDE_DIRS}
)

if(TARGET Eigen3::Eigen)
  target_link_libraries(fastlio_mapping Eigen3::Eigen)
else()
  target_include_directories(fastlio_mapping PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(fastlio_mapping OpenMP::OpenMP_CXX)
endif()

install(DIRECTORY launch config rviz map
  DESTINATION share/${PROJECT_NAME})

ament_target_dependencies(fastlio_mapping
  geometry_msgs
  nav_msgs
  pcl_conversions
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_ros
  vm_navigation_msgs
)

ament_target_dependencies(fastlio_mapping livox_ros_driver2)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
)

install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME})

ament_package()
