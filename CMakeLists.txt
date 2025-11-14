cmake_minimum_required(VERSION 3.0.2)

project(franka_concerto) # Project name

## Find the necessary ROS packages
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs # Para el nodo conversor
  skeleton_3d # This package contains the Skeleton3D message
  message_generation
  actionlib
  actionlib_msgs
  urdf
)

catkin_python_setup() # Para ejecutar el setup.py para ubicar los módulos. 

## Specify the message files to be generated
add_message_files(
  FILES
  FusionModel.msg
)

add_action_files(
  DIRECTORY action
  FILES 
  MoveFR3.action
  CartesianImpedanceMove.action
  CartesianVelocityMove.action
  HomingEffortMove.action

)

## Generate messages and services with the listed dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs  # Asegúrate de que geometry_msgs esté aquí
  sensor_msgs
  skeleton_3d
  actionlib_msgs
)

## Define the package and its dependencies
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    geometry_msgs
    sensor_msgs # Para el nodo conversor
    skeleton_3d # This package contains the Skeleton3D message
    message_runtime
    actionlib
    actionlib_msgs
    
)

## Include directories for the project
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Install Python executable scripts
install(PROGRAMS
  scripts/publish_eq_pose.py
  scripts/hri_states_publisher.py
  scripts/cartesian_path_planner.py
  scripts/desired_pose_publisher.py
  scripts/hri_states_machine.py
  scripts/fusion_model_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install additional resources (launch files, configs, RViz settings)
install(DIRECTORY 
  launch 
  config
  rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
# RCO: mirar que hace esto
## Optional: Add environment hooks to reload configurations
# catkin_add_env_hooks(99.skeleton_3d
#  SHELLS sh
#  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks
# )

## Display a status message once the CMake configuration completes
message(STATUS "CMake configuration for franka_concerto completed successfully.")
