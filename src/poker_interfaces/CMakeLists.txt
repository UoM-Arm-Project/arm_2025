cmake_minimum_required(VERSION 3.8)
project(poker_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TargetPose.msg"
  "msg/ServoCommand.msg"
  "msg/MotorFeedback.msg"
  "msg/TargetJoints.msg"
  "action/MoveJoints.action"
  "action/MovePose.action"
  "msg/CardDetection.msg"
  "msg/CardDetectionArray.msg"
  "msg/PotEstimate.msg"
  "msg/PlayerInput.msg"
  "msg/GameState.msg"
  "action/PokerAction.action"
  "action/DealerAction.action"
  "srv/ReadCards.srv"
  "srv/ReadPot.srv"
  DEPENDENCIES action_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()