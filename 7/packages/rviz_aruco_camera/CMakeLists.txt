cmake_minimum_required(VERSION 3.16)
project(rviz_aruco_camera)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(ffmpeg_encoder_decoder REQUIRED)
find_package(ffmpeg_image_transport_msgs REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS aruco core imgproc)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz_default_plugins REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(yaml_cpp_vendor REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(aruco_processor STATIC src/aruco_processor.cpp)
# the shared display library links the static processor
set_target_properties(aruco_processor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(aruco_processor PUBLIC cxx_std_17)
target_include_directories(aruco_processor PUBLIC src ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(aruco_processor cv_bridge sensor_msgs)
target_link_libraries(aruco_processor ${OpenCV_LIBS} nlohmann_json::nlohmann_json yaml-cpp)

add_library(${PROJECT_NAME} SHARED src/aruco_camera_display.cpp)
# generate qt metadata for the display class
set_target_properties(${PROJECT_NAME} PROPERTIES AUTOMOC ON)
ament_target_dependencies(
  ${PROJECT_NAME} ffmpeg_encoder_decoder ffmpeg_image_transport_msgs pluginlib rclcpp
  rviz_common rviz_default_plugins sensor_msgs std_msgs)
target_link_libraries(${PROJECT_NAME} aruco_processor Qt5::Widgets Threads::Threads)
# expose the display in the rviz add menu
pluginlib_export_plugin_description_file(rviz_common plugins.xml)
install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_aruco_processor test/test_aruco_processor.cpp)
  if(TARGET test_aruco_processor)
    target_link_libraries(test_aruco_processor aruco_processor)
  endif()
endif()

ament_package()
