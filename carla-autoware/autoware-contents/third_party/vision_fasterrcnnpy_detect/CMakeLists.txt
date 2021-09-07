cmake_minimum_required(VERSION 2.8.12)
project(vision_fasterrcnnpy_detect)

include(FindPkgConfig)

find_package(autoware_build_flags REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  autoware_config_msgs
  autoware_msgs
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
  tf
)

find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(Python3 COMPONENTS Development)

catkin_package()

set(CMAKE_CXX_FLAGS "-O2 -g -Wall ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
  ${catkin_INCLUDE_DIRS}
  include/
)


#####Pytorch dependencies########
AW_CHECK_CUDA()

# Temporary hack to force compilation.
if(TRUE)#USE_CUDA AND EXISTS "${TORCH_LIBRARIES}")
  include_directories(${CUDA_INCLUDE_DIRS})

  if("${CMAKE_SYSTEM_PROCESSOR}" MATCHES "^arm")
    link_directories(/usr/lib/arm-linux-gnueabihf/tegra)
  endif()
  add_executable(vision_fasterrcnnpy_detect
    src/vision_fasterrcnnpy_detect.cpp
    src/vision_fasterrcnnpy_detect_node.cpp
  )

  target_link_libraries(vision_fasterrcnnpy_detect
    ${catkin_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_curand_LIBRARY}
    ${TORCH_LIBRARIES}
    glog
    TorchVision::TorchVision
    Python3::Python
  )

  add_dependencies(vision_fasterrcnnpy_detect
    ${catkin_EXPORTED_TARGETS}
  )

  set_property(TARGET vision_fasterrcnnpy_detect PROPERTY CXX_STANDARD 14)

  # The following code block is suggested to be used on Windows.
  # According to https://github.com/pytorch/pytorch/issues/25457,
  # the DLLs need to be copied to avoid memory errors.
  if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:example-app>)
  endif (MSVC)

  install(DIRECTORY include/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  )

  install(TARGETS vision_fasterrcnnpy_detect
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
  install(DIRECTORY launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
    PATTERN ".svn" EXCLUDE
  )
else()
  message("'TorchScript' is not available. This will not be built.")
endif()
