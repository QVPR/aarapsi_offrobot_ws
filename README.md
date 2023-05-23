# aarapsi_offrobot_ws

## Packages:
1. robot_gui_bridge
  - Web server + gui (html + css + javascript -> ROS) for interfacing with robot
2. aarapsi_intro_pack
  - Main package
3. hdl_graph_slam
4. hdl_localization
5. hdl_global_localization
6. ndt_omp
7. fast_gicp

To build:
- On arm64/aarch64:
  - Edit CMakeLists.txt for the hdl_localization and hdl_graph_slam:
      ```
      if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        add_definitions(-std=c++14) # was 11
        set(CMAKE_CXX_FLAGS "-std=c++14") # was 11
      ```
  - catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_SYSTEM_PROCESSOR="aarch64" -DBUILD_WITH_MARCH_NATIVE=True -DBUILD_VGICP_CUDA=ON
- Otherwise:
  - catkin_make -DCMAKE_BUILD_TYPE=Release
