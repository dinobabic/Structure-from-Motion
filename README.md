# Structure from Motion

This repository contains implementation of Structure from Motion (SfM) approach for estimating camera poses and 3D landmarks from a set of cameras with known intrinsics. The solution is wrapped a ROS2 node and camera poses and estimated point cloud are visualized in **RViz2**.


## Running the Code

Create a workspace and clone the repository:

```bash
mkdir -p ws/src
cd ws/src
git clone <this-repository>
```

Build the ROS2 package:

```bash
cd ..
colcon build
source install/local_setup.bash
```

Run the nodes:

```bash
ros2 run sfm sfm_node
```

```bash
ros2 run rviz2 rviz2
```

```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 -1.5708 map world
```

