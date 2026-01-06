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
Execute following command to publish transformation from map to wrold coordinate frame (enables proper visualization);
```bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 -1.5708 map world
```

## Implementation Details

As mentioned before, the program expects a set of images depicting the same scene with known camera intrinsics. Each image is then processed sequentially and its pose is computed and 3D points appended to the existing point cloud. The complete processing pipeline is:
1. Train a DBoW2 bag of words database - enables efficient retrieval of top k similar images to the query image.
2. Randomly choose the first image. Additionally, using previously trained database, find the most similar image to the first selected image.
3. Estimate relative motion between the two images using epipolar geometry, this step involves following:
   - detect and match features between the two images
   - estimate the Essential matrix using RANSAC
   - decompose the Essential matrix in a rotation matrix and a translation vector
   - with a known pose, triangulate matches and obtain initial point cloud
4. For all the remaining images, do following:
   - using trained databse, find the most similar image to the previously processed one
   - with the known 3D-2D correspondances, find pose of the new image using PnP
   - match features between newely added image and all previosuly processed images - triangulate the matches and add them to the point cloud


## TODO
- Correctly integrate bundle adjustment into the pipeline - it is implemented using **g2o** library, but there is bug.
- Fix bug related to PnP - number of valid 3D-2D correspondances entering the PnP procedure rapidly decreases as more images is being processed. 
