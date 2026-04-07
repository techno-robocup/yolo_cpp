# YOLO_CPP_DEMO
This is a C++ demo for running YOLO object detection using LibTorch.

This loads the best.torchscript model(which is from [best.pt](https://github.com/techno-robocup/robocup2026-raspberrypi-program/blob/main/best.pt)).

It outputs the detected bounding boxes as "output.jpg" in the current directory.

# How to use

1. Clone this repository and navigate to the project directory.
2. Make sure to have OpenCV installed in the system
3. Download libtorch from the official PyTorch website and extract it at the top directory of this project.
4. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
5. Run the demo:
   ```bash
   ./yolov8_libtorch_inference
   ```
