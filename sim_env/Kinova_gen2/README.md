# Kinova Gen2 Arm Control with PS4 Controller

This project implements a system for controlling a Kinova Gen2 robotic arm using a PS4 controller and recording the movements and camera images for training robotic manipulation models.

## System Overview

The system consists of four main components:
1. PS4 Controller Interface
2. Kinova Arm Control System
3. Camera Interface
4. Data Recording and Processing Pipeline

## Requirements

- Kinova Gen2 Robotic Arm
- JACO-SDK
- PS4 Controller
- USB Camera or RealSense Camera
- Python 3.x
- ROS (Robot Operating System)
- Required Python packages:
  - `pyPS4Controller` for PS4 controller support
  - `opencv-python` for camera interface
  - `numpy` for data processing


## Implementation Plan

### 1. Controller Setup
- Install PS4 controller drivers
- Implement controller input mapping
- Configure joystick axes for arm control:
  - Left stick: X-Y translation
  - Right stick: Z translation and rotation
  - Triggers: Gripper control
  - Buttons: Mode switching and recording control

### 2. Kinova Arm Control
- Initialize JACO-SDK
- Implement control modes:
  - Cartesian position control
  - Joint velocity control
  - Gripper control
- Create safety limits and emergency stop functionality

### 3. Camera System
- Initialize USB/RealSense camera
- Configure camera parameters:
  - Resolution: 640x480 (configurable)
  - Frame rate: 30 FPS
  - Image format: RGB
- Implement camera calibration
- Synchronize image capture with arm movements

### 4. Data Recording System
- Design data structure for synchronized recording:
  - Joint positions/velocities
  - End-effector pose
  - Controller inputs
  - Camera images
  - Timestamps
- Implement recording functionality:
  - Start/stop recording with controller buttons
  - Save trajectories and images in structured format
  - Include metadata for each recording session

### 5. Data Processing Pipeline
- Convert raw recordings to training data
- Implement data preprocessing:
  - Trajectory smoothing
  - Resampling
  - Feature extraction
- Export data in format suitable for ML training

## Project Structure
```
.
├── src/
│   ├── controller/      # PS4 controller interface
│   ├── kinova_control/  # Arm control implementation
│   ├── data_recorder/   # Recording functionality
│   
├── data/              # Recorded trajectories
└── main.py      # Data Recording System and data processing scripts
```

## Usage

1. Connect PS4 controller and Kinova arm
2. Launch the control interface:
   ```bash
   python src/main.py
   ```
3. Use controller to operate the arm:
   - [Button mappings to be defined]
   - Press [Button] to start/stop recording
   - Press [Button] for emergency stop





---


# Flow

1. **System Setup**
   - Load the Kinova API library:
     ```cpp
     void* commandLayer_handle = dlopen("Kinova.API.USBCommandLayerUbuntu.so", RTLD_NOW|RTLD_GLOBAL);
     ```
   - Initialize required API functions (key ones for our use case):
     - `InitAPI()`: Start communication
     - `GetDevices()`: Detect connected robots
     - `SetActiveDevice()`: Select our Jaco arm
     - `SendBasicTrajectory()`: Send movement commands
     - `GetAngularCommand()`: Read current joint positions

2. **PS4 Controller Integration**
   - Initialize PS4 controller using `pyPS4Controller` library
   - Map controller inputs to robot commands:
     - Left stick (X,Y): Control joints 1 & 2 (base rotation & shoulder)
     - Right stick (X,Y): Control joints 3 & 4 (elbow & wrist)
     - L2/R2 triggers: Control joint 5 (wrist rotation)
     - L1/R1 buttons: Control joint 6 (end effector rotation)
     - Face buttons: Control gripper open/close
     - Options button: Home position
     - Share button: Emergency stop

3. **Main Control Loop**
   ```
   1. Read PS4 controller input
   2. Convert inputs to robot commands:
      - For position control:
        - Create TrajectoryPoint with Type = ANGULAR_POSITION
        - Map stick values to joint angles
      - For velocity control:
        - Create TrajectoryPoint with Type = ANGULAR_VELOCITY
        - Map stick values to joint velocities
   3. Send commands to robot:
      - Use SendBasicTrajectory() every 5ms
      - Include safety checks for joint limits
   4. Repeat loop
   ```


5. **Program Structure**
   ```
   Initialize API
   ↓
   Connect to Robot
   ↓
   Initialize PS4 Controller
   ↓
   Move to Home Position
   ↓
   Enter Control Loop:
     ├─► Read Controller Input
     ├─► Process Input
     ├─► Generate Robot Command
     ├─► Send Command
     └─► Collect Data(robot trajectory, camera image)
   ↓
   (On Exit) Cleanup:
     ├─► Move to Safe Position
     ├─► Close Controller Connection
     └─► Close API
   ```

This flow provides real-time control of the Jaco arm using the PS4 controller while maintaining safety and stability. The SDK's low-level functions handle the actual communication with the arm, while our control loop translates controller inputs into appropriate robot commands.
