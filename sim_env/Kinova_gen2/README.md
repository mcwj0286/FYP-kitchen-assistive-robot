# Kinova Gen2 Arm Control with PS4 Controller

This project implements a system for controlling a Kinova Gen2 robotic arm using a PS4 controller and recording the movements for training robotic manipulation models.

## System Overview

The system consists of three main components:
1. PS4 Controller Interface
2. Kinova Arm Control System
3. Data Recording and Processing Pipeline

## Requirements

- Kinova Gen2 Robotic Arm
- JACO-SDK
- PS4 Controller
- Python 3.x
- ROS (Robot Operating System)
- `ds4drv` or `pyPS4Controller` for PS4 controller support

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

### 3. Data Recording System
- Design data structure for movement recording:
  - Joint positions/velocities
  - End-effector pose
  - Controller inputs
  - Timestamps
- Implement recording functionality:
  - Start/stop recording with controller buttons
  - Save trajectories in structured format (JSON/CSV)
  - Include metadata for each recording session

### 4. Data Processing Pipeline
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
│   └── processing/      # Data processing scripts
├── config/             # Configuration files
├── data/              # Recorded trajectories
└── scripts/           # Utility scripts
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

## Data Collection Guidelines

1. Plan movements before recording
2. Ensure consistent speed and smooth trajectories
3. Record multiple variations of each movement
4. Document environmental conditions and task parameters

## Next Steps

1. [ ] Set up development environment
2. [ ] Implement PS4 controller interface
3. [ ] Develop Kinova arm control system
4. [ ] Create data recording pipeline
5. [ ] Test and validate system
6. [ ] Collect initial dataset
7. [ ] Implement data processing pipeline

## Safety Considerations

- Always maintain clear workspace
- Test emergency stop functionality before operation
- Monitor arm movements during operation
- Implement velocity and acceleration limits
- Ensure proper calibration before use
