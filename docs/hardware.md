# Hardware Setup Guide

This guide covers the physical setup, calibration, and configuration of hardware components for the Kitchen Assistive Robot system.

## Hardware Components

### 1. Robotic Arm - Kinova Gen2 JACO

The system uses a Kinova Gen2 JACO 6-DOF robotic arm as the primary manipulation platform.

#### Specifications
- **Degrees of Freedom**: 6 (3 arm joints + 3 wrist joints)
- **Payload**: Up to 2.6 kg
- **Reach**: 98.5 cm
- **Repeatability**: ±0.1 mm
- **Weight**: 5.3 kg
- **Power**: 19V DC, 90W

#### Mounting Requirements
- **Base Mounting**: Secure mounting to stable surface (workbench or robot base)
- **Power Supply**: 19V DC power adapter (included with robot)
- **Communication**: Ethernet connection to control computer
- **Safety**: Emergency stop button within easy reach

### 2. Vision System

The dual-camera system provides comprehensive visual feedback for manipulation tasks.

#### Camera Specifications
- **Model**: Rapoo USB Cameras (or equivalent USB 2.0/3.0 cameras)
- **Resolution**: 1920x1080 @ 30fps (downsampled to 128x128 for processing)
- **Field of View**: Minimum 60° horizontal
- **Auto-focus**: Preferred for varying object distances
- **Low-light Performance**: Important for consistent operation

#### Camera Placement

##### Environment Camera (Overhead)
```
Mounting Position: Above workspace, 1-2 meters height
Angle: 45-60° downward angle for optimal view
Coverage: Full workspace including robot arm and objects
Purpose: Scene understanding, object layout, user interaction monitoring
```

##### Wrist Camera (End-effector Mounted)
```
Mounting Position: Robot wrist, aligned with gripper approach direction
Angle: Parallel to gripper closing direction
Coverage: Close-up view of manipulation target
Purpose: Fine manipulation guidance, grasp verification
```

### 3. Control Computer

#### Minimum Specifications
- **CPU**: Intel i5 or AMD Ryzen 5 (4+ cores)
- **RAM**: 16GB (32GB recommended for training)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070 or better)
- **Storage**: 256GB SSD minimum (1TB recommended)
- **OS**: Ubuntu 20.04 LTS (recommended) or Windows 10+

#### On-Robot Computing (Intel NUC 11)
For mobile deployment, the system can run on compact computers:
- **Model**: Intel NUC 11 with i3/i5 processor
- **RAM**: 16GB DDR4
- **Storage**: 256GB NVMe SSD
- **Connectivity**: WiFi 6, Ethernet, multiple USB 3.0 ports
- **Power**: 12V DC input (can share robot power supply)

### 4. Input Devices

#### PS4 Controller (Teleoperation)
- **Model**: Sony DualShock 4 or DualSense controller
- **Connection**: USB or Bluetooth
- **Purpose**: Human demonstration collection, manual override
- **Battery**: Ensure fully charged for data collection sessions

#### Emergency Stop
- **Type**: Physical emergency stop button
- **Placement**: Within immediate reach of operator
- **Function**: Immediate robot power cutoff and brake engagement

## Physical Setup

### 1. Robot Arm Installation

#### Mounting Procedure
```bash
# 1. Prepare mounting surface
- Ensure flat, stable surface (minimum 1m x 1m)
- Surface must support minimum 20kg distributed load
- Avoid vibration sources (HVAC, heavy machinery)

# 2. Mount robot base
- Use provided mounting holes in robot base
- Secure with M6 bolts (minimum 25mm thread engagement)
- Apply thread locker to prevent loosening
- Verify mount is level using spirit level

# 3. Connect power
- Connect 19V DC power supply to robot base
- Verify power LED illuminates (green)
- Do not power on robot until network configuration complete
```

#### Network Configuration
```bash
# Set up robot network connection
# Default robot IP: 192.168.1.10
# Required: Computer and robot on same subnet

# Configure computer network:
sudo ip addr add 192.168.1.100/24 dev eth0  # Linux
# or use Network Settings GUI

# Test connectivity:
ping 192.168.1.10

# Expected response:
# 64 bytes from 192.168.1.10: icmp_seq=1 ttl=64 time=0.xxx ms
```

### 2. Camera System Setup

#### Environment Camera Mounting
```bash
# Materials needed:
- Camera mounting arm or ceiling mount
- USB 3.0 extension cable (5m maximum)
- Cable management clips

# Installation steps:
1. Position mount 1-2 meters above workspace center
2. Angle camera 45-60° downward
3. Ensure full workspace visibility
4. Test view covers robot base to workspace edge
5. Secure all cables to prevent interference with robot motion
```

#### Wrist Camera Mounting
```bash
# Materials needed:
- Custom 3D printed camera mount (see /hardware/3d_models/)
- M3 bolts and nuts
- USB extension cable (1m)
- Cable spiral wrap

# Installation procedure:
1. Attach custom mount to robot wrist flange
2. Orient camera parallel to gripper axis
3. Secure camera in mount with M3 bolts
4. Route cable along robot arm using spiral wrap
5. Ensure cable has sufficient slack for full robot motion
6. Test full range of motion without cable strain
```

### 3. Workspace Preparation

#### Workspace Layout
```
Recommended workspace: 1.5m x 1.0m area in front of robot

Layout zones:
┌─────────────────────────────────────┐
│  Storage Zone    │   Interaction    │
│  (objects at     │   Zone           │
│   rest)          │   (active work)  │
├─────────────────────────────────────┤
│           Robot Base Zone           │
│        (keep clear for safety)     │
└─────────────────────────────────────┘

Distance from robot base to workspace edge: 40-80cm
```

#### Safety Considerations
```bash
# Safety zone marking
- Mark robot reach radius on floor (1m radius)
- Install safety barriers if needed
- Ensure emergency stop accessible
- Remove fragile items from robot reach
- Verify adequate lighting (minimum 500 lux)

# Safety checklist:
□ Emergency stop tested and functional
□ Robot reach zone clearly marked
□ No loose cables in robot path
□ Workspace clear of obstacles
□ Lighting adequate for camera operation
□ Power connections secure
```

## Software Configuration

### 1. Robot Controller Setup

#### Install Kinova SDK
```bash
# Download and install Kinova Kortex SDK
wget https://artifactory.kinovaapps.com/artifactory/generic-local/kortex/API/2.3.0/kortex_api-2.3.0.post34-py3-none-any.whl
pip install kortex_api-2.3.0.post34-py3-none-any.whl

# Test SDK installation
python -c "import kortex_api; print('Kortex SDK installed successfully')"
```

#### Robot Connection Test
```python
# test_robot_connection.py
from llm_ai_agent.hardware_tools import KinovaArmController

try:
    # Initialize robot connection
    arm = KinovaArmController(
        ip_address="192.168.1.10",
        username="admin", 
        password="admin"
    )
    
    # Test basic connectivity
    print("Robot connection successful")
    print(f"Robot serial number: {arm.get_serial_number()}")
    print(f"Current position: {arm.get_current_position()}")
    
    # Test basic movement
    print("Testing basic movement...")
    arm.move_home()
    print("Home position reached")
    
except Exception as e:
    print(f"Robot connection failed: {e}")
    print("Check network configuration and robot power")
```

### 2. Camera Configuration

#### Camera Detection and Testing
```bash
# List available cameras (Linux)
v4l2-ctl --list-devices

# Test cameras
python test_cameras.py
```

```python
# test_cameras.py
import cv2

def test_camera(camera_index):
    """Test camera connectivity and basic functionality."""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Camera {camera_index}: NOT DETECTED")
        return False
    
    # Test frame capture
    ret, frame = cap.read()
    if ret:
        height, width = frame.shape[:2]
        print(f"Camera {camera_index}: OK - Resolution {width}x{height}")
        
        # Save test image
        cv2.imwrite(f'camera_{camera_index}_test.jpg', frame)
        print(f"Test image saved: camera_{camera_index}_test.jpg")
    else:
        print(f"Camera {camera_index}: FAILED to capture frame")
        return False
    
    cap.release()
    return True

# Test multiple camera indices
for i in range(4):
    test_camera(i)
```

#### Camera Calibration
```python
# calibrate_cameras.py
import cv2
import numpy as np

def calibrate_camera(camera_index, calibration_images_path):
    """
    Calibrate camera using checkerboard pattern.
    Print checkerboard (9x6 squares, 25mm square size) for calibration.
    """
    # Checkerboard dimensions
    CHECKERBOARD = (9, 6)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= 25  # 25mm square size
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    cap = cv2.VideoCapture(camera_index)
    
    print(f"Calibrating camera {camera_index}")
    print("Position checkerboard in view and press SPACE to capture")
    print("Press 'q' when done (minimum 10 images recommended)")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        if ret_corners:
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners, ret_corners)
            cv2.putText(frame, f"Pattern found - Press SPACE to capture ({capture_count} captured)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No pattern detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow(f'Camera {camera_index} Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and ret_corners:
            # Capture calibration image
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            capture_count += 1
            print(f"Captured image {capture_count}")
            
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if capture_count < 5:
        print(f"Insufficient calibration images ({capture_count}). Minimum 5 required.")
        return None
    
    # Perform calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    if ret:
        print(f"Camera {camera_index} calibration successful!")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {dist_coeffs.flatten()}")
        
        # Save calibration data
        np.savez(f'camera_{camera_index}_calibration.npz',
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                rvecs=rvecs,
                tvecs=tvecs)
        
        return camera_matrix, dist_coeffs
    else:
        print(f"Camera {camera_index} calibration failed!")
        return None

# Calibrate both cameras
environment_cam_params = calibrate_camera(0, './calibration_images/environment/')
wrist_cam_params = calibrate_camera(1, './calibration_images/wrist/')
```

### 3. System Integration Testing

#### Hardware Integration Test
```python
# integration_test.py
from llm_ai_agent.agents import create_agent
from llm_ai_agent.hardware_tools import test_all_hardware

def run_integration_test():
    """Comprehensive hardware integration test."""
    
    print("=== Kitchen Assistive Robot Hardware Integration Test ===\n")
    
    # Test 1: Hardware connectivity
    print("1. Testing hardware connectivity...")
    hardware_status = test_all_hardware()
    
    for component, status in hardware_status.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"   {component}: {status_str}")
    
    if not all(hardware_status.values()):
        print("\n⚠️ Some hardware components failed. Check connections and try again.")
        return False
    
    # Test 2: Agent creation with hardware
    print("\n2. Testing agent initialization...")
    try:
        agent = create_agent(
            agent_type="kitchen_assistant",
            use_hardware=True,
            capture_image="environment"
        )
        print("   Agent creation: ✓ PASS")
    except Exception as e:
        print(f"   Agent creation: ✗ FAIL - {e}")
        return False
    
    # Test 3: Basic robot movement
    print("\n3. Testing basic robot movement...")
    try:
        response = agent.process_to_string("Move to home position")
        print("   Robot movement: ✓ PASS")
    except Exception as e:
        print(f"   Robot movement: ✗ FAIL - {e}")
        return False
    
    # Test 4: Vision system
    print("\n4. Testing vision system...")
    try:
        response = agent.process_to_string("What do you see in the workspace?")
        if "see" in response.lower() or "image" in response.lower():
            print("   Vision system: ✓ PASS")
        else:
            print("   Vision system: ⚠️ PARTIAL - Response may not include visual analysis")
    except Exception as e:
        print(f"   Vision system: ✗ FAIL - {e}")
        return False
    
    # Test 5: Memory system
    print("\n5. Testing memory system...")
    try:
        response = agent.process_to_string("Where is the coffee located?")
        if "coffee" in response.lower():
            print("   Memory system: ✓ PASS")
        else:
            print("   Memory system: ⚠️ PARTIAL - No coffee location in memory")
    except Exception as e:
        print(f"   Memory system: ✗ FAIL - {e}")
        return False
    
    print("\n=== Integration Test Complete ===")
    print("✅ All systems operational. Robot ready for use.")
    return True

if __name__ == "__main__":
    success = run_integration_test()
    exit(0 if success else 1)
```

## Calibration Procedures

### 1. Robot Calibration

#### Joint Calibration
```python
# robot_calibration.py
from llm_ai_agent.hardware_tools import KinovaArmController
import numpy as np

def calibrate_robot_joints():
    """Calibrate robot joint limits and accuracy."""
    
    arm = KinovaArmController()
    
    print("Starting robot joint calibration...")
    
    # Move to each joint limit and record positions
    joint_limits = {}
    
    for joint_id in range(6):  # 6 DOF arm
        print(f"Calibrating joint {joint_id + 1}...")
        
        # Move to positive limit
        arm.move_joint_to_limit(joint_id, direction='positive')
        pos_limit = arm.get_joint_position(joint_id)
        
        # Move to negative limit
        arm.move_joint_to_limit(joint_id, direction='negative')
        neg_limit = arm.get_joint_position(joint_id)
        
        joint_limits[joint_id] = {
            'min': neg_limit,
            'max': pos_limit,
            'range': abs(pos_limit - neg_limit)
        }
        
        print(f"   Joint {joint_id + 1} range: {neg_limit:.2f}° to {pos_limit:.2f}°")
    
    # Save calibration data
    np.savez('robot_joint_calibration.npz', **joint_limits)
    print("Robot joint calibration complete!")
    
    return joint_limits
```

#### End-Effector Calibration
```python
def calibrate_end_effector():
    """Calibrate end-effector position accuracy."""
    
    arm = KinovaArmController()
    
    # Define test positions (Cartesian coordinates)
    test_positions = [
        [0.5, 0.0, 0.3, 0.0, 0.0, 0.0],  # Front center
        [0.3, 0.3, 0.3, 0.0, 0.0, 0.0],  # Front right
        [0.3, -0.3, 0.3, 0.0, 0.0, 0.0], # Front left
        [0.6, 0.0, 0.2, 0.0, 0.0, 0.0],  # Extended reach
    ]
    
    position_errors = []
    
    for i, target_pos in enumerate(test_positions):
        print(f"Testing position {i + 1}: {target_pos}")
        
        # Move to target position
        arm.move_to_position(target_pos)
        
        # Wait for movement completion
        arm.wait_for_movement_complete()
        
        # Read actual position
        actual_pos = arm.get_current_position()
        
        # Calculate error
        error = np.linalg.norm(np.array(target_pos[:3]) - np.array(actual_pos[:3]))
        position_errors.append(error)
        
        print(f"   Target: {target_pos[:3]}")
        print(f"   Actual: {actual_pos[:3]}")
        print(f"   Error: {error:.4f}m")
    
    avg_error = np.mean(position_errors)
    max_error = np.max(position_errors)
    
    print(f"\nCalibration Results:")
    print(f"Average position error: {avg_error:.4f}m")
    print(f"Maximum position error: {max_error:.4f}m")
    
    if max_error < 0.005:  # 5mm tolerance
        print("✅ Robot calibration PASSED")
    else:
        print("⚠️ Robot calibration requires attention - errors exceed 5mm tolerance")
    
    return position_errors
```

### 2. Vision-Robot Calibration

#### Hand-Eye Calibration
```python
def perform_hand_eye_calibration():
    """
    Perform hand-eye calibration to determine transformation
    between robot base and camera coordinate systems.
    """
    
    arm = KinovaArmController()
    
    # Calibration target positions (robot base coordinates)
    calibration_positions = [
        [0.4, 0.2, 0.15, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.15, 0.0, 0.0, 0.0],
        [0.4, -0.2, 0.15, 0.0, 0.0, 0.0],
        [0.3, 0.1, 0.25, 0.0, 0.0, 0.0],
        [0.45, 0.15, 0.1, 0.0, 0.0, 0.0]
    ]
    
    robot_positions = []
    camera_positions = []
    
    print("Starting hand-eye calibration...")
    print("Place calibration target (checkerboard) at each robot position")
    
    for i, pos in enumerate(calibration_positions):
        print(f"\nPosition {i + 1}/{len(calibration_positions)}")
        
        # Move robot to calibration position
        arm.move_to_position(pos)
        arm.wait_for_movement_complete()
        
        # Record robot position
        actual_pos = arm.get_current_position()
        robot_positions.append(actual_pos)
        
        input("Position calibration target and press Enter to capture...")
        
        # Capture image and detect calibration target
        camera_pos = detect_calibration_target_in_image()
        
        if camera_pos is not None:
            camera_positions.append(camera_pos)
            print(f"Target detected at camera coordinates: {camera_pos}")
        else:
            print("Failed to detect calibration target - retrying...")
            i -= 1  # Retry this position
            continue
    
    # Compute hand-eye transformation
    transformation_matrix = compute_hand_eye_transformation(
        robot_positions, camera_positions
    )
    
    # Save calibration results
    np.savez('hand_eye_calibration.npz', 
             transformation_matrix=transformation_matrix,
             robot_positions=robot_positions,
             camera_positions=camera_positions)
    
    print("Hand-eye calibration complete!")
    return transformation_matrix
```

## Maintenance and Troubleshooting

### Regular Maintenance Schedule

#### Daily Checks (Before Operation)
```bash
# Daily maintenance checklist
□ Verify robot power and network connectivity
□ Test emergency stop functionality
□ Check camera image quality
□ Verify workspace is clear of obstacles
□ Test basic robot movement (home position)
```

#### Weekly Maintenance
```bash
# Weekly maintenance tasks
□ Clean camera lenses with appropriate cleaning cloth
□ Check all cable connections for wear or damage
□ Verify robot joint movement smoothness
□ Update system logs and check for errors
□ Test full range of robot motion
```

#### Monthly Maintenance
```bash
# Monthly maintenance procedures
□ Re-run calibration verification tests
□ Check robot arm for unusual wear or noise
□ Update software dependencies
□ Backup system configuration and calibration data
□ Perform comprehensive system integration test
```

### Common Issues and Solutions

#### Robot Connection Issues
```bash
# Symptom: Cannot connect to robot
# Troubleshooting steps:

1. Check network connectivity:
   ping 192.168.1.10

2. Verify robot power:
   - Check power LED (should be green)
   - Verify 19V power supply connection

3. Check robot status:
   - Look for error LEDs on robot base
   - Check robot teach pendant for error messages

4. Reset robot connection:
   - Power cycle robot (off 10 seconds, then on)
   - Restart robot control software
   - Re-run connection test
```

#### Camera Issues
```bash
# Symptom: Camera not detected or poor image quality

1. Check camera connections:
   - Verify USB cable connections
   - Try different USB ports
   - Test with shorter USB cables if using extensions

2. Check camera permissions (Linux):
   sudo usermod -a -G video $USER
   # Log out and back in

3. Test camera manually:
   # Linux
   cheese  # GUI camera test
   # or
   v4l2-ctl --list-formats-ext -d /dev/video0

4. Adjust camera settings:
   # Disable auto-exposure for consistent lighting
   v4l2-ctl -d /dev/video0 -c exposure_auto=1
   v4l2-ctl -d /dev/video0 -c exposure_absolute=300
```

#### Movement Accuracy Issues
```bash
# Symptom: Robot movements are inaccurate or inconsistent

1. Re-run robot calibration:
   python robot_calibration.py

2. Check for mechanical issues:
   - Listen for unusual sounds during movement
   - Check for loose connections or worn components
   - Verify mounting stability

3. Verify coordinate systems:
   - Check hand-eye calibration accuracy
   - Re-run vision-robot calibration if needed

4. Update robot firmware:
   # Contact Kinova support for firmware updates
```

### Safety Protocols

#### Emergency Procedures
```bash
# In case of malfunction or dangerous situation:

1. IMMEDIATE ACTIONS:
   - Press emergency stop button
   - Do not attempt to physically stop robot arm
   - Clear area around robot

2. ASSESSMENT:
   - Identify cause of emergency
   - Check for damage to robot or surroundings
   - Document incident details

3. RECOVERY:
   - Only restart after identifying and resolving cause
   - Perform full system check before resuming operation
   - Update safety procedures if needed
```

#### Pre-Operation Safety Check
```python
def safety_check():
    """Perform safety check before robot operation."""
    
    checks = {
        'emergency_stop': 'Press and release emergency stop - robot should stop immediately',
        'workspace_clear': 'Verify workspace is clear of people and obstacles',
        'cable_routing': 'Check all cables are properly routed and secured',
        'lighting': 'Verify adequate lighting for camera operation',
        'communication': 'Test robot communication and response'
    }
    
    print("=== PRE-OPERATION SAFETY CHECK ===")
    
    for check, description in checks.items():
        response = input(f"{description} - OK? (y/n): ")
        if response.lower() != 'y':
            print(f"❌ Safety check failed: {check}")
            print("Do not operate robot until all safety checks pass.")
            return False
    
    print("✅ All safety checks passed. Robot ready for operation.")
    return True
```

This hardware setup guide provides comprehensive coverage of the physical installation, configuration, and maintenance requirements for the Kitchen Assistive Robot system.