#!/usr/bin/env python3

import os
import time
import logging
import random
import cv2
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from llm_agent.hardware_tools import Tool, RobotArmTool, CameraTool, SpeakerTool

logger = logging.getLogger("llm_agent")

class MockRobotArmTool(Tool):
    """Mock implementation of the robot arm tool for testing without hardware."""
    
    def __init__(self):
        """Initialize the mock robot arm tool."""
        super().__init__(
            name="robot_arm",
            description=(
                "Control the Kinova robot arm. Commands:\n"
                "- move_home: Move arm to home position\n"
                "- move_default: Move arm to default position\n"
                "- move_joint: Move arm joints (angles in degrees). Usage: move_joint j1 j2 j3 j4 j5 j6\n"
                "- move_position: Move arm to Cartesian position. Usage: move_position x y z rx ry rz\n"
                "- gripper: Control the gripper. Usage: gripper open|close|position (0-6000)\n"
                "- get_position: Get current position of the arm\n"
                "- get_joints: Get current joint angles"
            ),
            function=self._process_robot_command
        )
        # Mock state
        self.current_position = {
            "x": 0.2, "y": 0.0, "z": 0.3, 
            "rx": 3.14, "ry": 0.0, "rz": 0.0
        }
        self.current_joints = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.gripper_position = 0  # 0 = open, 6000 = closed
        self.home_position = {
            "x": 0.4, "y": 0.0, "z": 0.4, 
            "rx": 3.14, "ry": 0.0, "rz": 0.0
        }
        self.default_position = {
            "x": 0.3, "y": 0.0, "z": 0.3, 
            "rx": 3.14, "ry": 0.0, "rz": 0.0
        }
        
        # Create debug directory if it doesn't exist
        os.makedirs("debug_mock", exist_ok=True)
    
    def _process_robot_command(self, command_str: str) -> str:
        """Process robot arm commands."""
        parts = command_str.split()
        cmd = parts[0] if parts else ""
        
        # Simulate a short delay
        time.sleep(0.5)
        
        # Log the command for debugging
        logger.info(f"Mock robot arm executing: {command_str}")
        
        # Process commands
        if cmd == "move_home":
            self.current_position = self.home_position.copy()
            self._update_visualization()
            return "Successfully moved to home position"
            
        elif cmd == "move_default":
            self.current_position = self.default_position.copy()
            self._update_visualization()
            return "Successfully moved to default position"
            
        elif cmd == "move_position" and len(parts) >= 7:
            try:
                # Parse position values
                x, y, z, rx, ry, rz = map(float, parts[1:7])
                self.current_position = {
                    "x": x, "y": y, "z": z,
                    "rx": rx, "ry": ry, "rz": rz
                }
                self._update_visualization()
                return f"Successfully moved to position: x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}"
            except Exception as e:
                return f"Error parsing position values: {str(e)}"
                
        elif cmd == "move_joint" and len(parts) >= 7:
            try:
                # Parse joint values
                self.current_joints = list(map(float, parts[1:7]))
                self._update_visualization()
                return f"Successfully moved joints to: {' '.join(map(str, self.current_joints))}"
            except Exception as e:
                return f"Error parsing joint values: {str(e)}"
                
        elif cmd == "gripper":
            if len(parts) < 2:
                return "Error: Missing gripper command. Use 'open', 'close', or 'position'"
                
            subcmd = parts[1]
            
            if subcmd == "open":
                self.gripper_position = 0
                self._update_visualization()
                return "Gripper opened successfully"
                
            elif subcmd == "close":
                self.gripper_position = 6000
                self._update_visualization()
                return "Gripper closed successfully"
                
            elif subcmd == "position" and len(parts) >= 3:
                try:
                    # Fix: First convert to float, then to int to handle values like "2544.0000"
                    position = int(float(parts[2]))
                    if 0 <= position <= 6000:
                        self.gripper_position = position
                        self._update_visualization()
                        return f"Gripper position set to {position}"
                    else:
                        return "Error: Gripper position must be between 0 and 6000"
                except Exception as e:
                    return f"Error parsing gripper position: {str(e)}"
            else:
                return f"Unknown gripper command: {subcmd if len(parts) > 1 else 'missing'}"
                
        elif cmd == "get_position":
            pos = self.current_position
            return f"Current position:\nX: {pos['x']}\nY: {pos['y']}\nZ: {pos['z']}\nRoll: {pos['rx']}\nPitch: {pos['ry']}\nYaw: {pos['rz']}"
            
        elif cmd == "get_joints":
            return f"Current joint angles:\nJoint 1: {self.current_joints[0]}\nJoint 2: {self.current_joints[1]}\nJoint 3: {self.current_joints[2]}\nJoint 4: {self.current_joints[3]}\nJoint 5: {self.current_joints[4]}\nJoint 6: {self.current_joints[5]}"
            
        else:
            return f"Unknown command: {cmd}"
    
    def _update_visualization(self):
        """Update visualization of robot state (optional)."""
        # This could create a simple image showing robot state
        # For now, just log the state
        state = {
            "position": self.current_position,
            "joints": self.current_joints,
            "gripper": self.gripper_position
        }
        logger.info(f"Mock robot state updated: {state}")
        
        # Create a simple visualization
        viz_img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White background
        
        # Draw robot arm position (simplified visualization)
        cv2.putText(viz_img, "Robot Arm Visualization", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Position info
        pos_text = f"Position: X={self.current_position['x']:.2f}, Y={self.current_position['y']:.2f}, Z={self.current_position['z']:.2f}"
        cv2.putText(viz_img, pos_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Joint info
        joint_text = f"Joints: {' '.join([f'{j:.1f}' for j in self.current_joints])}"
        cv2.putText(viz_img, joint_text, (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Gripper info
        gripper_text = f"Gripper: {self.gripper_position}/6000"
        gripper_state = "OPEN" if self.gripper_position < 1000 else ("PARTIAL" if self.gripper_position < 5000 else "CLOSED")
        cv2.putText(viz_img, f"{gripper_text} ({gripper_state})", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw a simple arm visualization
        # Base
        cv2.rectangle(viz_img, (290, 400), (350, 450), (100, 100, 100), -1)
        
        # Calculate arm position (simplified)
        arm_length = 200
        # Scale Z position to drawing
        z_factor = 1.0 - min(1.0, max(0.0, self.current_position['z']))
        arm_length_scaled = int(arm_length * (0.5 + z_factor * 0.5))
        
        # Draw arm
        cv2.line(viz_img, (320, 400), (320, 400 - arm_length_scaled), (0, 0, 200), 4)
        
        # Calculate end effector position based on X and Y
        x_offset = int(self.current_position['x'] * 100)
        y_offset = int(self.current_position['y'] * 100)
        end_x = 320 + x_offset
        end_y = 400 - arm_length_scaled
        
        # Draw end effector
        if self.gripper_position < 1000:  # Open
            cv2.line(viz_img, (end_x - 20, end_y), (end_x + 20, end_y), (0, 200, 0), 3)
        elif self.gripper_position >= 5000:  # Closed
            cv2.circle(viz_img, (end_x, end_y), 10, (0, 200, 0), -1)
        else:  # Partially open
            cv2.line(viz_img, (end_x - 10, end_y), (end_x + 10, end_y), (0, 200, 0), 3)
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_path = f"debug_mock/robot_arm_{timestamp}.jpg"
        cv2.imwrite(viz_path, viz_img)
        logger.info(f"Robot visualization saved to: {viz_path}")
        
    def close(self):
        """Close the mock robot arm tool."""
        logger.info("Mock robot arm closed")
        return "Robot arm connection closed"


class MockCameraTool(Tool):
    """Mock implementation of the camera tool for testing without hardware."""
    
    def __init__(self):
        """Initialize the mock camera tool."""
        super().__init__(
            name="camera",
            description=(
                "Access camera feeds. Commands:\n"
                "- capture: Capture images from all cameras\n"
                "- capture <camera_id>: Capture image from a specific camera\n"
                "- list: List available cameras"
            ),
            function=self._process_camera_command
        )
        # Mock state
        self.cameras = {
            1: {"resolution": (640, 480)},  # Main camera
            2: {"resolution": (320, 240)}   # Secondary camera (optional)
        }
        self.last_images = {}
        self.last_image_paths = {}
        
        # Create debug directories
        os.makedirs("debug_images", exist_ok=True)
        os.makedirs("debug_mock", exist_ok=True)
        
        # Generate some mock scenarios for testing
        self._generate_mock_scenarios()
    
    def _generate_mock_scenarios(self):
        """Generate mock scenarios for testing."""
        # Create base image (empty scene)
        base_image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray
        
        # Add a table surface
        cv2.rectangle(base_image, (50, 350), (590, 450), (120, 80, 40), -1)  # Brown table
        
        # Add some ambient elements
        cv2.rectangle(base_image, (400, 100), (500, 200), (0, 0, 200), -1)  # Blue object
        cv2.circle(base_image, (150, 150), 40, (0, 200, 0), -1)  # Green object
        
        # Draw text indicating this is a mock image
        cv2.putText(base_image, "MOCK CAMERA - Test Environment", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Save base scenario (empty)
        self.scenarios = {
            "empty": base_image.copy(),
        }
        
        # Scenario with jar
        jar_image = base_image.copy()
        # Draw a jar
        cv2.circle(jar_image, (320, 300), 40, (200, 200, 200), -1)  # Jar body
        cv2.rectangle(jar_image, (295, 260), (345, 270), (100, 100, 100), -1)  # Jar lid
        cv2.putText(jar_image, "JAR", (310, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        self.scenarios["jar_on_table"] = jar_image
        
        # Scenario with jar on gripper
        jar_on_gripper = base_image.copy()
        # Draw robot arm
        cv2.rectangle(jar_on_gripper, (290, 300), (350, 450), (100, 100, 100), 2)  # Arm
        # Draw a jar on the gripper
        cv2.circle(jar_on_gripper, (320, 280), 35, (200, 200, 200), -1)  # Jar body
        cv2.rectangle(jar_on_gripper, (300, 245), (340, 255), (100, 100, 100), -1)  # Jar lid
        cv2.putText(jar_on_gripper, "JAR ON GRIPPER", (250, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        self.scenarios["jar_on_gripper"] = jar_on_gripper
        
        # Scenario with opened jar
        jar_opened = base_image.copy()
        # Draw robot arm with gripper
        cv2.rectangle(jar_opened, (290, 300), (350, 450), (100, 100, 100), 2)  # Arm
        # Draw an opened jar
        cv2.circle(jar_opened, (320, 280), 35, (200, 200, 200), -1)  # Jar body
        cv2.rectangle(jar_opened, (250, 245), (290, 255), (100, 100, 100), -1)  # Jar lid (opened)
        cv2.putText(jar_opened, "JAR OPENED", (250, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        self.scenarios["jar_opened"] = jar_opened
        
        # Save all scenarios to disk for reference
        for name, img in self.scenarios.items():
            path = f"debug_mock/scenario_{name}.jpg"
            cv2.imwrite(path, img)
            logger.info(f"Mock scenario saved: {path}")
    
    def _process_camera_command(self, command_str: str) -> str:
        """Process camera commands."""
        parts = command_str.split()
        cmd = parts[0] if parts else ""
        
        # Log the command for debugging
        logger.info(f"Mock camera executing: {command_str}")
        
        # Process commands
        if cmd == "list":
            return f"Available cameras: {', '.join(map(str, self.cameras.keys()))}"
            
        elif cmd == "capture":
            camera_id = int(parts[1]) if len(parts) > 1 else None
            
            # Decide which scenario to use
            # In a real test, you might want to control this programmatically
            # based on the current state of your test
            scenario_name = os.getenv("MOCK_CAMERA_SCENARIO", "empty")
            if scenario_name not in self.scenarios:
                scenario_name = "empty"
                
            if camera_id:
                # Capture from specific camera
                if camera_id not in self.cameras:
                    return f"Error: Camera {camera_id} not found"
                
                # Get or create mock image
                mock_image = self.scenarios[scenario_name].copy()
                
                # Resize if needed
                resolution = self.cameras[camera_id]["resolution"]
                if mock_image.shape[1] != resolution[0] or mock_image.shape[0] != resolution[1]:
                    mock_image = cv2.resize(mock_image, resolution)
                
                # Save the capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"debug_images/camera_{camera_id}_{timestamp}.jpg"
                cv2.imwrite(image_path, mock_image)
                
                # Store last image and path
                self.last_images[camera_id] = mock_image
                self.last_image_paths[camera_id] = image_path
                
                logger.info(f"Debug image saved to {image_path}")
                print(f"Debug image saved to {image_path}")
                
                return f"Successfully captured image from camera {camera_id}"
            else:
                # Capture from all cameras
                success_count = 0
                for cam_id in self.cameras.keys():
                    # Get or create mock image for this camera
                    mock_image = self.scenarios[scenario_name].copy()
                    
                    # Add camera ID to the image
                    cv2.putText(mock_image, f"Camera {cam_id}", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    
                    # Resize if needed
                    resolution = self.cameras[cam_id]["resolution"]
                    if mock_image.shape[1] != resolution[0] or mock_image.shape[0] != resolution[1]:
                        mock_image = cv2.resize(mock_image, resolution)
                    
                    # Save the capture
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = f"debug_images/camera_{cam_id}_{timestamp}.jpg"
                    cv2.imwrite(image_path, mock_image)
                    
                    # Store last image and path
                    self.last_images[cam_id] = mock_image
                    self.last_image_paths[cam_id] = image_path
                    
                    logger.info(f"Debug image saved to {image_path}")
                    print(f"Debug image saved to {image_path}")
                    
                    success_count += 1
                
                return f"Successfully captured images from {success_count} cameras"
        else:
            return f"Unknown command: {cmd}"
    
    def get_last_image(self, camera_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get the last captured image.
        
        Args:
            camera_id (int, optional): The camera ID. If None, returns the first available image.
            
        Returns:
            Optional[np.ndarray]: The last captured image or None if no image is available.
        """
        if camera_id is not None:
            return self.last_images.get(camera_id)
        elif self.last_images:
            return next(iter(self.last_images.values()))
        return None
    
    def get_last_image_path(self) -> Optional[str]:
        """
        Get the path of the last captured image.
        
        Returns:
            Optional[str]: Path to the last captured image or None if no image is available.
        """
        if self.last_image_paths:
            return next(iter(self.last_image_paths.values()))
        return None
    
    def get_b64_image(self, camera_id: Optional[int] = None) -> Optional[str]:
        """
        Get the last captured image as a base64 data URI.
        
        Args:
            camera_id (int, optional): The camera ID. If None, returns the first available image.
            
        Returns:
            Optional[str]: Base64 data URI of the image or None if no image is available.
        """
        import base64
        
        image = self.get_last_image(camera_id)
        if image is None:
            return None
        
        # Convert to JPEG
        success, buffer = cv2.imencode('.jpg', image)
        if not success:
            return None
        
        # Convert to base64
        b64_bytes = base64.b64encode(buffer)
        b64_string = b64_bytes.decode('utf-8')
        
        return f"data:image/jpeg;base64,{b64_string}"
    
    def close(self):
        """Close the mock camera tool."""
        logger.info("Mock camera closed")
        return "Camera interface closed"


class MockSpeakerTool(Tool):
    """Mock implementation of the speaker tool for testing without hardware."""
    
    def __init__(self):
        """Initialize the mock speaker tool."""
        super().__init__(
            name="speaker",
            description=(
                "Text-to-speech and audio playback. Commands:\n"
                "- speak <text>: Convert text to speech\n"
                "- play <audio_file>: Play an audio file\n"
                "- stop: Stop current playback\n"
                "- set_voice <voice>: Set the voice to use\n"
                "- set_volume <0.0-1.0>: Set the volume level\n"
                "- set_rate <0.5-2.0>: Set the speech rate\n"
                "- set_pitch <0.5-2.0>: Set the voice pitch"
            ),
            function=self._process_speaker_command
        )
        # Mock state
        self.voice = "Samantha"
        self.volume = 1.0
        self.rate = 1.0
        self.pitch = 1.0
        self.is_playing = False
        self.play_history = []
    
    def _process_speaker_command(self, command_str: str) -> str:
        """Process speaker commands."""
        parts = command_str.split(maxsplit=1)
        cmd = parts[0] if parts else ""
        
        # Log the command for debugging
        logger.info(f"Mock speaker executing: {command_str}")
        
        # Process commands
        if cmd == "speak" and len(parts) > 1:
            text = parts[1]
            self.is_playing = True
            
            # Simulate speech
            logger.info(f"MOCK SPEECH: '{text}' (voice: {self.voice}, rate: {self.rate}, volume: {self.volume})")
            print(f"🔊 MOCK SPEECH: '{text}'")
            
            # Record in history
            self.play_history.append({
                "type": "speech",
                "text": text,
                "voice": self.voice,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Simulate speech duration (roughly based on text length)
            speech_time = len(text) * 0.05 * (1.0 / self.rate)  # ~50ms per character
            speech_time = min(10.0, max(0.5, speech_time))  # Clamp between 0.5 and 10 seconds
            
            # For testing, we can sleep a short time to simulate
            time.sleep(min(0.5, speech_time))
            
            self.is_playing = False
            return f"Spoke text: '{text}'"
            
        elif cmd == "play" and len(parts) > 1:
            audio_file = parts[1]
            self.is_playing = True
            
            logger.info(f"MOCK AUDIO PLAYBACK: '{audio_file}' (volume: {self.volume})")
            print(f"🔊 MOCK AUDIO: '{audio_file}'")
            
            # Record in history
            self.play_history.append({
                "type": "audio",
                "file": audio_file,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Simulate short playback
            time.sleep(0.5)
            
            self.is_playing = False
            return f"Played audio: '{audio_file}'"
            
        elif cmd == "stop":
            was_playing = self.is_playing
            self.is_playing = False
            
            if was_playing:
                return "Stopped current playback"
            else:
                return "Nothing was playing"
                
        elif cmd == "set_voice" and len(parts) > 1:
            self.voice = parts[1]
            return f"Voice set to: {self.voice}"
            
        elif cmd == "set_volume" and len(parts) > 1:
            try:
                volume = float(parts[1])
                if 0.0 <= volume <= 1.0:
                    self.volume = volume
                    return f"Volume set to: {volume}"
                else:
                    return "Error: Volume must be between 0.0 and 1.0"
            except ValueError:
                return f"Error: Invalid volume value: {parts[1]}"
                
        elif cmd == "set_rate" and len(parts) > 1:
            try:
                rate = float(parts[1])
                if 0.5 <= rate <= 2.0:
                    self.rate = rate
                    return f"Speech rate set to: {rate}"
                else:
                    return "Error: Rate must be between 0.5 and 2.0"
            except ValueError:
                return f"Error: Invalid rate value: {parts[1]}"
                
        elif cmd == "set_pitch" and len(parts) > 1:
            try:
                pitch = float(parts[1])
                if 0.5 <= pitch <= 2.0:
                    self.pitch = pitch
                    return f"Pitch set to: {pitch}"
                else:
                    return "Error: Pitch must be between 0.5 and 2.0"
            except ValueError:
                return f"Error: Invalid pitch value: {parts[1]}"
        else:
            return f"Unknown command: {cmd}"
    
    def is_playing(self) -> bool:
        """Check if speech or audio is currently playing."""
        return self.is_playing
    
    def close(self):
        """Close the mock speaker tool."""
        logger.info("Mock speaker closed")
        return "Speaker interface closed"


class MockHardwareToolManager:
    """Manager for mock hardware tool interfaces."""
    
    def __init__(self):
        """Initialize the mock hardware tool manager."""
        self.robot_arm_tool = None
        self.camera_tool = None
        self.speaker_tool = None
        self.initialize_tools()
    
    def initialize_tools(self):
        """Initialize all available mock hardware tools."""
        # Initialize robot arm tool
        try:
            self.robot_arm_tool = MockRobotArmTool()
            print("Mock robot arm tool initialized")
            # Force immediate initialization
            self.robot_arm_tool._update_visualization()
            print("Mock robot arm hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize mock robot arm tool: {e}")
        
        # Initialize camera tool
        try:
            self.camera_tool = MockCameraTool()
            print("Mock camera tool initialized")
            # Generate scenarios right away
            self.camera_tool._generate_mock_scenarios()
            print("Mock camera hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize mock camera tool: {e}")
        
        # Initialize speaker tool
        try:
            self.speaker_tool = MockSpeakerTool()
            print("Mock speaker tool initialized")
            # No special initialization needed for speaker
            print("Mock speaker hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize mock speaker tool: {e}")
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all initialized mock hardware tools.
        
        Returns:
            List[Tool]: List of available mock hardware tools.
        """
        tools = []
        
        if self.robot_arm_tool:
            tools.append(self.robot_arm_tool)
        
        if self.camera_tool:
            tools.append(self.camera_tool)
        
        if self.speaker_tool:
            tools.append(self.speaker_tool)
        
        return tools
    
    def close_all(self):
        """Close all mock hardware tool interfaces."""
        # Close robot arm interface
        if self.robot_arm_tool:
            try:
                self.robot_arm_tool.close()
            except Exception as e:
                print(f"Error closing mock robot arm tool: {e}")
        
        # Close camera interface
        if self.camera_tool:
            try:
                self.camera_tool.close()
            except Exception as e:
                print(f"Error closing mock camera tool: {e}")
        
        # Close speaker interface
        if self.speaker_tool:
            try:
                self.speaker_tool.close()
            except Exception as e:
                print(f"Error closing mock speaker tool: {e}")


# For testing the mock tools
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize manager
    manager = MockHardwareToolManager()
    
    try:
        # Test robot arm
        if manager.robot_arm_tool:
            print("\n=== Testing Mock Robot Arm ===")
            print(manager.robot_arm_tool("move_home"))
            print(manager.robot_arm_tool("get_position"))
            print(manager.robot_arm_tool("move_position 0.3 0.1 0.2 3.14 0.0 0.0"))
            print(manager.robot_arm_tool("gripper close"))
            print(manager.robot_arm_tool("get_joints"))
        
        # Test camera
        if manager.camera_tool:
            print("\n=== Testing Mock Camera ===")
            print(manager.camera_tool("list"))
            print(manager.camera_tool("capture"))
            print(manager.camera_tool("capture 1"))
        
        # Test speaker
        if manager.speaker_tool:
            print("\n=== Testing Mock Speaker ===")
            print(manager.speaker_tool("speak Hello, this is a test of the mock speaker"))
            print(manager.speaker_tool("set_voice Alex"))
            print(manager.speaker_tool("set_rate 1.5"))
            print(manager.speaker_tool("speak Now I'm speaking faster with a different voice"))
        
        print("\nAll mock tools tested successfully!")
        
    finally:
        # Clean up
        manager.close_all() 