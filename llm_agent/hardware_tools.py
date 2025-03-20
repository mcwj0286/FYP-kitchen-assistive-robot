#!/usr/bin/env python3

import os
import sys
import json
import time
import base64
import re
from typing import Dict, Any, Optional, Union, List
import numpy as np
import cv2
import datetime
import logging

# Import Tool from ai_agent
from .ai_agent import Tool

# Add the path to Kinova modules
KINOVA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         "sim_env/Kinova_gen2/src/devices")
if KINOVA_PATH not in sys.path:
    sys.path.append(KINOVA_PATH)

# Import hardware interfaces
try:
    from sim_env.Kinova_gen2.src.devices.kinova_arm_interface import KinovaArmInterface
    from sim_env.Kinova_gen2.src.devices.camera_interface import CameraInterface, MultiCameraInterface
    from sim_env.Kinova_gen2.src.devices.speaker_interface import SpeakerInterface
except ImportError as e:
    print(f"Warning: Failed to import hardware interfaces - {e}")
    print(f"Make sure the paths are correct and the modules are available.")

logger = logging.getLogger("llm_agent")

class RobotArmTool(Tool):
    """Tool for controlling the Kinova robot arm."""
    
    def __init__(self, robot_arm: Optional[KinovaArmInterface] = None):
        """
        Initialize the robot arm tool.
        
        Args:
            robot_arm (KinovaArmInterface, optional): An existing robot arm interface instance.
                If None, a new instance will be created when needed.
        """
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
        self._robot_arm = robot_arm
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure the robot arm is initialized."""
        if self._robot_arm is None:
            print("Initializing robot arm...")
            self._robot_arm = KinovaArmInterface()
            self._robot_arm.connect()
        self._initialized = True
    
    def _process_robot_command(self, command_str: str) -> str:
        """
        Process a command for the robot arm.
        
        Args:
            command_str (str): The command string to process
            
        Returns:
            str: Result of the command execution
        """
        try:
            # Parse the command
            parts = command_str.strip().split()
            if not parts:
                return "Error: Empty command"
            
            command = parts[0].lower()
            
            # Initialize the robot arm if needed
            try:
                self._ensure_initialized()
            except Exception as e:
                return f"Error initializing robot arm: {e}"
            
            # Process the command
            if command == "move_home":
                self._robot_arm.move_home()
                return "Robot arm is moving to home position"
                
            elif command == "move_default":
                self._robot_arm.move_default()
                return "Robot arm is moving to default position"
                
            elif command == "move_joint":
                if len(parts) < 7:
                    return "Error: move_joint requires 6 joint angles"
                
                try:
                    joint_angles = [float(angle) for angle in parts[1:7]]
                    self._robot_arm.set_angular_control()
                    self._robot_arm.send_angular_position(joint_angles)
                    return f"Robot arm is moving to joint angles: {joint_angles}"
                except ValueError:
                    return "Error: Joint angles must be numeric values"
                
            elif command == "move_position":
                if len(parts) < 7:
                    return "Error: move_position requires x y z rx ry rz values"
                
                try:
                    position = [float(parts[1]), float(parts[2]), float(parts[3])]
                    rotation = [float(parts[4]), float(parts[5]), float(parts[6])]
                    
                    # Optional: Add fingers parameter if provided
                    fingers = [0.0, 0.0, 0.0]  # Default to open
                    if len(parts) > 7:
                        fingers = [float(parts[7])] * 3
                    
                    self._robot_arm.set_cartesian_control()
                    self._robot_arm.send_cartesian_position(position, rotation, fingers=fingers)
                    return f"Robot arm is moving to position: {position}, rotation: {rotation}"
                except ValueError:
                    return "Error: Position and rotation values must be numeric"
                
            elif command == "gripper":
                if len(parts) < 2:
                    return "Error: gripper command requires open|close|position parameter"
                
                action = parts[1].lower()
                
                if action == "open":
                    # Get current position
                    current_joints = self._robot_arm.get_joint_angles()
                    if current_joints:
                        # Keep joint angles, set fingers to open
                        self._robot_arm.send_angular_position(current_joints[:6], fingers=(0.0, 0.0, 0.0))
                        return "Gripper is opening"
                    else:
                        return "Error: Failed to get current joint positions"
                        
                elif action == "close":
                    # Get current position
                    current_joints = self._robot_arm.get_joint_angles()
                    if current_joints:
                        # Keep joint angles, set fingers to closed (around 6000)
                        self._robot_arm.send_angular_position(current_joints[:6], fingers=(6000.0, 6000.0, 6000.0))
                        return "Gripper is closing"
                    else:
                        return "Error: Failed to get current joint positions"
                        
                elif action == "position":
                    if len(parts) < 3:
                        return "Error: gripper position requires a value (0-6000)"
                    
                    try:
                        position = float(parts[2])
                        # Get current position
                        current_joints = self._robot_arm.get_joint_angles()
                        if current_joints:
                            # Keep joint angles, set fingers to specified position
                            self._robot_arm.send_angular_position(current_joints[:6], fingers=(position, position, position))
                            return f"Gripper is moving to position: {position}"
                        else:
                            return "Error: Failed to get current joint positions"
                    except ValueError:
                        return "Error: Gripper position must be numeric"
                        
                else:
                    return f"Error: Unknown gripper action: {action}"
                
            elif command == "get_position":
                position = self._robot_arm.get_cartesian_position()
                if position:
                    return f"Current position: {position}"
                else:
                    return "Error: Failed to get current position"
                
            elif command == "get_joints":
                joint_angles = self._robot_arm.get_joint_angles()
                if joint_angles:
                    return f"Current joint angles: {joint_angles}"
                else:
                    return "Error: Failed to get current joint angles"
                
            else:
                return f"Error: Unknown command: {command}"
                
        except Exception as e:
            return f"Error executing robot command: {e}"

    def close(self):
        """Clean up the robot arm connection."""
        if self._robot_arm:
            try:
                self._robot_arm.close()
                print("Robot arm connection closed")
            except Exception as e:
                print(f"Error closing robot arm: {e}")
            finally:
                self._robot_arm = None
                self._initialized = False

class CameraTool(Tool):
    """Tool for capturing and processing images from cameras."""
    
    def __init__(self, cameras: Optional[MultiCameraInterface] = None):
        """
        Initialize the camera tool.
        
        Args:
            cameras (MultiCameraInterface, optional): An existing camera interface instance.
                If None, a new instance will be created when needed.
        """
        super().__init__(
            name="camera",
            description=(
                "Camera interface for capturing images. Commands:\n"
                "- capture: Capture images from all available cameras\n"
                "- list: List available cameras"
            ),
            function=self._process_camera_command
        )
        self._cameras = cameras
        self._initialized = False
        self._last_images = {}  # Dictionary mapping camera IDs to frames
        self._last_image_path = None  # Path to the last saved image
    
    def _ensure_initialized(self):
        """Ensure the camera interface is initialized."""
        if self._cameras is None:
            max_attempts = 3
            attempt = 1
            last_error = None
            
            while attempt <= max_attempts:
                try:
                    print(f"Initializing camera (attempt {attempt}/{max_attempts})...")
                    self._cameras = MultiCameraInterface()
                    time.sleep(1)  # Give cameras time to initialize
                    print(f"Camera initialized successfully with {self._cameras.num_cameras} cameras on attempt {attempt}")
                    self._initialized = True
                    return
                except Exception as e:
                    last_error = e
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    attempt += 1
                    time.sleep(1)  # Wait before retry
            
            # If we reach here, all attempts failed
            raise RuntimeError(f"Failed to initialize camera after {max_attempts} attempts. Last error: {last_error}")
    
    def _capture_images(self) -> Dict[int, str]:
        """
        Capture images from all available cameras.
        
        Returns:
            Dict[int, str]: Dictionary mapping camera IDs to file paths of saved images.
        """
        # Capture from all cameras
        frames = self._cameras.capture_frames()
        if not frames:
            return {}
            
        # Save debug images with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = {}
        
        for cam_id, (success, frame) in frames.items():
            if success:
                os.makedirs("debug_images", exist_ok=True)
                debug_path = f"debug_images/camera_{cam_id}_{timestamp}.jpg"
                cv2.imwrite(debug_path, frame)
                saved_paths[cam_id] = debug_path
        
        # Store successful captures
        self._last_images = {k: frame for k, (success, frame) in frames.items() if success}
        
        # Set the last image path for convenience
        if saved_paths:
            self._last_image_path = next(iter(saved_paths.values()))
        
        return saved_paths
    
    def _process_camera_command(self, command_str: str) -> str:
        """
        Process a camera command string.
        
        Args:
            command_str (str): The command string.
            
        Returns:
            str: The result of the command execution.
        """
        try:
            # Parse the command
            command_parts = command_str.strip().split()
            command = command_parts[0].lower() if command_parts else ""
            
            # Ensure camera interface is initialized (should already be initialized at startup)
            self._ensure_initialized()
            
            # Execute the command
            if command == "capture":
                # Since camera is already initialized, directly capture images
                captured_paths = self._capture_images()
                
                if not captured_paths:
                    return "Failed to capture images from any camera"
                
                # Debug logging and saving the captured images
                for camera_id, frame in self._last_images.items():
                    if frame is not None:
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_path = f"debug_images/camera_{camera_id}_{timestamp}.jpg"
                        os.makedirs("debug_images", exist_ok=True)
                        cv2.imwrite(debug_path, frame)
                        logging.info(f"Debug image saved to {debug_path}")
                        print(f"Debug image saved to {debug_path}")
                
                camera_count = len(captured_paths)
                return f"Successfully captured images from {camera_count} cameras"
                
            elif command == "list":
                if not self._cameras:
                    return "No cameras available"
                
                camera_info = []
                for cam_id in range(self._cameras.num_cameras):
                    resolution = self._cameras.get_camera_resolution(cam_id)
                    fps = self._cameras.get_camera_fps(cam_id)
                    camera_info.append(f"Camera {cam_id}: {resolution[0]}x{resolution[1]} @ {fps}fps")
                
                return "Available cameras:\n" + "\n".join(camera_info)
                
            else:
                return f"Unknown camera command: {command}"
        
        except Exception as e:
            logging.error(f"Error processing camera command: {e}")
            return f"Error: {str(e)}"
    
    def get_last_image(self, camera_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get the last captured image.
        
        Args:
            camera_id (int, optional): Specific camera ID to get image from.
                                     If None, returns the first available image.
        
        Returns:
            np.ndarray or None: The image as a numpy array, or None if no image available.
        """
        if not self._last_images:
            return None
        
        if camera_id is not None:
            return self._last_images.get(camera_id)
        else:
            # Return the first image in the dictionary
            return next(iter(self._last_images.values()), None)
    
    def get_last_image_path(self) -> Optional[str]:
        """
        Get the path to the last saved debug image.
        
        Returns:
            str or None: Path to the last saved image, or None if no image available
        """
        return getattr(self, '_last_image_path', None)
    
    def get_b64_image(self, camera_id: Optional[int] = None) -> Optional[str]:
        """
        Get the last captured image as a base64 string.
        
        Args:
            camera_id (int, optional): Specific camera ID to get image from.
                                     If None, returns the first available image.
        
        Returns:
            str or None: Base64 encoded image string, or None if no image available.
        """
        import base64
        
        image = self.get_last_image(camera_id)
        if image is None:
            return None
        
        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{encoded_image}"
    
    def close(self):
        """Close the camera interface."""
        if self._cameras:
            try:
                self._cameras.close()
                print("Camera interface closed")
            except Exception as e:
                print(f"Error closing camera interface: {e}")

class SpeakerTool(Tool):
    """Tool for text-to-speech and audio playback."""
    
    def __init__(self, speaker: Optional[SpeakerInterface] = None):
        """
        Initialize the speaker tool.
        
        Args:
            speaker (SpeakerInterface, optional): An existing speaker interface instance.
                If None, a new instance will be created when needed.
        """
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
        self._speaker = speaker
        self._initialized = False
    
    def _ensure_initialized(self):
        """Ensure the speaker is initialized."""
        if self._speaker is None:
            print("Initializing speaker...")
            self._speaker = SpeakerInterface()
            self._initialized = True
    
    def _process_speaker_command(self, command_str: str) -> str:
        """
        Process a command for the speaker.
        
        Args:
            command_str (str): The command string to process
            
        Returns:
            str: Result of the command execution
        """
        try:
            # Parse the command (keeping quoted text intact)
            match = re.match(r'(\w+)\s*(.*)$', command_str.strip())
            if not match:
                return "Error: Invalid command format"
            
            command = match.group(1).lower()
            arg_text = match.group(2).strip()
            
            # Initialize the speaker if needed
            try:
                self._ensure_initialized()
            except Exception as e:
                return f"Error initializing speaker: {e}"
            
            # Process the command
            if command == "speak":
                if not arg_text:
                    return "Error: speak command requires text"
                
                success = self._speaker.speak(arg_text)
                if success:
                    return f"Speaking: {arg_text}"
                else:
                    return "Error: Failed to speak text"
            
            elif command == "play":
                if not arg_text:
                    return "Error: play command requires an audio file path"
                
                success = self._speaker.play_audio(arg_text)
                if success:
                    return f"Playing audio: {arg_text}"
                else:
                    return f"Error: Failed to play audio file: {arg_text}"
            
            elif command == "stop":
                success = self._speaker.stop()
                if success:
                    return "Stopped playback"
                else:
                    return "Error: Failed to stop playback"
            
            elif command == "set_voice":
                if not arg_text:
                    return "Error: set_voice command requires a voice identifier"
                
                self._speaker.set_voice(arg_text)
                return f"Voice set to: {arg_text}"
            
            elif command == "set_volume":
                if not arg_text:
                    return "Error: set_volume command requires a value (0.0-1.0)"
                
                try:
                    volume = float(arg_text)
                    self._speaker.set_volume(volume)
                    return f"Volume set to: {self._speaker.volume}"
                except ValueError:
                    return "Error: Volume must be a numeric value"
            
            elif command == "set_rate":
                if not arg_text:
                    return "Error: set_rate command requires a value (0.5-2.0)"
                
                try:
                    rate = float(arg_text)
                    self._speaker.set_rate(rate)
                    return f"Speech rate set to: {self._speaker.rate}"
                except ValueError:
                    return "Error: Rate must be a numeric value"
            
            elif command == "set_pitch":
                if not arg_text:
                    return "Error: set_pitch command requires a value (0.5-2.0)"
                
                try:
                    pitch = float(arg_text)
                    self._speaker.set_pitch(pitch)
                    return f"Voice pitch set to: {self._speaker.pitch}"
                except ValueError:
                    return "Error: Pitch must be a numeric value"
            
            else:
                return f"Error: Unknown command: {command}"
                
        except Exception as e:
            return f"Error executing speaker command: {e}"
    
    def close(self):
        """Clean up the speaker resources."""
        if self._speaker:
            try:
                self._speaker.stop()
                print("Speaker resources released")
            except Exception as e:
                print(f"Error closing speaker: {e}")
            finally:
                self._speaker = None
                self._initialized = False

class HardwareToolManager:
    """Manager for hardware tool interfaces."""
    
    def __init__(self):
        """Initialize the hardware tool manager."""
        self.robot_arm_tool = None
        self.camera_tool = None
        self.speaker_tool = None
        self.initialize_tools()
    
    def initialize_tools(self):
        """Initialize all available hardware tools."""
        # Initialize robot arm tool if available
        try:
            self.robot_arm_tool = RobotArmTool()
            print("Robot arm tool initialized")
            # Force immediate hardware initialization
            self.robot_arm_tool._ensure_initialized()
            print("Robot arm hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize robot arm tool: {e}")
        
        # Initialize camera tool if available
        try:
            self.camera_tool = CameraTool()
            print("Camera tool initialized")
            # Force immediate hardware initialization
            self.camera_tool._ensure_initialized()
            print("Camera hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize camera tool: {e}")
        
        # Initialize speaker tool if available
        try:
            self.speaker_tool = SpeakerTool()
            print("Speaker tool initialized")
            # Force immediate hardware initialization
            self.speaker_tool._ensure_initialized()
            print("Speaker hardware fully initialized")
        except Exception as e:
            print(f"Warning: Could not initialize speaker tool: {e}")
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all initialized hardware tools.
        
        Returns:
            List[Tool]: List of available hardware tools.
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
        """Close all hardware tool interfaces."""
        # Close robot arm interface
        if self.robot_arm_tool:
            try:
                self.robot_arm_tool.close()
            except Exception as e:
                print(f"Error closing robot arm tool: {e}")
        
        # Close camera interface
        if self.camera_tool:
            try:
                self.camera_tool.close()
            except Exception as e:
                print(f"Error closing camera tool: {e}")
        
        # Close speaker interface
        if self.speaker_tool:
            try:
                self.speaker_tool.close()
            except Exception as e:
                print(f"Error closing speaker tool: {e}")

def main():
    """Test the hardware tools."""
    tool_manager = HardwareToolManager()
    
    try:
        # Initialize the tools
        tool_manager.initialize_tools()
        tools = tool_manager.get_all_tools()
        
        print(f"Available tools: {[tool.name for tool in tools]}")
        
        # Test camera tool if available
        if tool_manager.camera_tool:
            print("\nTesting camera tool...")
            result = tool_manager.camera_tool("list")
            print(f"Camera list result: {result}")
            
            result = tool_manager.camera_tool("capture")
            print(f"Camera capture result: {result}")
        
        # Test speaker tool if available
        if tool_manager.speaker_tool:
            print("\nTesting speaker tool...")
            result = tool_manager.speaker_tool("speak Hello, I am the AI agent's voice interface")
            print(f"Speak result: {result}")
        
        # Test robot arm tool if available
        if tool_manager.robot_arm_tool:
            print("\nTesting robot arm tool...")
            result = tool_manager.robot_arm_tool("get_position")
            print(f"Get position result: {result}")
        
    except Exception as e:
        print(f"Error testing hardware tools: {e}")
    finally:
        # Clean up
        tool_manager.close_all()

if __name__ == "__main__":
    main() 