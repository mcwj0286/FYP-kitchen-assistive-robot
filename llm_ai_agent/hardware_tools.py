#!/usr/bin/env python3
"""
Hardware tools for the Kitchen Assistant Agent.
This module provides a set of tools for interacting with physical hardware:
- Cameras (environment and wrist-mounted)
- Robotic arm (Kinova)
- Speaker system

These tools can be imported and used by the agent to perform physical actions.
"""

import os
import sys
import time
import logging
import base64
import io
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image
import numpy as np
import subprocess
import threading

# Add sim_env to path for imports
# Get the absolute path to the project root directory (one level up from llm_ai_agent)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also add sim_env directory to path explicitly
sim_env_path = os.path.join(project_root, 'sim_env')
if os.path.exists(sim_env_path) and sim_env_path not in sys.path:
    sys.path.insert(0, sim_env_path)

# Import hardware interfaces - adjust paths as needed
try:
    # First try direct import from sim_env
    from sim_env.Kinova_gen2.src.devices.camera_interface import CameraInterface, MultiCameraInterface
    from sim_env.Kinova_gen2.src.devices.kinova_arm_interface import KinovaArmInterface
    from sim_env.Kinova_gen2.src.devices.speaker_interface import SpeakerInterface
    
    # Flag to track if hardware is available
    HARDWARE_AVAILABLE = True
    print(f"Successfully imported hardware interfaces from: {sim_env_path}")
except ImportError as e:
    try:
        # Try alternative import path (directly from Kinova_gen2)
        from Kinova_gen2.src.devices.camera_interface import CameraInterface, MultiCameraInterface
        from Kinova_gen2.src.devices.kinova_arm_interface import KinovaArmInterface
        from Kinova_gen2.src.devices.speaker_interface import SpeakerInterface
        
        HARDWARE_AVAILABLE = True
        print(f"Successfully imported hardware interfaces using alternative path")
    except ImportError as e2:
        # If hardware interfaces are not available, use mock classes for testing
        logging.warning(f"Hardware interfaces not found: {e2}. Using mock classes.")
        HARDWARE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define mock classes for testing without hardware
class MockCameraInterface:
    """Mock implementation of CameraInterface for testing."""
    
    def __init__(self, camera_id=0, width=320, height=240, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        logger.info(f"Initialized mock camera {camera_id} with resolution {width}x{height}")
    
    def capture_frame(self):
        """Return a mock frame (gray image with text)."""
        # Create a simple gray image with text for testing
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (self.width, self.height), color=(200, 200, 200))
        d = ImageDraw.Draw(img)
        d.text((self.width//2-50, self.height//2), f"Mock Camera {self.camera_id}", fill=(0, 0, 0))
        
        # Convert to numpy array (simulating OpenCV format)
        frame = np.array(img)
        return True, frame
    
    def close(self):
        logger.info(f"Closed mock camera {self.camera_id}")

class MockMultiCameraInterface:
    """Mock implementation of MultiCameraInterface for testing."""
    
    def __init__(self, camera_ids=None, width=320, height=240, fps=30):
        self.cameras = {}
        camera_ids = camera_ids or [0, 1]  # Default to two cameras: environment and wrist
        
        for cam_id in camera_ids:
            self.cameras[cam_id] = MockCameraInterface(camera_id=cam_id, width=width, height=height, fps=fps)
        
        logger.info(f"Initialized {len(self.cameras)} mock cameras")
    
    def capture_frames(self):
        """Capture frames from all mock cameras."""
        frames = {}
        for cam_id, camera in self.cameras.items():
            frames[cam_id] = camera.capture_frame()
        return frames
    
    def close(self):
        for camera in self.cameras.values():
            camera.close()
        logger.info("All mock cameras released")

class MockKinovaArmInterface:
    """Mock implementation of KinovaArmInterface for testing."""
    
    def __init__(self):
        self.current_position = [0.25, 0.0, 0.4, 1.5, 0.5, 0.0]  # Default position [x,y,z,theta_x,theta_y,theta_z]
        self.current_fingers = [0.0, 0.0, 0.0]  # Default finger positions (open)
        logger.info("Initialized mock Kinova arm")
    
    def connect(self):
        logger.info("Connected to mock Kinova arm")
        return True
    
    def move_home(self):
        self.current_position = [0.25, 0.0, 0.4, 1.5, 0.5, 0.0]
        self.current_fingers = [0.0, 0.0, 0.0]
        logger.info("Mock arm: Moving to home position")
        return True
    
    def send_angular_position(self, joint_angles, fingers=(0.0, 0.0, 0.0), speed_factor=0.3):
        logger.info(f"Mock arm: Moving to joint angles {joint_angles}")
        return True
    
    def send_cartesian_position(self, position, rotation, fingers=(0.0, 0.0, 0.0), duration=5.0, period=0.05):
        self.current_position = list(position) + list(rotation)
        self.current_fingers = list(fingers)
        logger.info(f"Mock arm: Moving to position {position}, rotation {rotation}")
        return True
    
    def get_cartesian_position(self):
        return self.current_position + [self.current_fingers[0]]
    
    def close(self):
        logger.info("Closed connection to mock Kinova arm")

class MockSpeakerInterface:
    """Mock implementation of SpeakerInterface for testing."""
    
    def __init__(self, voice="Samantha", volume=1.0, rate=1.0, pitch=1.0):
        self.voice = voice
        self.volume = volume
        self.rate = rate
        self.pitch = pitch
        self._is_playing = False
        logger.info(f"Initialized mock speaker with voice {voice}")
    
    def speak(self, text, wait=True):
        self._is_playing = True
        logger.info(f"Mock speaker: '{text}'")
        
        if wait:
            # Simulate speaking time based on text length
            time.sleep(len(text) * 0.05)  # ~200 chars per second
            self._is_playing = False
        else:
            # Start a thread to simulate async speaking
            def _simulate_speaking():
                time.sleep(len(text) * 0.05)
                self._is_playing = False
            
            threading.Thread(target=_simulate_speaking, daemon=True).start()
        
        return True
    
    def play_audio(self, audio_file, wait=True):
        self._is_playing = True
        logger.info(f"Mock speaker: Playing audio file '{audio_file}'")
        
        if wait:
            time.sleep(2.0)  # Simulate 2 seconds of audio
            self._is_playing = False
        else:
            # Start a thread to simulate async audio playback
            def _simulate_playback():
                time.sleep(2.0)
                self._is_playing = False
            
            threading.Thread(target=_simulate_playback, daemon=True).start()
        
        return True
    
    def is_playing(self):
        return self._is_playing
    
    def stop(self):
        self._is_playing = False
        logger.info("Mock speaker: Stopped playback")
        return True


# Camera Tools
class CameraTools:
    """Tools for interacting with cameras."""
    
    def __init__(self, use_mock=not HARDWARE_AVAILABLE, width=320 , height=240, fps=30):
        """
        Initialize camera tools.
        
        Args:
            use_mock: Whether to use mock interfaces (for testing without hardware)
            width: Camera width resolution (default: 320)
            height: Camera height resolution (default: 240)
            fps: Camera frames per second (default: 30)
        """
        self.use_mock = use_mock
        self.cameras = None
        self.env_camera_id = 0  # ID for the environment camera
        self.wrist_camera_id = 1  # ID for the wrist-mounted camera
        self.width = width
        self.height = height
        self.fps = fps
        self._initialize_cameras()
    
    def _initialize_cameras(self):
        """Initialize camera interfaces."""
        try:
            if self.use_mock:
                self.cameras = MockMultiCameraInterface(camera_ids=[self.env_camera_id, self.wrist_camera_id], 
                                                       width=self.width, height=self.height, fps=self.fps)
                logger.info(f"Initialized mock cameras with resolution {self.width}x{self.height}")
            else:
                # Detect available cameras
                available_cameras = CameraInterface.list_available_cameras()
                logger.info(f"Available cameras: {available_cameras}")
                
                if len(available_cameras) >= 2:
                    # We have at least two cameras (environment and wrist)
                    self.env_camera_id = available_cameras[0]
                    self.wrist_camera_id = available_cameras[1]
                elif len(available_cameras) == 1:
                    # Only one camera available, use it for environment
                    self.env_camera_id = available_cameras[0]
                    self.wrist_camera_id = None
                    logger.warning("Only one camera found. Using it as environment camera.")
                else:
                    logger.error("No cameras found. Using mock cameras instead.")
                    self.use_mock = True
                    self.cameras = MockMultiCameraInterface(width=self.width, height=self.height, fps=self.fps)
                    return
                
                # Initialize the cameras with specified resolution
                self.cameras = MultiCameraInterface(camera_ids=[cam_id for cam_id in available_cameras], 
                                                  width=self.width, height=self.height, fps=self.fps)
                logger.info(f"Initialized {len(available_cameras)} real cameras with resolution {self.width}x{self.height}")
        
        except Exception as e:
            logger.error(f"Error initializing cameras: {e}")
            self.use_mock = True
            self.cameras = MockMultiCameraInterface(width=self.width, height=self.height, fps=self.fps)
    
    def _encode_image_to_base64(self, frame):
        """
        Encode a frame as a base64 string for LLM processing.
        
        Args:
            frame: The image frame (numpy array)
            
        Returns:
            A base64-encoded data URI
        """
        try:
            # Convert the frame to a PIL Image
            pil_img = Image.fromarray(frame)
            
            # Save the image to a bytes buffer
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            
            # Get the bytes and encode as base64
            encoded_image = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            
            # Format as a data URI
            return f"data:image/jpeg;base64,{encoded_image}"
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return None
    
    def capture_environment(self) -> str:
        """
        Capture an image from the environment camera and return as base64-encoded data URI.
        
        Returns:
            A base64-encoded image data URI or an error message
        """
        if not self.cameras:
            return "Error: Cameras not initialized"
        
        try:
            frames = self.cameras.capture_frames()
            if self.env_camera_id in frames:
                success, frame = frames[self.env_camera_id]
                if success:
                    # Encode the frame as base64
                    encoded_image = self._encode_image_to_base64(frame)
                    if encoded_image:
                        # Return the encoded image with dimensions
                        height, width = frame.shape[:2]
                        return {
                            "image": encoded_image,
                            "description": f"Environment camera image ({width}x{height})"
                        }
            
            return "Failed to capture image from environment camera"
            
        except Exception as e:
            logger.error(f"Error capturing from environment camera: {e}")
            return f"Error capturing image: {str(e)}"
    
    def capture_wrist(self) -> str:
        """
        Capture an image from the wrist-mounted camera and return as base64-encoded data URI.
        
        Returns:
            A base64-encoded image data URI or an error message
        """
        if not self.cameras:
            return "Error: Cameras not initialized"
        
        if self.wrist_camera_id is None:
            return "Error: Wrist camera not available"
        
        try:
            frames = self.cameras.capture_frames()
            if self.wrist_camera_id in frames:
                success, frame = frames[self.wrist_camera_id]
                if success:
                    # Encode the frame as base64
                    encoded_image = self._encode_image_to_base64(frame)
                    if encoded_image:
                        # Return the encoded image with dimensions
                        height, width = frame.shape[:2]
                        return {
                            "image": encoded_image,
                            "description": f"Wrist camera image ({width}x{height})"
                        }
            
            return "Failed to capture image from wrist camera"
            
        except Exception as e:
            logger.error(f"Error capturing from wrist camera: {e}")
            return f"Error capturing image: {str(e)}"
    
    def capture_both(self) -> str:
        """
        Capture images from both environment and wrist cameras simultaneously.
        
        Returns:
            Dictionary containing both camera images as base64-encoded data URIs
        """
        if not self.cameras:
            return "Error: Cameras not initialized"
        
        try:
            frames = self.cameras.capture_frames()
            result = {"environment": None, "wrist": None}
            
            # Process environment camera
            if self.env_camera_id in frames:
                success, frame = frames[self.env_camera_id]
                if success:
                    encoded_image = self._encode_image_to_base64(frame)
                    if encoded_image:
                        height, width = frame.shape[:2]
                        result["environment"] = {
                            "image": encoded_image,
                            "description": f"Environment camera image ({width}x{height})"
                        }
            
            # Process wrist camera
            if self.wrist_camera_id in frames and self.wrist_camera_id is not None:
                success, frame = frames[self.wrist_camera_id]
                if success:
                    encoded_image = self._encode_image_to_base64(frame)
                    if encoded_image:
                        height, width = frame.shape[:2]
                        result["wrist"] = {
                            "image": encoded_image,
                            "description": f"Wrist camera image ({width}x{height})"
                        }
            
            # Check if we got at least one camera
            if result["environment"] is None and result["wrist"] is None:
                return "Failed to capture images from any camera"
                
            return result
            
        except Exception as e:
            logger.error(f"Error capturing from cameras: {e}")
            return f"Error capturing images: {str(e)}"
    
    def close(self):
        """Close all camera connections."""
        if self.cameras:
            self.cameras.close()
            self.cameras = None
            logger.info("All cameras closed")


# Speaker Tools
class SpeakerTools:
    """Tools for text-to-speech and audio playback with cross-platform support."""
    
    def __init__(self, use_mock=not HARDWARE_AVAILABLE):
        """
        Initialize speaker tools.
        
        Args:
            use_mock: Whether to use mock interfaces (for testing without hardware)
        """
        self.use_mock = use_mock
        self.speaker = None
        self._platform = self._detect_platform()
        self._tts_cmd = self._get_tts_command()
        self._audio_cmd = self._get_audio_command()
        self._initialize_speaker()
        self._is_speaking = False
        self._current_process = None
    
    def _detect_platform(self):
        """Detect the operating system platform."""
        import platform
        system = platform.system().lower()
        if system == 'darwin':
            return 'macos'
        elif system == 'linux':
            return 'linux'
        elif system == 'windows':
            return 'windows'
        else:
            return 'unknown'
    
    def _get_tts_command(self):
        """Get the appropriate text-to-speech command for the platform."""
        if self._platform == 'macos':
            return 'say'
        elif self._platform == 'linux':
            # Check for available TTS engines
            import shutil
            if shutil.which('espeak'):
                return 'espeak'
            elif shutil.which('festival'):
                return 'festival --tts'
            elif shutil.which('pico2wave'):
                return 'pico2wave'
            else:
                logger.warning("No TTS engine found on Linux. Speech functionality will be unavailable.")
                return None
        else:
            return None
    
    def _get_audio_command(self):
        """Get the appropriate audio playback command for the platform."""
        if self._platform == 'macos':
            return 'afplay'
        elif self._platform == 'linux':
            import shutil
            if shutil.which('aplay'):
                return 'aplay'
            elif shutil.which('play'):
                return 'play'
            else:
                logger.warning("No audio player found on Linux. Audio playback will be unavailable.")
                return None
        else:
            return None
    
    def _initialize_speaker(self):
        """Initialize speaker interface."""
        try:
            if self.use_mock:
                self.speaker = MockSpeakerInterface()
                logger.info("Initialized mock speaker")
            else:
                # Check if we have TTS available
                if self._tts_cmd:
                    logger.info(f"Initialized real speaker using {self._platform} TTS command: {self._tts_cmd}")
                    self.speaker = True  # Just a placeholder, we'll use our own implementation
                else:
                    logger.warning(f"TTS not available on {self._platform}. Using mock speaker.")
                    self.use_mock = True
                    self.speaker = MockSpeakerInterface()
        except Exception as e:
            logger.error(f"Error initializing speaker: {e}")
            self.use_mock = True
            self.speaker = MockSpeakerInterface()
    
    def speak(self, text: str) -> str:
        """
        Convert text to speech.
        
        Args:
            text: The text to speak
            
        Returns:
            A string indicating success or failure
        """
        if self.use_mock:
            self.speaker.speak(text, wait=False)
            return f"Speaking: '{text}'"
        
        try:
            # Clean up any current process
            self._cleanup_current_process()
            self._is_speaking = True
            
            # Use platform-specific TTS
            if self._platform == 'macos':
                # Use macOS 'say' command
                def _run_async():
                    try:
                        self._current_process = subprocess.Popen(['say', text], 
                                                                stdout=subprocess.DEVNULL,
                                                                stderr=subprocess.DEVNULL)
                        self._current_process.wait()
                    except Exception as e:
                        logger.error(f"Error in macOS speech: {e}")
                    finally:
                        self._is_speaking = False
                        self._current_process = None
                
                thread = threading.Thread(target=_run_async, daemon=True)
                thread.start()
                return f"Speaking: '{text}'"
                
            elif self._platform == 'linux':
                # Use Linux TTS commands
                if self._tts_cmd == 'espeak':
                    def _run_async():
                        try:
                            self._current_process = subprocess.Popen(['espeak', text], 
                                                                    stdout=subprocess.DEVNULL,
                                                                    stderr=subprocess.DEVNULL)
                            self._current_process.wait()
                        except Exception as e:
                            logger.error(f"Error in Linux speech: {e}")
                        finally:
                            self._is_speaking = False
                            self._current_process = None
                    
                    thread = threading.Thread(target=_run_async, daemon=True)
                    thread.start()
                    return f"Speaking: '{text}'"
                
                elif self._tts_cmd == 'festival --tts':
                    def _run_async():
                        try:
                            # Create a temp file for festival
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                                f.write(text)
                                temp_file = f.name
                            
                            self._current_process = subprocess.Popen(['festival', '--tts', temp_file], 
                                                                    stdout=subprocess.DEVNULL,
                                                                    stderr=subprocess.DEVNULL)
                            self._current_process.wait()
                            
                            # Clean up temp file
                            os.unlink(temp_file)
                        except Exception as e:
                            logger.error(f"Error in Linux speech: {e}")
                        finally:
                            self._is_speaking = False
                            self._current_process = None
                    
                    thread = threading.Thread(target=_run_async, daemon=True)
                    thread.start()
                    return f"Speaking: '{text}'"
                
                elif self._tts_cmd == 'pico2wave':
                    def _run_async():
                        try:
                            # Create temp files for pico2wave
                            import tempfile
                            wav_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                            
                            # Generate speech to WAV file
                            subprocess.run(['pico2wave', '-w', wav_file, text], 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL)
                            
                            # Play the WAV file
                            if self._audio_cmd == 'aplay':
                                self._current_process = subprocess.Popen(['aplay', wav_file], 
                                                                        stdout=subprocess.DEVNULL,
                                                                        stderr=subprocess.DEVNULL)
                            else:
                                self._current_process = subprocess.Popen(['play', wav_file], 
                                                                        stdout=subprocess.DEVNULL,
                                                                        stderr=subprocess.DEVNULL)
                            
                            self._current_process.wait()
                            
                            # Clean up temp file
                            os.unlink(wav_file)
                        except Exception as e:
                            logger.error(f"Error in Linux speech: {e}")
                        finally:
                            self._is_speaking = False
                            self._current_process = None
                    
                    thread = threading.Thread(target=_run_async, daemon=True)
                    thread.start()
                    return f"Speaking: '{text}'"
            
            return f"Speaking: '{text}'"
                
        except Exception as e:
            logger.error(f"Error speaking text: {e}")
            return f"Error speaking text: {str(e)}"
    
    def _cleanup_current_process(self):
        """Clean up any current TTS process."""
        if self._current_process:
            try:
                self._current_process.terminate()
            except:
                pass
            self._current_process = None
            self._is_speaking = False
    
    def play_audio(self, audio_file: str) -> str:
        """
        Play an audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            A string indicating success or failure
        """
        if self.use_mock:
            self.speaker.play_audio(audio_file, wait=False)
            return f"Playing audio: '{audio_file}'"
        
        if not os.path.exists(audio_file):
            return f"Error: Audio file not found: '{audio_file}'"
        
        if not self._audio_cmd:
            return "Error: No audio player available for this platform"
        
        try:
            # Clean up any current process
            self._cleanup_current_process()
            self._is_speaking = True
            
            if self._platform == 'macos':
                def _run_async():
                    try:
                        self._current_process = subprocess.Popen(['afplay', audio_file], 
                                                                stdout=subprocess.DEVNULL,
                                                                stderr=subprocess.DEVNULL)
                        self._current_process.wait()
                    except Exception as e:
                        logger.error(f"Error playing audio: {e}")
                    finally:
                        self._is_speaking = False
                        self._current_process = None
                
                thread = threading.Thread(target=_run_async, daemon=True)
                thread.start()
                return f"Playing audio: '{audio_file}'"
                
            elif self._platform == 'linux':
                if self._audio_cmd == 'aplay':
                    def _run_async():
                        try:
                            self._current_process = subprocess.Popen(['aplay', audio_file], 
                                                                    stdout=subprocess.DEVNULL,
                                                                    stderr=subprocess.DEVNULL)
                            self._current_process.wait()
                        except Exception as e:
                            logger.error(f"Error playing audio: {e}")
                        finally:
                            self._is_speaking = False
                            self._current_process = None
                    
                    thread = threading.Thread(target=_run_async, daemon=True)
                    thread.start()
                    return f"Playing audio: '{audio_file}'"
                    
                elif self._audio_cmd == 'play':
                    def _run_async():
                        try:
                            self._current_process = subprocess.Popen(['play', audio_file], 
                                                                    stdout=subprocess.DEVNULL,
                                                                    stderr=subprocess.DEVNULL)
                            self._current_process.wait()
                        except Exception as e:
                            logger.error(f"Error playing audio: {e}")
                        finally:
                            self._is_speaking = False
                            self._current_process = None
                    
                    thread = threading.Thread(target=_run_async, daemon=True)
                    thread.start()
                    return f"Playing audio: '{audio_file}'"
            
            return f"Playing audio: '{audio_file}'"
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            return f"Error playing audio: {str(e)}"
    
    def is_speaking(self) -> str:
        """
        Check if the speaker is currently active.
        
        Returns:
            A string indicating whether speech is in progress
        """
        if self.use_mock:
            is_playing = self.speaker.is_playing()
        else:
            is_playing = self._is_speaking
            
        return f"Speaker is {'active' if is_playing else 'inactive'}"
    
    def stop_speaking(self) -> str:
        """
        Stop any current speech or audio playback.
        
        Returns:
            A string indicating success or failure
        """
        if self.use_mock:
            self.speaker.stop()
            return "Stopped speaker output"
        
        self._cleanup_current_process()
        return "Stopped speaker output"

    def close(self):
        """Close the speaker interface."""
        # Stop any ongoing speech
        self._cleanup_current_process()
        self.speaker = None
        logger.info("Speaker closed")


# Robotic Arm Tools
class RoboticArmTools:
    """Tools for controlling the Kinova robotic arm."""
    
    def __init__(self, use_mock=not HARDWARE_AVAILABLE):
        """
        Initialize robotic arm tools.
        
        Args:
            use_mock: Whether to use mock interfaces (for testing without hardware)
        """
        self.use_mock = use_mock
        self.arm = None
        self._initialize_arm()
    
    def _initialize_arm(self):
        """Initialize robotic arm interface."""
        try:
            if self.use_mock:
                self.arm = MockKinovaArmInterface()
                logger.info("Initialized mock robotic arm")
            else:
                self.arm = KinovaArmInterface()
                # Connect to the arm
                self.arm.connect()
                logger.info("Initialized and connected to real robotic arm")
        
        except Exception as e:
            logger.error(f"Error initializing robotic arm: {e}")
            self.use_mock = True
            self.arm = MockKinovaArmInterface()
    
    def move_home(self) -> str:
        """
        Move the arm to the home position.
        
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            self.arm.move_home()
            time.sleep(5)
            # Note: we don't wait for completion as per requirements
            return "Successfully moved to home position"
                
        except Exception as e:
            logger.error(f"Error moving to home position: {e}")
            return f"Error moving to home position: {str(e)}"
    
    def move_position(self, x: float, y: float, z: float, 
                     theta_x: float = None, theta_y: float = None, theta_z: float = None,fingers: float = None) -> str:
        """
        Move the arm to a specific cartesian position.
        
        Args:
            x, y, z: Cartesian coordinates in meters
            theta_x, theta_y, theta_z: Optional rotation angles
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            # Get current position for any missing rotation parameters
            current_pos = self.arm.get_cartesian_position()
            if not current_pos:
                return "Error: Could not get current position"
            
            # Use current rotation values if not specified
            if theta_x is None:
                theta_x = current_pos[3]
            if theta_y is None:
                theta_y = current_pos[4]
            if theta_z is None:
                theta_z = current_pos[5]
            if fingers is None:
                fingers = current_pos[6]
            
            # Send the position command
            position = (float(x), float(y), float(z))
            rotation = (float(theta_x), float(theta_y), float(theta_z))
            
            self.arm.send_cartesian_position(
                position=position,
                rotation=rotation,
                fingers=(float(fingers), float(fingers), float(fingers)),  # Default to open fingers
                duration=5.0  # Reasonable duration for movement
            )
            time.sleep(5)
            # Note: we don't wait for completion as per requirements
            return f"Successfully moved to position ({x}, {y}, {z}) with rotation ({theta_x}, {theta_y}, {theta_z}) and fingers {fingers}"
                
        except Exception as e:
            logger.error(f"Error moving to position: {e}")
            return f"Error moving to position: {str(e)}"
    
    def close_gripper(self, strength: float = 0.5) -> str:
        """
        Close the gripper to grasp an object.
        
        Args:
            strength: Grasping strength from 0.0 (open) to 1.0 (closed)
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            # Validate and scale the strength parameter
            strength = max(0.0, min(strength, 1.0))
            finger_velocity = strength * 3000.0  # Scale to finger position range (0-6000)
            
            # Send velocity command
            self.arm.send_cartesian_velocity(
                linear_velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                fingers=(finger_velocity, finger_velocity, finger_velocity),
                hand_mode=1,
                duration=2.0
            )
            #TODO: update the duration based the arm speed
            time.sleep(2)
            
            return f"Successfully grasped with strength {strength:.2f}"
                
        except Exception as e:
            logger.error(f"Error grasping: {e}")
            return f"Error grasping: {str(e)}"
    
    def open_gripper(self) -> str:
        """
        Open the gripper to release an object.
        
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            self.arm.send_cartesian_velocity(
                linear_velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                fingers=(-3000, -3000, -3000),
                hand_mode=1,
                duration=2.0
            )
            time.sleep(2)
            return "Successfully released"
                
        except Exception as e:
            logger.error(f"Error releasing: {e}")
            return f"Error releasing: {str(e)}"
    
    def get_position(self) -> str:
        """
        Get the current position of the robotic arm.
        
        Returns:
            A string describing the current position
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            current_pos = self.arm.get_cartesian_position()
            if not current_pos:
                return "Error: Could not get current position"
            
            # Format the position information
            position_str = f"Current position:\n"
            position_str += f"- Cartesian: ({current_pos[0]:.4f}, {current_pos[1]:.4f}, {current_pos[2]:.4f}) meters\n"
            position_str += f"- Rotation: ({current_pos[3]:.4f}, {current_pos[4]:.4f}, {current_pos[5]:.4f}) radians\n"
            position_str += f"- Gripper: {current_pos[6]:.1f} (0=open, 6000=closed)"
            
            return position_str
                
        except Exception as e:
            logger.error(f"Error getting position: {e}")
            return f"Error getting position: {str(e)}"
    
    def move_default(self) -> str:
        """
        Move the arm to a default position suitable for kitchen tasks.
        
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            success = self.arm.move_default(duration=5.0, monitor=False)
            
            # Note: we don't wait for completion as per requirements
            return "Command sent: Moving to default kitchen position"
                
        except Exception as e:
            logger.error(f"Error moving to default position: {e}")
            return f"Error moving to default position: {str(e)}"
    
    def turn_left(self, degree: float) -> str:
        """
        Turn the arm left around the Y axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=0.0, angular_y=degree, angular_z=0.0,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error turning left: {e}")
            return f"Error turning left: {str(e)}"
    
    def turn_right(self, degree: float) -> str:
        """
        Turn the arm right around the Y axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=0.0, angular_y=-degree, angular_z=0.0,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error turning right: {e}")
            return f"Error turning right: {str(e)}"
    
    def turn_down(self, degree: float) -> str:
        """
        Turn the arm down around the X axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=degree, angular_y=0.0, angular_z=0.0,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error turning down: {e}")
            return f"Error turning down: {str(e)}"
    
    def turn_up(self, degree: float) -> str:
        """
        Turn the arm up around the X axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=-degree, angular_y=0.0, angular_z=0.0,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error turning up: {e}")
            return f"Error turning up: {str(e)}"
    
    def rotate_left(self, degree: float) -> str:
        """
        Rotate the arm left around the Z axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=0.0, angular_y=0.0, angular_z=-degree,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error rotating left: {e}")
            return f"Error rotating left: {str(e)}"
    
    def rotate_right(self, degree: float) -> str:
        """
        Rotate the arm right around the Z axis.
        
        Args:
            degree: Angular velocity in degrees/second
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            return self.send_cartesian_velocity(
                linear_x=0.0, linear_y=0.0, linear_z=0.0,
                angular_x=0.0, angular_y=0.0, angular_z=degree,
                duration=1.0
            )
        except Exception as e:
            logger.error(f"Error rotating right: {e}")
            return f"Error rotating right: {str(e)}"
    
    def close(self):
        """Close the connection to the robotic arm."""
        if self.arm:
            self.arm.close()
            self.arm = None
            logger.info("Robotic arm connection closed")
    
    def send_cartesian_velocity(self, linear_x: float, linear_y: float, linear_z: float, 
                                angular_x: float, angular_y: float, angular_z: float,
                                fingers_velocity: float = 0.0, hand_mode: int = 1, duration: float = 1.0):
        """
        Send velocity commands to move the arm at specified speeds.
        
        Args:
            linear_x: Linear velocity in X direction (meters/second)
            linear_y: Linear velocity in Y direction (meters/second)
            linear_z: Linear velocity in Z direction (meters/second)
            angular_x: Angular velocity around X axis (radians/second)
            angular_y: Angular velocity around Y axis (radians/second)
            angular_z: Angular velocity around Z axis (radians/second)
            fingers_velocity: Gripper fingers velocity (default: 0.0)
            hand_mode: Hand mode (0-2) (default: 1)
            duration: Duration to apply velocity (seconds)
            
        Returns:
            A string indicating the command was sent
        """
        if not self.arm:
            return "Error: Robotic arm not initialized"
        
        try:
            # Create velocity tuples
            linear_velocity = [linear_x, linear_y, linear_z]
            angular_velocity = [angular_x, angular_y, angular_z]
            fingers = (fingers_velocity, fingers_velocity, fingers_velocity)
            
            # Send command to arm
            self.arm.send_cartesian_velocity(
                linear_velocity=linear_velocity, 
                angular_velocity=angular_velocity, 
                fingers=fingers, 
                hand_mode=hand_mode,
                duration=duration
            )
            
            return f"Successfully sent cartesian velocity command for {duration} seconds"
                
        except Exception as e:
            logger.error(f"Error sending velocity command: {e}")
            return f"Error sending velocity command: {str(e)}"

# Combined hardware tools class for easier management
class HardwareTools:
    """Management class for all hardware tools."""
    
    def __init__(self, use_mock=not HARDWARE_AVAILABLE, 
                 enable_camera=True, 
                 enable_speaker=True, 
                 enable_arm=True,
                 camera_width=320, 
                 camera_height=240, 
                 camera_fps=30):
        """
        Initialize all hardware tools.
        
        Args:
            use_mock: Whether to use mock interfaces (for testing without hardware)
            enable_camera: Whether to enable camera tools
            enable_speaker: Whether to enable speaker tools
            enable_arm: Whether to enable robotic arm tools
            camera_width: Width resolution for cameras (default: 320)
            camera_height: Height resolution for cameras (default: 240)
            camera_fps: Frames per second for cameras (default: 30)
        """
        self.use_mock = use_mock
        
        # Initialize each hardware component only if enabled
        if enable_camera:
            self.camera_tools = CameraTools(use_mock=use_mock, width=camera_width, height=camera_height, fps=camera_fps)
            logger.info(f"Camera tools initialized (using {'mock' if use_mock else 'real'} interfaces)")
            logger.info(f"Camera resolution set to {camera_width}x{camera_height} @ {camera_fps}fps")
        else:
            self.camera_tools = None
            logger.info("Camera tools disabled")
        
        if enable_speaker:
            self.speaker_tools = SpeakerTools(use_mock=use_mock)
            logger.info(f"Speaker tools initialized (using {'mock' if use_mock else 'real'} interfaces)")
        else:
            self.speaker_tools = None
            logger.info("Speaker tools disabled")
        
        if enable_arm:
            self.arm_tools = RoboticArmTools(use_mock=use_mock)
            logger.info(f"Robotic arm tools initialized (using {'mock' if use_mock else 'real'} interfaces)")
        else:
            self.arm_tools = None
            logger.info("Robotic arm tools disabled")
    
    def close(self):
        """Close all hardware connections."""
        if self.camera_tools:
            self.camera_tools.close()
        
        if self.speaker_tools:
            self.speaker_tools.close()
        
        if self.arm_tools:
            self.arm_tools.close()
            
        logger.info("All hardware connections closed")


# Example usage
def main():
    """Test the hardware tools."""
    # Create hardware tools with mock interfaces
    hardware = HardwareTools(use_mock=True)
    
    try:
        # Test camera tools
        print("\n=== Testing Camera Tools ===")
        print(hardware.camera_tools.capture_environment())
        print(hardware.camera_tools.capture_wrist())
        print(hardware.camera_tools.capture_both())
        
        # Test speaker tools
        print("\n=== Testing Speaker Tools ===")
        print(hardware.speaker_tools.speak("Hello, I am the kitchen assistant robot."))
        print(hardware.speaker_tools.is_speaking())
        time.sleep(1)
        
        # Test robotic arm tools
        print("\n=== Testing Robotic Arm Tools ===")
        print(hardware.arm_tools.get_position())
        print(hardware.arm_tools.move_home())
        print(hardware.arm_tools.move_position(0.3, 0.1, 0.4))
        print(hardware.arm_tools.close_gripper(0.7))
        print(hardware.arm_tools.get_position())
        print(hardware.arm_tools.open_gripper())
        
    finally:
        # Clean up
        hardware.close()


if __name__ == "__main__":
    main() 