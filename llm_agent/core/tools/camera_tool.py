import os
import time
from typing import Dict, Optional, Any

from ..base_agent import Tool

class CameraTool(Tool):
    """
    A tool for capturing and processing images from cameras.
    """
    def __init__(
        self, 
        name: str = "camera",
        description: str = "Captures and processes images from cameras",
        camera_interface = None,
        height: int = 240,
        width: int = 320
    ):
        """
        Initialize the camera tool.
        
        Args:
            name: The name of the tool.
            description: A description of the tool.
            camera_interface: Optional camera interface to use for capturing frames.
            height: The height of the captured images.
            width: The width of the captured images.
        """
        super().__init__(name, description)
        self.camera_interface = camera_interface
        self.height = height
        self.width = width
        
        # Initialize the camera interface if not provided
        if self.camera_interface is None:
            try:
                # Try to import the MultiCameraInterface
                from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
                self.camera_interface = MultiCameraInterface(height=height, width=width)
                time.sleep(3)  # Wait for cameras to initialize
                print("Camera interface initialized successfully.")
            except ImportError:
                print("Warning: MultiCameraInterface not found. No images will be captured.")
                self.camera_interface = None
    
    def execute(self, action: str = "capture", *args, **kwargs) -> Any:
        """
        Execute the camera tool functionality.
        
        Args:
            action: The action to perform ("capture", "process", or "capture_and_process").
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Any: The result of the action.
        """
        if action == "capture":
            return self.capture_frames()
        elif action == "process":
            frames = kwargs.get("frames")
            if not frames:
                raise ValueError("Frames must be provided for processing")
            return self.process_frames(frames)
        elif action == "capture_and_process":
            frames = self.capture_frames()
            return self.process_frames(frames)
        else:
            raise ValueError(f"Unknown action: {action}")
    
    def capture_frames(self) -> Dict[str, Any]:
        """
        Capture frames from cameras.
        
        Returns:
            Dict[str, Any]: Dictionary of camera IDs to frame data.
        """
        if self.camera_interface is None:
            print("No camera interface available. Cannot capture images.")
            return {}
        
        try:
            # Capture frames from cameras
            frames = self.camera_interface.capture_frames()
            return frames
        except Exception as e:
            print(f"Error capturing images: {e}")
            return {}
    
    def process_frames(self, frames: Dict[str, Any]) -> Dict[str, str]:
        """
        Process frames into image files.
        
        Args:
            frames: Dictionary of camera IDs to frame data.
            
        Returns:
            Dict[str, str]: Dictionary of camera IDs to image file paths.
        """
        try:
            # Import save_images from get_prompt module
            from ...get_prompt import save_images
            
            # Save the captured frames and return their paths
            return save_images(frames)
        except ImportError:
            # Fallback implementation if save_images is not available
            print("Warning: save_images function not found. Using fallback implementation.")
            
            import cv2
            import os
            import tempfile
            
            image_paths = {}
            for cam_id, frame in frames.items():
                # Create a temporary file for the image
                fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                os.close(fd)
                
                # Save the frame as an image
                cv2.imwrite(temp_path, frame)
                
                # Add the path to the dictionary
                image_paths[cam_id] = temp_path
            
            return image_paths
    
    def close(self) -> None:
        """
        Close the camera interface.
        """
        if self.camera_interface and hasattr(self.camera_interface, "close"):
            try:
                self.camera_interface.close()
                print("Camera interface closed successfully.")
            except Exception as e:
                print(f"Error closing camera interface: {e}") 