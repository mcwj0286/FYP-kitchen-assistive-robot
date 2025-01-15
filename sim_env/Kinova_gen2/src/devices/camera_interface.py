#!/usr/bin/env python3

import cv2
import time
import signal
import sys

class CameraInterface:
    def __init__(self, camera_id=0, width=640, height=480, fps=30):
        """Initialize camera interface
        
        Args:
            camera_id (int): Camera device ID (default: 0)
            width (int): Image width (default: 640)
            height (int): Image height (default: 480)
            fps (int): Frames per second (default: 30)
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        
        # Initialize camera
        self.initialize_camera()

    @staticmethod
    def list_available_cameras(max_cameras=10):
        """List all available camera devices
        
        Args:
            max_cameras (int): Maximum number of cameras to check
            
        Returns:
            list: List of available camera indices
        """
        available_cameras = []
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available_cameras.append(i)
                cap.release()
            except Exception as e:
                print(f"Error checking camera {i}: {e}")
            finally:
                if 'cap' in locals():
                    cap.release()
        return available_cameras
        
    def initialize_camera(self):
        """Initialize the camera with specified parameters"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                raise Exception(f"Failed to open camera {self.camera_id}")
                
            # Read a test frame
            ret, frame = self.cap.read()
            if not ret:
                raise Exception(f"Failed to read from camera {self.camera_id}")
                
            print(f"Camera {self.camera_id} initialized successfully:")
            print(f"- Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"- FPS: {int(self.cap.get(cv2.CAP_PROP_FPS))}")
            
        except Exception as e:
            print(f"Error initializing camera {self.camera_id}: {e}")
            if self.cap is not None:
                self.cap.release()
            raise
            
    def capture_frame(self):
        """Capture a single frame
        
        Returns:
            tuple: (success, frame)
                - success (bool): Whether capture was successful
                - frame (numpy.ndarray): The captured frame
        """
        if not self.cap:
            return False, None
            
        ret, frame = self.cap.read()
        return ret, frame
        
    def close(self):
        """Release the camera"""
        if self.cap:
            self.cap.release()
            self.cap = None
            print(f"Camera {self.camera_id} released")

class MultiCameraInterface:
    def __init__(self, camera_ids=None, width=128, height=128, fps=30):
        """Initialize multiple cameras
        
        Args:
            camera_ids (list): List of camera IDs to initialize. If None, use all available cameras
            width (int): Image width for all cameras
            height (int): Image height for all cameras
            fps (int): FPS for all cameras
        """
        # If no camera IDs specified, find all available cameras
        if camera_ids is None:
            camera_ids = CameraInterface.list_available_cameras()
            
        if not camera_ids:
            raise Exception("No cameras found")
            
        self.cameras = {}
        for cam_id in camera_ids:
            try:
                self.cameras[cam_id] = CameraInterface(camera_id=cam_id, width=width, height=height, fps=fps)
            except Exception as e:
                print(f"Failed to initialize camera {cam_id}: {e}")
                
        if not self.cameras:
            raise Exception("Failed to initialize any cameras")
            
        print(f"Initialized {len(self.cameras)} cameras")
        
    def capture_frames(self):
        """Capture frames from all cameras
        
        Returns:
            dict: Dictionary mapping camera IDs to (success, frame) tuples
        """
        frames = {}
        for cam_id, camera in self.cameras.items():
            frames[cam_id] = camera.capture_frame()
        return frames
        
    def close(self):
        """Release all cameras"""
        for camera in self.cameras.values():
            camera.close()
        self.cameras.clear()

def main():
    """Test the camera interfaces"""
    def signal_handler(sig, frame):
        print("\nSignal received, cleaning up...")
        if 'cameras' in locals():
            cameras.close()
        cv2.destroyAllWindows()
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTSTP, signal_handler)  # Ctrl+Z

    try:
        # List available cameras
        available_cameras = CameraInterface.list_available_cameras()
        print(f"Available cameras: {available_cameras}")
        
        if not available_cameras:
            print("No cameras found!")
            return
            
        # Initialize multi-camera interface
        cameras = MultiCameraInterface(camera_ids=available_cameras)
        
        # Capture and display frames from all cameras
        print("\nCapturing frames (press 'q' to stop)...")
        while True:
            frames = cameras.capture_frames()
            
            # Display frames from all cameras
            for cam_id, (success, frame) in frames.items():
                if success:
                    cv2.imshow(f'Camera {cam_id}', frame)
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            time.sleep(1/30)  # Limit to ~30 FPS
            
    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        if 'cameras' in locals():
            cameras.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
