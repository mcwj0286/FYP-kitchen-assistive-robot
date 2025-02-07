from devices.camera_interface import main as camera_main
from robot_controller import main as robot_main
import signal
import sys
import threading

def main():
    # Flag to control thread execution
    running = True
    
    def signal_handler(sig, frame):
        nonlocal running
        print("\nSignal received, cleaning up...")
        running = False
        
    # Set up signal handler in main thread
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTSTP, signal_handler)

    # Create threads for camera and robot
    camera_thread = threading.Thread(target=camera_main)
    robot_thread = threading.Thread(target=robot_main)

    # Start both threads
    camera_thread.daemon = True  # Make threads daemon so they exit when main thread exits
    robot_thread.daemon = True
    camera_thread.start() 
    robot_thread.start()

    # Keep main thread running until signal is received
    try:
        while running:
            signal.pause()  # Wait for signals
    except KeyboardInterrupt:
        running = False
    finally:
        print("Shutting down threads...")
        # Give threads a chance to cleanup
        camera_thread.join(timeout=2)
        robot_thread.join(timeout=2)
        print("Cleanup complete")

if __name__ == "__main__":
    main()