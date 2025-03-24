#!/usr/bin/env python3
"""
Hardware testing script for Kitchen Assistant Robot.
This script tests camera, speaker, and robotic arm functionality.
"""

import os
import sys
import argparse
import time
import logging
from typing import List, Dict, Any, Optional

# Set up proper import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import hardware tools
try:
    from llm_ai_agent.hardware_tools import CameraTools, SpeakerTools, RoboticArmTools
    HARDWARE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    print("Warning: Hardware tools import failed. Running in test mode only.")
    HARDWARE_AVAILABLE = False

def test_camera(use_mock=False, resolution=(640, 480)):
    """Test camera functionality."""
    print("\n===== Testing Camera =====")
    try:
        width, height = resolution
        print(f"Requesting camera with resolution {width}x{height}...")
        camera = CameraTools(use_mock=use_mock, width=width, height=height)
        
        # Display the configured resolution (from the CameraTools object)
        print(f"CameraTools configured with resolution {camera.width}x{camera.height}")
        
        print("\nTesting environment camera...")
        env_result = camera.capture_environment()
        print(f"Environment camera result: {env_result}")
        
        print("\nTesting wrist camera...")
        wrist_result = camera.capture_wrist()
        print(f"Wrist camera result: {wrist_result}")
        
        print("\nTesting image analysis...")
        analysis_result = camera.analyze_image("environment")
        print(f"Image analysis result: {analysis_result}")
        
        print("\nClosing camera...")
        camera.close()
        print("Camera closed")
        return True
    except Exception as e:
        logger.error(f"Error testing camera: {e}")
        print(f"Camera test failed: {e}")
        return False

def test_speaker(use_mock=False):
    """Test speaker functionality."""
    print("\n===== Testing Speaker =====")
    try:
        print("Initializing speaker...")
        speaker = SpeakerTools(use_mock=use_mock)
        print("Speaker initialized")
        
        print("Speaking a test phrase...")
        speaker.speak("This is a test of the speaker system.")
        print("Speech command sent")
        
        time.sleep(2)  # Give some time for speech
        
        print("Checking if still speaking...")
        is_speaking = speaker.is_speaking()
        print(f"Speaker status: {'Speaking' if is_speaking else 'Not speaking'}")
        
        print("Closing speaker...")
        speaker.close()
        print("Speaker closed")
        return True
    except Exception as e:
        logger.error(f"Error testing speaker: {e}")
        print(f"Speaker test failed: {e}")
        return False

def test_robotic_arm(use_mock=False):
    """Test robotic arm functionality."""
    print("\n===== Testing Robotic Arm =====")
    try:
        print("Initializing robotic arm...")
        arm = RoboticArmTools(use_mock=use_mock)
        print("Robotic arm initialized")
        
        print("Moving arm to home position...")
        result = arm.move_home()
        print(f"Home position result: {result}")
        
        print("Getting current position...")
        position = arm.get_position()
        print(f"Current position: {position}")
        
        print("Closing robotic arm...")
        arm.close()
        print("Robotic arm closed")
        return True
    except Exception as e:
        logger.error(f"Error testing robotic arm: {e}")
        print(f"Robotic arm test failed: {e}")
        return False

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Test Kitchen Robot Hardware')
    parser.add_argument('--mock', action='store_true', help='Use mock implementations')
    parser.add_argument('--camera', action='store_true', help='Test camera only')
    parser.add_argument('--speaker', action='store_true', help='Test speaker only')
    parser.add_argument('--arm', action='store_true', help='Test robotic arm only')
    parser.add_argument('--resolution', type=str, default='640,480', 
                      help='Camera resolution as width,height (e.g., 640,480)')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split(','))
        resolution = (width, height)
        print(f"Using requested resolution: {width}x{height}")
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default (640, 480).")
        resolution = (640, 480)
    
    # If no specific tests are selected, run all tests
    run_all = not (args.camera or args.speaker or args.arm)
    
    if not HARDWARE_AVAILABLE and not args.mock:
        print("Hardware tools are not available. Running with --mock option.")
        args.mock = True
    
    results = {}
    
    # Run selected tests
    if args.camera or run_all:
        results['camera'] = test_camera(use_mock=args.mock, resolution=resolution)
    
    if args.speaker or run_all:
        results['speaker'] = test_speaker(use_mock=args.mock)
    
    if args.arm or run_all:
        results['arm'] = test_robotic_arm(use_mock=args.mock)
    
    # Print summary
    print("\n===== Test Results =====")
    for component, result in results.items():
        print(f"{component.title()}: {'✅ PASS' if result else '❌ FAIL'}")

if __name__ == "__main__":
    main() 