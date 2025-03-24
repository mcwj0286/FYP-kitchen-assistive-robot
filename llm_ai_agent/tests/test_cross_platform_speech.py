#!/usr/bin/env python3
"""
Cross-platform speech test for Kitchen Assistant Robot.
This script tests the platform-aware speech functionality.
"""

import os
import sys
import platform
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

# Import the SpeakerTools class
try:
    from llm_ai_agent.hardware_tools import SpeakerTools
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(f"Error: Could not import SpeakerTools. Make sure you're running this from the project root directory.\nPaths: {sys.path}")

def print_header(title: str):
    """Print a header for a test section."""
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

def get_platform_info():
    """Get detailed information about the platform."""
    system = platform.system()
    release = platform.release()
    version = platform.version()
    architecture = platform.machine()
    
    print(f"Operating System: {system}")
    print(f"Release: {release}")
    print(f"Version: {version}")
    print(f"Architecture: {architecture}")
    
    # Check for TTS commands on the system
    print("\nTTS commands availability:")
    for cmd in ['say', 'espeak', 'festival', 'pico2wave']:
        if os.system(f"which {cmd} > /dev/null 2>&1") == 0:
            print(f"✅ {cmd} - Available")
        else:
            print(f"❌ {cmd} - Not available")
    
    # Check for audio players
    print("\nAudio players availability:")
    for cmd in ['afplay', 'aplay', 'play']:
        if os.system(f"which {cmd} > /dev/null 2>&1") == 0:
            print(f"✅ {cmd} - Available")
        else:
            print(f"❌ {cmd} - Not available")

def test_basic_speech(use_mock=False):
    """Test basic speech functionality."""
    print_header("BASIC SPEECH TEST")
    
    speaker = SpeakerTools(use_mock=use_mock)
    print(f"Using {'mock' if use_mock else 'real'} speaker")
    print(f"Detected platform: {speaker._platform}")
    print(f"TTS command: {speaker._tts_cmd}")
    print(f"Audio command: {speaker._audio_cmd}")
    
    # Short phrase
    print("\nTest 1: Speaking a short phrase")
    result = speaker.speak("Hello, I am testing cross-platform speech.")
    print(f"Result: {result}")
    time.sleep(3)  # Give it time to speak
    
    # Check status
    status = speaker.is_speaking()
    print(f"Status after speech: {status}")
    
    # Longer phrase
    print("\nTest 2: Speaking a longer phrase")
    result = speaker.speak("This is a longer phrase to test the cross-platform speech interface. It should demonstrate how the system handles longer utterances.")
    print(f"Result: {result}")
    
    # Check status immediately
    status = speaker.is_speaking()
    print(f"Status during speech: {status}")
    
    # Wait some time then check again
    time.sleep(2)
    status = speaker.is_speaking()
    print(f"Status after 2 seconds: {status}")
    
    # Test stopping speech
    print("\nTest 3: Stopping speech")
    result = speaker.speak("This is a very long phrase that should take some time to speak. We will attempt to stop it before it completes to test the stop functionality.")
    print(f"Result: {result}")
    time.sleep(1)  # Let it speak for a moment
    
    # Stop the speech
    stop_result = speaker.stop_speaking()
    print(f"Stop result: {stop_result}")
    
    print("\nBasic speech tests complete")
    speaker.close()

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='Test Cross-Platform Speech Interface')
    parser.add_argument('--mock', action='store_true', help='Use mock implementations')
    parser.add_argument('--info', action='store_true', help='Show platform information only')
    
    args = parser.parse_args()
    
    try:
        if args.info:
            print_header("PLATFORM INFORMATION")
            get_platform_info()
            return
            
        # Always show platform info first
        print_header("PLATFORM INFORMATION")
        get_platform_info()
        
        # Run speech tests
        test_basic_speech(use_mock=args.mock)
        
        print("\nAll tests completed.")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 