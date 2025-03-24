#!/usr/bin/env python3
"""
Direct speaker test for Kitchen Assistant Robot.
This script directly tests the macOS speaker functionality using the 'say' command.
"""

import os
import sys
import subprocess
import threading
import time
import logging
import platform
from typing import Optional

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

def speak_sync(text: str) -> bool:
    """
    Speak text synchronously using the 'say' command on macOS.
    
    Args:
        text: The text to speak
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if platform.system() != 'Darwin':
            print(f"This test is designed for macOS, but detected {platform.system()} platform.")
            return False
            
        print(f"Speaking (sync): {text}")
        result = subprocess.run(['say', text], check=False)
        print(f"Command completed with return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error in sync speech: {e}")
        return False

def speak_async(text: str) -> threading.Thread:
    """
    Speak text asynchronously using the 'say' command on macOS.
    
    Args:
        text: The text to speak
        
    Returns:
        threading.Thread: The thread running the speech command
    """
    def _speak_thread():
        try:
            if platform.system() != 'Darwin':
                print(f"This test is designed for macOS, but detected {platform.system()} platform.")
                return
                
            print(f"Speaking (async): {text}")
            result = subprocess.run(['say', text], check=False)
            print(f"Async command completed with return code: {result.returncode}")
        except Exception as e:
            logger.error(f"Error in async speech: {e}")
    
    thread = threading.Thread(target=_speak_thread)
    thread.daemon = True
    thread.start()
    return thread

def main():
    """Run the direct speaker tests."""
    print("\n===== Direct macOS Speaker Test =====")
    
    # Test 1: Synchronous speech
    print("\nTest 1: Synchronous speech")
    success = speak_sync("Hello, this is a direct test of the speaker.")
    print(f"Synchronous speech {'succeeded' if success else 'failed'}")
    
    # Test 2: Asynchronous speech
    print("\nTest 2: Asynchronous speech")
    print("Starting asynchronous speech in 2 seconds...")
    time.sleep(2)
    thread = speak_async("This is an asynchronous test of the speaker.")
    
    print("Waiting for asynchronous speech to complete...")
    thread.join(timeout=10)
    print("Asynchronous speech test complete")
    
    print("\nAll direct speaker tests completed.")

if __name__ == "__main__":
    main() 