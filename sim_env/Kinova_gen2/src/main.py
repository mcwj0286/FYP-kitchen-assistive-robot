from devices.ps4controller_interface import PS4Interface

def main():
    # Initialize the PS4 controller
    # Note: The interface path might need to be adjusted based on your system
    # Common paths are:
    # - /dev/input/js0
    # - /dev/input/js1
    # You can check available devices using: ls /dev/input/js*
    
    try:
        controller = PS4Interface(interface="/dev/input/js0", connecting_using_ds4drv=False)
        print("PS4 Controller initialized. Press buttons to see output.")
        print("Press Ctrl+C to exit.")
        
        # Start listening for controller inputs
        controller.listen(timeout=60)  # 60 second timeout, adjust as needed
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your PS4 controller is connected and the correct interface path is specified.")
        print("You can check available devices using: ls /dev/input/js*")

if __name__ == "__main__":
    main() 