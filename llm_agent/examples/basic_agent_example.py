import os
import sys
import time
import tempfile
from pathlib import Path

# Add the parent directory to the path to import the core modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_agent import BaseAgent, Tool
from core.tools.camera_tool import CameraTool
from core.parsers.base_parser import ActionParser

# Import OpenCV at the top level
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    print("OpenCV not available. Some functionality will be limited.")
    CV2_AVAILABLE = False

# Define the MockCameraTool outside of main to avoid scope issues
class MockCameraTool(Tool):
    def __init__(self, image_path):
        super().__init__("camera", "Mocked camera tool")
        self.image_path = image_path
        
    def execute(self, action, *args, **kwargs):
        if not CV2_AVAILABLE:
            return {}
            
        if action == "capture":
            # Return a dummy frame
            try:
                frame = cv2.imread(self.image_path)
                if frame is None:
                    print(f"Error: Could not read image from {self.image_path}")
                    return {}
                return {"test_camera": frame}
            except Exception as e:
                print(f"Error capturing mock frame: {e}")
                return {}
        elif action == "process":
            # Save the frame to a test file
            frames = kwargs.get("frames", {})
            image_paths = {}
            for cam_id, frame in frames.items():
                try:
                    # Create a temporary file for the image
                    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
                    os.close(fd)
                    
                    # Save the frame as an image
                    cv2.imwrite(temp_path, frame)
                    
                    # Add the path to the dictionary
                    image_paths[cam_id] = temp_path
                except Exception as e:
                    print(f"Error processing mock frame: {e}")
                    continue
            return image_paths
        else:
            print(f"Unknown action for MockCameraTool: {action}")
            return {}

class SimpleExampleAgent(BaseAgent):
    """
    A simple example agent that demonstrates how to use the BaseAgent class.
    """
    def __init__(
        self,
        name: str = "simple_example_agent",
        description: str = "A simple example agent",
        model_name: str = None,
        system_prompt: str = None,
        server_url: str = None,
    ):
        """
        Initialize the simple example agent.
        
        Args:
            name: The name of the agent.
            description: A description of the agent.
            model_name: The name of the LLM model to use.
            system_prompt: The system prompt to use for LLM interactions.
            server_url: URL of the server to upload images to.
        """
        super().__init__(
            name=name,
            description=description,
            model_name=model_name,
            system_prompt=system_prompt,
            server_url=server_url,
            enable_short_term_memory=True,
            enable_long_term_memory=False
        )
        
        # Register a camera tool
        self.register_tool(CameraTool())
        
        # Set a default system prompt if none is provided
        if not self.system_prompt:
            self.system_prompt = """
            You are a helpful assistant for a kitchen robot. Your role is to analyze images 
            of the kitchen environment and provide insights and suggestions.
            
            Please describe what you see in the image and suggest how a robot might assist.
            """
        
        # Create a parser for extracting structured information from LLM responses
        self.action_parser = ActionParser()

    def process(self, prompt: str, use_camera: bool = True):
        """
        Process a user prompt with optional image input.
        
        Args:
            prompt: The user prompt to process.
            use_camera: Whether to capture and use camera images.
            
        Returns:
            dict: The processing result.
        """
        self.logger.info(f"Processing prompt: {prompt}")
        
        # Capture images if requested
        image_paths = {}
        if use_camera:
            self.logger.info("Capturing images...")
            camera_tool = self.get_tool("camera")
            if camera_tool:
                frames = self.execute_tool("camera", "capture")
                if frames:
                    image_paths = self.execute_tool("camera", "process", frames=frames)
                    self.logger.info(f"Captured images: {list(image_paths.keys())}")
        
        # Upload images
        image_urls = {}
        if image_paths:
            self.logger.info("Uploading images...")
            image_urls = self.upload_images(image_paths)
            self.logger.info(f"Uploaded image URLs: {list(image_urls.keys())}")
        
        # Call LLM with prompt and images
        self.logger.info("Calling LLM...")
        llm_response = self.call_llm(prompt, image_urls)
        
        # Store the interaction in memory
        self.memory.store({
            "prompt": prompt,
            "image_paths": list(image_paths.keys()),
            "response": llm_response
        })
        
        # Return the result
        return {
            "prompt": prompt,
            "image_paths": image_paths,
            "image_urls": image_urls,
            "response": llm_response,
            "parsed_response": self.action_parser.parse(llm_response) if llm_response else None,
            "timestamp": time.time()
        }

def main():
    """
    Main function to demonstrate the simple example agent.
    """
    print("Initializing Simple Example Agent...")
    agent = SimpleExampleAgent()
    
    # Process a prompt without using the camera
    test_prompt = "What can a kitchen robot help with?"
    print(f"\nProcessing prompt: {test_prompt}")
    result = agent.process(test_prompt, use_camera=False)
    
    print(f"\nLLM Response: {result['response']}")
    
    # Create the directory for saving images if it doesn't exist
    os.makedirs("llm_agent/examples/test_images", exist_ok=True)
    
    # Create a test image (since we might not have a camera)
    test_image_path = "/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/llm_agent/assortment-delicious-healthy-food_23-2149043057.jpg"
    
    # Check if we have a test image or need to create a placeholder
    if not os.path.exists(test_image_path) and CV2_AVAILABLE:
        try:
            # Create a simple test image (a colored rectangle with text)
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:, :] = (200, 200, 100)  # Light blue-gray background
            
            # Add text
            cv2.putText(
                img, 
                "Test Kitchen Image", 
                (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 0), 
                2
            )
            
            # Save the image
            cv2.imwrite(test_image_path, img)
            print(f"Created a test image at {test_image_path}")
        except Exception as e:
            print(f"Error creating test image: {e}")
            test_image_path = None
    
    # Process a prompt with an image (if available)
    if os.path.exists(test_image_path):
        test_prompt_with_image = "What do you see in this kitchen image? How could a robot help?"
        print(f"\nProcessing prompt with image: {test_prompt_with_image}")
        
        # Register the mock camera tool with the test image path
        mock_camera = MockCameraTool(test_image_path)
        agent.register_tool(mock_camera)
        
        # Process with the mocked camera
        result_with_image = agent.process(test_prompt_with_image, use_camera=True)
        
        print(f"\nLLM Response with image: {result_with_image['response']}")
        
        # Check memory
        memory_entries = agent.memory.retrieve()
        print(f"\nMemory entries: {len(memory_entries)}")
        for i, entry in enumerate(memory_entries):
            print(f"Entry {i+1}: {entry['data']['prompt']}")
    else:
        print("No test image available. Skipping image processing example.")
    
    print("\nExample completed successfully.")

if __name__ == "__main__":
    main() 