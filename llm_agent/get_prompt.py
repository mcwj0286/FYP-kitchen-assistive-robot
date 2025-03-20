#%% NEW: Capture images from multiple cameras, upload to Cloudinary, and send to LLM
from dotenv import load_dotenv

load_dotenv()  # Load .env variables

import os
import base64
import json
import requests
workspace_path = os.getenv("WORKSPACE_PATH")
model_name = os.getenv("MODEL_NAME")
system_prompt = os.getenv("SYSTEM_PROMPT")
import sys
sys.path.append(workspace_path)  # Add workspace path to Python path

import cv2
import time
# Import the MultiCameraInterface from your camera interface module.
# from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
def upload_images_to_cloudinary(captured_paths):
    import cloudinary
    import cloudinary.uploader
    from cloudinary.utils import cloudinary_url
    cloudinary.config( 
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"), 
        api_key=os.getenv("CLOUDINARY_API_KEY"), 
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True
    )
    uploaded_urls = {}
    for cam_id, file_path in captured_paths.items():
        upload_result = cloudinary.uploader.upload(file_path, public_id=f"my_image_cam_{cam_id}", format="jpg")
        url = upload_result["secure_url"]
        uploaded_urls[cam_id] = url
        print(f"Uploaded image URL for camera {cam_id}:", url)
    return uploaded_urls

def upload_image_to_server(server_url, image_path):
    """
    Upload an image to a server via HTTP POST request.
    
    Args:
        server_url (str): The URL of the server to upload to (e.g., "http://localhost:8000/images/")
        image_path (str): Path to the image file to upload
        
    Returns:
        str: URL of the uploaded image if successful, None otherwise
    """
    import requests
    import os
    import mimetypes
    
    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Ensure URL has trailing slash if needed
    if not server_url.endswith('/'):
        server_url = server_url + '/'
    
    # Get the filename and determine the content type
    filename = os.path.basename(image_path)
    content_type, _ = mimetypes.guess_type(image_path)
    if not content_type:
        # Default to jpeg if we can't determine the type
        content_type = 'image/jpeg'
    
    try:
        # Open the file in binary mode
        with open(image_path, 'rb') as img_file:
            # Create multipart form with filename and content-type
            files = {
                'file': (filename, img_file, content_type)
            }
            
            # Set the accept header
            headers = {'Accept': 'application/json'}
            
            # Send the POST request
            response = requests.post(server_url, files=files, headers=headers)
            
            # Handle the response
            if response.status_code in (200, 201):
                try:
                    data = response.json()
                    # Return just the URL
                    if 'url' in data:
                        return data['url']
                    return None
                except:
                    return None
            else:
                return None
                
    except Exception:
        return None
    
def llm_api_call(user_prompt: str, system_prompt: str, image_path: str, max_tokens: int = 256, debug=False) -> str:
    """
    Calls the OpenRouter LLM API with a text prompt, system prompt, and an image file.

    Parameters:
        user_prompt (str): The user text prompt.
        system_prompt (str): The system prompt (instructions/context).
        image_path (str): The file path to the JPEG image.
        max_tokens (int, optional): Maximum number of tokens for the response. Defaults to 256.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Returns:
        str: The text response from the LLM.
    """
    # Encode the image as a Base64 data URI
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    # Make sure the data URI includes the proper MIME type for JPEG images.
    image_data_uri = f"data:image/jpeg;base64,{encoded_string}"

    # Construct the messages payload.
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri
                    }
                }
            ]
        }
    ]

    payload = {
        "model": os.getenv('MODEL_NAME') or model_name,
        "messages": messages,
        "max_tokens": max_tokens
    }

    # Define the API endpoint and headers.
    api_endpoint = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Replace with your site URL.
        "X-Title": "<YOUR_SITE_NAME>",      # Optional. Replace with your site name.
    }

    try:
        # Make the POST request to the OpenRouter API.
        response = requests.post(api_endpoint, headers=headers, json=payload)
        
        # Raise an exception if the request failed.
        response.raise_for_status()
        
        # Parse the JSON response
        result = response.json()
        
        if result and "choices" in result and len(result["choices"]) > 0:
            output = result["choices"][0]["message"]["content"].strip()
            token_usage = result["usage"]["total_tokens"] if "usage" in result else "n/a"
            if debug:
                print("LLM Output:", output)
                print("Total tokens used:", token_usage)
            return output
        else:
            error_msg = "Error: No valid response received from the API. Please check your API key, model, or network connectivity."
            print(error_msg)
            return "error: failed to get response from LLM"  # Return a default error message
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return "error: exception occurred during LLM call"  # Return a default error message

def save_images(frames):
        # Save captured images for each camera.
    captured_image_paths = {}
    for cam_id, (success, frame) in frames.items():
        if not success or frame is None:
            print(f"Error: Failed to capture image from camera {cam_id}")
            continue
        # Optionally display the captured image for a short time.
        cv2.imshow(f"Captured Image from Cam {cam_id}", frame)
        cv2.waitKey(2000)  # Display for 2 seconds.
        cv2.destroyWindow(f"Captured Image from Cam {cam_id}")
        # Save the captured frame as a JPG file.
        file_path = f"workflow/captured_image_cam_{cam_id}.jpg"
        cv2.imwrite(file_path, frame)
        captured_image_paths[cam_id] = file_path

    if not captured_image_paths:
        print("Error: No images were captured successfully.")
        exit(1)
    return captured_image_paths

def get_prompt(user_prompt: str):
    """
    Capture images from multiple cameras, upload them to Cloudinary,
    and call the LLM API using the provided user_prompt.
    
    Args:
        user_prompt (str): The prompt that will be sent to the LLM.
    """
    # Initialize the multi-camera interface (uses all available cameras).
    multi_camera = MultiCameraInterface(height=240, width=320)
    time.sleep(3)
    frames = multi_camera.capture_frames()  # returns a dict: {camera_id: (success, frame)}
    multi_camera.close()  # Release cameras after capturing

    captured_image_paths = save_images(frames)

    # Use the first captured image for LLM call (modify as needed to handle multiple images)
    if captured_image_paths:
        first_image_path = next(iter(captured_image_paths.values()))
        llm_response = llm_api_call(
            user_prompt, 
            system_prompt, 
            first_image_path
        )
        return llm_response
    else:
        return "Error: No images captured"



if __name__ == "__main__":

    # Testing image upload time and LLM response time 
    image_path = '/Users/johnmok/Documents/GitHub/AI_monitor/backend_server/sustainable_inner.jpg'
    
    # Record start time
    start_time = time.time()
    
    # Test with LLM
    print("\n=== Testing LLM with image ===")
    user_prompt = "where is the table?"
    
    llm_response = llm_api_call(
        user_prompt, 
        system_prompt=os.getenv("SYSTEM_PROMPT") or "You are a helpful assistant. Answer the question in a short way.",
        image_path=image_path,
        debug=False
    )
    
    print(f"LLM Response: {llm_response}")
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")
#####################
# New helper functions
#####################


# %%
