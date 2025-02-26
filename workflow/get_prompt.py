#%% NEW: Capture images from multiple cameras, upload to Cloudinary, and send to LLM

from dotenv import load_dotenv

load_dotenv()  # Load .env variables

import os
workspace_path = os.getenv("WORKSPACE_PATH")
model_name = os.getenv("MODEL_NAME")
system_prompt = os.getenv("SYSTEM_PROMPT")
import sys
sys.path.append(workspace_path)  # Add workspace path to Python path

import cv2
import time
# Import the MultiCameraInterface from your camera interface module.
from sim_env.Kinova_gen2.src.devices.camera_interface import MultiCameraInterface
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

def call_llm_with_images(user_prompt, uploaded_urls, model_name=model_name, system_prompt=system_prompt, debug=False):
    from openai import OpenAI
    # Prepare list of image objects for the API call.
    image_objects = []
    for cam_id, url in uploaded_urls.items():
        image_objects.append({"type": "image_url", "image_url": {"url": url}})
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
 
    try:
        response = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Replace with your site URL.
                "X-Title": "<YOUR_SITE_NAME>",      # Optional. Replace with your site name.
            },
            extra_body={},
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_prompt}] + image_objects
                }
            ]
        )
        
        if response is not None and getattr(response, "choices", None):
            output = response.choices[0].message.content.strip()
            token_usage = response.usage.total_tokens if getattr(response, "usage", None) else "n/a"
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

def convert_image_to_local_url(frames):
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

    captured_image_paths = convert_image_to_local_url(frames)

    # Use modular functions to upload images and perform the API call.
    uploaded_image_urls = upload_images_to_cloudinary(captured_image_paths)
    llm_response =call_llm_with_images(user_prompt, uploaded_image_urls, model_name=model_name, system_prompt=system_prompt)
    return llm_response
if __name__ == "__main__":
    # Get prompt from the user and call get_prompt.
    user_prompt = input("Enter your prompt: ")
    start_time = time.time()
    llm_response = get_prompt(user_prompt)
    print(llm_response)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

#####################
# New helper functions
#####################


# %%
