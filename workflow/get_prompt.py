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

    # Use modular functions to upload images and perform the API call.
    # uploaded_image_urls = upload_images_to_cloudinary(captured_image_paths)
    uploaded_image_urls = upload_image_to_server(captured_image_paths)
    llm_response =call_llm_with_images(user_prompt, uploaded_image_urls, model_name=model_name, system_prompt=system_prompt)
    return llm_response



if __name__ == "__main__":

    # Testing image upload time and LLM response time 
    # Upload the image
    server_url = 'http://3.25.67.65:8000/images/'
    image_path = '/Users/johnmok/Documents/GitHub/FYP-kitchen-assistive-robot/workflow/assortment-delicious-healthy-food_23-2149043057.jpg'
    
    # Record start time
    start_time = time.time()
    
    image_url = upload_image_to_server(server_url, image_path)
    
    # Calculate and print upload time
    upload_time = time.time() - start_time
    print(f"Image upload took {upload_time:.2f} seconds")
    if image_url:
        print(f"Image uploaded successfully: {image_url}")
        
        # Test with LLM
        print("\n=== Testing LLM with uploaded image ===")
        user_prompt = "clean the table?"
        uploaded_image_urls = {'test_cam': image_url}
        model_name = os.getenv("MODEL_NAME")
        print("MODEL_NAME: ", model_name)
        llm_response = call_llm_with_images(
            user_prompt, 
            uploaded_image_urls, 
            model_name=model_name, 
            # system_prompt='you are helpful assistant. Answer the question in a short way.'
        )
        print(f"LLM Response: {llm_response}")
    else:
        print("Image upload failed")
    
    response_time = time.time() - start_time - upload_time
    print(f"LLM Response time: {response_time:.2f} seconds")
    total_time = upload_time + response_time
    print(f"Total time: {total_time:.2f} seconds")
    # ~6 seconds
#####################
# New helper functions
#####################


# %%

