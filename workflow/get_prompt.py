#%% NEW: Capture image from camera and upload to Cloudinary
import cv2
from sim_env.Kinova_gen2.src/devices/camera_interface import CameraInterface  # Import the camera control script

# Initialize camera and capture a single frame
camera = CameraInterface(camera_id=0, width=640, height=480, fps=30)
ret, frame = camera.capture_frame()
camera.close()

if not ret:
    print("Error: Failed to capture image from camera")
    exit(1)

# Save the captured frame as a JPG file (ensures the Cloudinary URL will be jpg)
captured_image_path = "captured_image.jpg"
cv2.imwrite(captured_image_path, frame)

import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env variables

# Configure Cloudinary       
cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Upload the captured image to Cloudinary, forcing JPG format
upload_result = cloudinary.uploader.upload(captured_image_path, public_id="my_image", format="jpg")
image_url = upload_result["secure_url"]
print("Uploaded image URL:", image_url)

# # Optional: Generate transformed URLs (if needed)
# optimize_url, _ = cloudinary_url("my_image", fetch_format="auto", quality="auto")
# print("Optimized URL:", optimize_url)
# auto_crop_url, _ = cloudinary_url("my_image", width=500, height=500, crop="auto", gravity="auto")
# print("Auto-cropped URL:", auto_crop_url)

from openai import OpenAI

# Initialize OpenRouter API client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Get prompt from the user
user_prompt = input("Enter your prompt: ")

response = client.chat.completions.create(
    extra_headers={
        "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Replace with your site URL.
        "X-Title": "<YOUR_SITE_NAME>",      # Optional. Replace with your site name.
    },
    extra_body={},
    model="google/gemini-2.0-flash-lite-preview-02-05:free",
    messages=[
        {
            "role": "system",
            "content": "You are an assistant that responds extremely concisely. Limit your output to no more than 100 tokens and do not include any extra commentary."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}  # Use the dynamically uploaded JPG URL
            ]
        }
    ]
)

print("API Response:", response)

# Safely display the returned message content
if response is not None and getattr(response, "choices", None):
    print(response.choices[0].message.content)
else:
    print("Error: No valid response received from the API. Please check your API key, model, or network connectivity.")

# %%
