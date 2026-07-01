import os
import json
import base64
from PIL import Image
from io import BytesIO

# Path to your JSON file
json_file_path = 'profiles.json'

# Load JSON data from file
with open(json_file_path, 'r') as file:
    profiles = json.load(file)

# Create a main directory 'data' if it doesn't exist
data_directory = 'images'
os.makedirs(data_directory, exist_ok=True)

# Process each profile
for profile in profiles:
    # Create a directory for each profile
    profile_directory = os.path.join(data_directory, profile['name'])
    os.makedirs(profile_directory, exist_ok=True)
    
    # Decode the base64 image
    image_data = base64.b64decode(profile['picture_base64'])
    image = Image.open(BytesIO(image_data))
    
    # Save the image in JPG format
    image_path = os.path.join(profile_directory, 'picture.jpg')
    image.save(image_path, 'JPEG')

print("Directories and images have been created successfully.")
