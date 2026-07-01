import json
import base64
import requests

# Function to save profiles to JSON file
def save_profiles(profiles):
    with open("profiles.json", "w") as json_file:
        json.dump(profiles, json_file, indent=4)
    print("Profile data saved to 'profiles.json'")

# Function to load profiles from JSON file
def load_profiles():
    try:
        with open("profiles.json", "r") as json_file:
            profiles = json.load(json_file)
        return profiles
    except FileNotFoundError:
        return []

# Load existing profiles from JSON file
profiles = load_profiles()

# Prompt the user for input iteratively
while True:
    print("Enter profile details:")
    name = input("Enter the name (or type 'quit' to exit): ")
    if name.lower() == 'quit':
        break
    picture_url = input("Enter the picture URL: ")
    education = input("Enter the education: ")

    # Fetch the image from the URL
    response = requests.get(picture_url)
    if response.status_code == 200:
        # Convert the image to base64
        image_base64 = base64.b64encode(response.content).decode("utf-8")
    else:
        print("Failed to fetch image from URL:", picture_url)
        image_base64 = None

    # Create a dictionary to store the data
    profile_data = {
        "name": name,
        "picture_base64": image_base64,
        "education": education
    }

    # Append the profile data to the list
    profiles.append(profile_data)

# Save the profiles to JSON file
save_profiles(profiles)
