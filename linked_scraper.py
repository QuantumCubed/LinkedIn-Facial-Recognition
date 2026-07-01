import requests
from PIL import Image
import cv2
import numpy as np
from io import BytesIO

URLs = ['https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTwxVdJ8MFiC6hxzr1TAcovtKJKvoZrGZAnyC36-v1XRA&s']

for url in URLs:

    response = requests.get(url)
    img_bytes = BytesIO(response.content)
    img : Image.Image = Image.open(img_bytes)
    img.show()