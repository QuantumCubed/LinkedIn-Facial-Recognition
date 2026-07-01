from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from chroma_db import VectorDB
import glob
import ast

#ADD FACENOTFOUND ERROR HANDELING
query_only = True
chroma = VectorDB(del_coll = False)
mtcnn = MTCNN()
facenet = InceptionResnetV1(pretrained = 'vggface2').eval()

embeddings = []
documents = []
metadatas = []
ids = []

directory = 'images'

if query_only is False:
    for i, file in enumerate(glob.iglob(f'{directory}/*')):
        print(file)
        img = np.array(Image.open(file))
        img_cropped = mtcnn(img)
        if img_cropped is None:
            continue
        img_embedding : torch.Tensor = facenet(img_cropped.unsqueeze(0))

        #embeddings.extend(torch.exp(img_embedding).detach().numpy())
        embeddings.extend(torch.exp(img_embedding).tolist())
        documents.append('img{}'.format(i))
        metadatas.append({ 'file' : '{}'.format(file), 'MatLike - B64' : '{}'.format(np.ndarray.tobytes(img)), 'shape' : '{}'.format(img.shape) })
        ids.append('{}'.format(i))

    chroma.insert_embedding(embeddings, documents, metadatas, ids)

#test_img = np.array(Image.open("C:\\Users\\anish\\Pictures\\WIN_20240416_20_22_01_Pro.jpg"))
#test_img = np.array(Image.open("C:\\Users\\anish\\Pictures\\Screenshots\\kp.jpg"))
test_img = np.array(Image.open("C:\\Users\\anish\\Pictures\\IMG_9065.jpg"))

test_crop = mtcnn(test_img)
test_embed : torch.Tensor = facenet(test_crop.unsqueeze(0))

te = []
te.extend(torch.exp(test_embed).tolist())

response = chroma.query_image(te, 1)['metadatas'][0][0]
mat_bytes = ast.literal_eval(response['MatLike - B64'])
mat_shape : tuple = ast.literal_eval(response['shape'])
mat_img = np.frombuffer(mat_bytes, dtype = np.uint8).reshape(mat_shape) #.astype(dtype = np.float32)


image_new = Image.fromarray(mat_img)
image_new.show()

#img_np = np.frombuffer(MatImg, dtype = np.float32)

#plt.imshow(img_np)
#plt.axis('off')
#plt.show()

