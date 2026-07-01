from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch
from chroma_db import VectorDB
import glob
import ast
class ImageProcessing:

    def __init__(self) -> None:
        
        self.chroma = VectorDB()
        self.mtcnn = MTCNN()
        self.facenet = InceptionResnetV1(pretrained = 'vggface2').eval()

        """
        self.ft_state_dict : dict = torch.load(f = 'fine_tuned_weights.pth')
        self.facenet = InceptionResnetV1(pretrained = 'vggface2', num_classes = 19)
        self.facenet.load_state_dict(self.ft_state_dict)
        self.facenet.eval()
        """
        

    def InitializeDB(self) -> None:

        embeddings = []
        documents = []
        metadatas = []
        ids = []

        directory = 'images'
        sub_directory = ''

        for i, dir in enumerate(glob.iglob(f'{directory}/*')):
            index = dir.strip().find('\\')
            name = dir[index + 1:]
            sub_directory = dir
            for j, img_loc in enumerate(glob.iglob(f'{sub_directory}/*')):
                #print(img)
                img = np.array(Image.open(img_loc))
                img_cropped = self.mtcnn(img)
                if img_cropped is None:
                    print(dir, 'Could Not Detect Face')
                    continue
                img_embedding : torch.Tensor = self.facenet(img_cropped.unsqueeze(0))

                #embeddings.extend(torch.exp(img_embedding).detach().numpy())

                embeddings.extend(torch.exp(img_embedding).tolist())
                documents.append('Folder: {}, Image: {}'.format(i, j))
                metadatas.append({ 'file' : '{}'.format(img_loc), 'name' : '{}'.format(name), 'MatLike - B64' : '{}'.format(np.ndarray.tobytes(img)), 'shape' : '{}'.format(img.shape) })
                ids.append('{}'.format(i))

        self.chroma.insert_embedding(embeddings = embeddings, documents = documents, metadatas = metadatas, ids = ids)

    def Frame2Tensor(self, frame) -> torch.Tensor:

        PIL_Frame = Image.fromarray(frame)
        image_crop : torch.Tensor = self.mtcnn(PIL_Frame)

        if image_crop is not None:
            tensor : torch.Tensor = self.facenet(image_crop.unsqueeze(0))
            return tensor

    def Tensor2Image(self, tensor) -> tuple:

        if tensor is None: return None #make sure this doesnt break stuff

        ve = []
        ve.extend(torch.exp(tensor).tolist())

        response = self.chroma.query_image(ve, 1)['metadatas'][0][0]
        person_name : str = (response['name'])
        mat_bytes = ast.literal_eval(response['MatLike - B64'])
        mat_shape : tuple = ast.literal_eval(response['shape'])
        mat_img = np.frombuffer(mat_bytes, dtype = np.uint8).reshape(mat_shape) #.astype(dtype = np.float32)

        image_new = Image.fromarray(mat_img)
        #image_new.show()
        
        return (person_name, image_new) if response is not None else None

#img_np = np.frombuffer(MatImg, dtype = np.float32)

#plt.imshow(img_np)
#plt.axis('off')
#plt.show()