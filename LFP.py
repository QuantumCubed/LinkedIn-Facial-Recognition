import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd
import keyboard

class LiveFrameProcesser:

    def FrameInputProcesser():

        #print(torch.cuda.is_available())
        #print(torch.cuda.device_count())
        #print(torch.cuda.current_device())
        #print(torch.cuda.get_device_name(0))

        #processing_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        #mtcnn = MTCNN(device = 'cuda:0')
        #print('MTCNN Device:', mtcnn.device)

        mtcnn = MTCNN()
        facenet = InceptionResnetV1(pretrained = 'vggface2').eval().to('cpu')

        cap = cv2.VideoCapture(1)

        if not cap.isOpened():
                print('Camera IO Exception')
                exit()
        
        while True:
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                deinverted_frame = cv2.flip(frame, 1)
                frame = deinverted_frame
                
                #cv2.imshow('frame', deinverted_frame)
                """
                pil_frame = Image.fromarray(frame)
                img_cropped = mtcnn(pil_frame)
                if img_cropped is not None:
                    img_embedding = facenet(img_cropped.unsqueeze(0))
                    print(img_embedding)

                """
                boxes, _ = mtcnn.detect(frame)

                if boxes is not None:
                    for box in boxes:
                        x, y, w, h = map(int, box)
                        
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        #face = frame[y : y + h, x : x + w]
                        pil_frame = Image.fromarray(frame)
                        img_cropped = mtcnn(pil_frame)
                        if img_cropped is not None:
                            img_embedding = facenet(img_cropped.unsqueeze(0))
                            print(img_embedding)
                        

                cv2.imshow('Live Feed - Face Detection', frame)

                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        return frame
    
    def CalculateEmbedding():
         pass
    
    def CompareEmbeddings(frame_embedding, LinkedIn_embedding):
         pass
    
    def collate_fn(x):
         return x[0]

    def FacialRecog():
         
         device = 'cpu'

         mtcnn = MTCNN(
         image_size=160, margin=0, min_face_size=20,
         thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
         device = device)

         facenet = InceptionResnetV1(pretrained = 'vggface2').eval().to(device)

         dataset = datasets.ImageFolder('./images')
         dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
         loader = DataLoader(dataset, collate_fn = collate_fn, num_workers = (0 if os.name == 'nt' else 4))
         aligned = []
         names = []
         for x, y in loader:
            x_aligned, prob = mtcnn(x, return_prob = True)
            if x_aligned is not None:
                print('Face detected with probability: {:8f}'.format(prob))
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

         aligned = torch.stack(aligned).to(device)
         embeddings = facenet(aligned).detach().cpu()
         dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
         print(pd.DataFrame(dists, columns=names, index=names))
    
    """
    def FrameProcess(frame):
         mtcnn = MTCNN(keep_all = True)
         resnet = InceptionResnetV1(pretrained = 'vggface2').eval()
         boxes, _ = mtcnn.detect(frame)

         if boxes is not None:
            x, y, w, h = map(int, box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    """

def main():
    LiveFrameProcesser.FrameInputProcesser()
    #LiveFrameProcesser.FacialRecog()

main()