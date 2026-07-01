import cv2
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import MiniBatchKMeans

class DataPreProcessing:
    
    def sample_kp(shape, size, stride):
        x = ((shape[1]) - 1 // size[1]) + 1
        y = ((shape[1]) - 1 // size[1]) + 1
        kp = np.zeros((x * y), dtype = cv2.KeyPoint)
        i = 0

        for u in range(0, shape[1] - stride + 1, size[1]):
            for v in range(0, shape[0] - stride + 1, size[0]):
                kp[i] = cv2.KeyPoint((u + stride // 2), (v + stride // 2), stride)
                i += 1

        return kp
    
    def extract_vocabulary(raw_data, key_point):

        sift = cv2.SIFT.create()
        descriptors = []

        for data in raw_data:
            kp, des = sift.compute(data, key_point)
            descriptors.extend(des)
        
        descriptors = np.array(descriptors)

        mk_means = MiniBatchKMeans(n_clusters = 50)
        mk_means.fit(descriptors)
        vocabulary = mk_means.cluster_centers_

        return vocabulary
    
    def extract_feat(raw_data, vocabulary, key_point):

        sift = cv2.SIFT.create()
        feat = np.zeros((len(raw_data), len(vocabulary)))
        neigh = neighbors.NearestNeighbors(n_neighbors = 1).fit(vocabulary)

        for i, data in enumerate(raw_data):
            kp, des = sift.compute(data, key_point)
            histogram = np.zeros(len(vocabulary))

            if des is not None:
                distance, indices = neigh.kneighbors(des)
                histogram[indices.flatten()] += 1
                feat[i] = histogram
        
        return feat


def main():

    