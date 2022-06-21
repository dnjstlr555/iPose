import paddlehub as hub
import numpy as np
from PIL import Image
from IPython.display import display

from PIL import Image
from numpy import dot
from numpy.linalg import norm
import cv2
import paddle

import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Reshape, Input, InputLayer
from keras.models import Sequential, Model
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint
import pickle
from transformers import DetrFeatureExtractor, DetrForObjectDetection
from ntpath import splitext
from sklearn.cluster import KMeans

def getTrainSet(data, test=0.1, seed=42):
    X_train, X_test = train_test_split(data, test_size=test, random_state=seed)
    return X_train, X_test

def build_autoencoder(img_shape, code_size=500):
    # The encoder
    encoder = Sequential()
    encoder.add(InputLayer(img_shape))
    encoder.add(Flatten())
    encoder.add(Dense(code_size+code_size))
    encoder.add(Dense(code_size))

    # The decoder
    decoder = Sequential()
    decoder.add(InputLayer((code_size,)))
    decoder.add(Dense(np.prod(img_shape))) # np.prod(img_shape) is the same as 32*32*3, it's more generic than saying 3072
    decoder.add(Reshape(img_shape))

    return encoder, decoder


def apply_gaussian_noise(X, sigma=0.1):
    noise = np.random.normal(loc=0.0, scale=sigma, size=X.shape)
    return X + noise

def makeModel(vector_size=500):
    img_shape=(38, 23, 23)
    print(img_shape)
    encoder, decoder = build_autoencoder(img_shape, vector_size)
    inp = Input(img_shape)
    code = encoder(inp)
    reconstruction = decoder(code)
    autoencoder = Model(inp,reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse')
    print(autoencoder.summary())
    return (encoder, code, decoder, autoencoder)

class AutoEncoder:
    def __init__(self, vector_size=500):
        (self.encoder, self.code, self.decoder, self.model)=makeModel(vector_size)
        self.vector_size=vector_size
    def getVector(self, data):
        return self.encoder.predict(data[None])[0]

class Poseimg:
    def __init__(self, path, raw_conv):
        self.path=path
        self.vec=np.array([])
        self.raw_conv=raw_conv
        self.people=-1
        self.like=0
        self.dislike=0
    def getvec(self):
        return self.vec

class Project:
    def __init__(self, vec_size=500, train="./dataset/train/",val="./dataset/val/",test="./dataset/test/", load=False, updateKmeans=False, k=15):
        self.PoseModel=hub.Module(name='openpose_body_estimation')
        self.AutoModel=AutoEncoder(vec_size)
        self.feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
        self.CountModel = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
        self.lastHistory={"loss":[], "val_loss":[]}
        if load==False:
            self.data={"train":self.folderToConv(train), "val":self.folderToConv(val), "test":self.folderToConv(test)}
            self.save()
        else:
            with open("./conv.vec", "rb") as f:
                self.data=pickle.loads(f.read())
            self.AutoModel.model.load_weights("./best_model.h5")
        if updateKmeans==True:
            self.kMeansCalculate(k)
            with open("kmeans.vec", "wb") as f:
                f.write(pickle.dumps(self.kmodel))
        else:
            with open("kmeans.vec", "rb") as f:
                self.kmodel=pickle.loads(f.read())
    def kMeansCalculate(self, k):
        df = np.array([i.vec for i in self.getData()])
        self.kmodel = KMeans(n_clusters = k)
        self.kmodel.fit(df)
        self.predict=self.kmodel.predict(df)
    def kMeansCenter(self):
        def sim(A, B):
            return dot(A, B)/(norm(A)*norm(B)) #ab 유사도 구하는거
        centers=self.kmodel.cluster_centers_
        res=[]
        for i in range(len(centers)):
            near=centers[i]
            distance=[]
            for k in self.getData(): #전체 데이터셋에 대해서
                distance.append([k.path, sim(k.vec, near)]) #distance list에 (경로, 유사도) 데이터를 넣는다.
            res.append(sorted(distance, key=lambda x:x[1], reverse=True)[0][0])
        return res
    
    def pathToConv(self, path):
        img=Image.open(path).resize((256,256))
        numpy_image=np.array(img)
        orgImg=cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        self.PoseModel.eval()
        data, imageToTest_padded, pad = self.PoseModel.transform(orgImg)
        Mconv7_stage6_L1, Mconv7_stage6_L2 = self.PoseModel.forward(paddle.to_tensor(data))
        a = Mconv7_stage6_L1.numpy() #vector
        b = Mconv7_stage6_L2.numpy() #landmark dots
        return a[0]

    def folderToConv(self, path):
        fnames = os.listdir(path)
        lst=[]
        for i in tqdm(fnames):
            if splitext(i)[1]==".jpg":
                rat=self.pathToConv(path+i)
                lst.append(Poseimg(path+i,rat))
        return np.array(lst)
    
    def getConvData(self, data):
        return np.array([i.raw_conv for i in data])
    
    def train(self, epochs, noise=True, sigma=0.2, test=0.1, seed=42):
        (train, test)=getTrainSet(self.getConvData(self.data["train"]), test, seed) #concat data["train"]["val"]
        mc = ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min',save_best_only=True)
        history={"loss":[], "val_loss":[]}
        for i in range(epochs):
            print("Epoch %i/25, Generating corrupted samples..."%(i+1))
            X_train_noise = apply_gaussian_noise(train, sigma)
            X_test_noise = apply_gaussian_noise(test, sigma)

            # We continue to train our model with new noise-augmented data
            log = self.AutoModel.model.fit(x=X_train_noise, y=train, epochs=1,
                            validation_data=[X_test_noise, test], callbacks=[mc])
            history['loss'].append(log.history['loss'])
            history['val_loss'].append(log.history['val_loss'])
        self.lastHistory=history
    
    def visualNoise(self,sigma=0.1):
        for i in self.data["test"][0].raw_conv:
            plt.subplot(1,2,1)
            plt.title("Original")
            plt.imshow(i, cmap='hot', interpolation='nearest')
            #show_image(img)

            plt.subplot(1,2,2)
            plt.title("Noise")
            plt.imshow(apply_gaussian_noise(i, sigma))
            plt.show()
    def visualTrain(self):
        plt.plot(self.lastHistory['loss'])
        plt.plot(self.lastHistory['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    def save(self):
        with open("./conv.vec", "wb") as f:
            f.write(pickle.dumps(self.data))
        self.AutoModel.model.save("./model.h5")
    def getData(self):
        return [*self.data["train"], *self.data["test"], *self.data["val"]]
    def vectorise(self):
        for k in [self.data["train"], self.data["test"], self.data["val"]]:
            for i in k:
                i.vec=self.AutoModel.getVector(i.raw_conv)
    def visualAutoEncoder(self,data=0,part=0):
        """Draws original, encoded and decoded images"""
        image=self.data["test"][data].raw_conv
        # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
        code = self.AutoModel.encoder.predict(image[None])[0]
        reco = self.AutoModel.decoder.predict(code[None])[0]

        plt.subplot(1,3,1)
        plt.title("Original")
        plt.imshow(image[part], cmap='hot', interpolation='nearest')
        #show_image(img)

        plt.subplot(1,3,2)
        plt.title("Code")
        plt.imshow(code.reshape([code.shape[-1]//2,-1]))

        plt.subplot(1,3,3)
        plt.title("Reconstructed")
        plt.imshow(reco[part], cmap='hot', interpolation='nearest')
        plt.show()
    def plot(self,prob):
        i=0
        for p in prob:
            if self.CountModel.config.id2label[p.argmax().item()] == 'person':
                i+=1
        return i

    def count(self,images):
        img = Image.open(images)
        inputs = self.feature_extractor(img, return_tensors="pt")
        outputs = self.CountModel(**inputs)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        return self.plot(probas[keep])
    def countPeople(self):
        for k in [self.data["train"], self.data["test"], self.data["val"]]:
            for i in tqdm(k):
                i.people=self.count(i.path)
    def GetSimPhoto(self, img):
        def sim(A, B):
            return dot(A, B)/(norm(A)*norm(B))
        near=[i for i in self.getData() if i.path==img][0]
        distance=[]
        for i in self.getData():
            if i==near: continue
            distance.append((i.path, sim(i.vec, near.vec)))
        return sorted(distance, key=lambda x:x[1], reverse=True)[:13]
