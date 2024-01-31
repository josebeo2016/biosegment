from .preprocess import extract_lfcc, extract_mfcc
from .CNN_breath import BIOTYPE, CNNClassifier, VectorDataSource, FeatLoader
from torch.utils.data import Dataset
import os
import librosa
import torch
import time
import yaml

class Wav2bioCNN():
    def __init__(self, device='cpu'):
        BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        self.device = device
        self.config = yaml.load(open(os.path.join(BASE_DIR, "datanew_jun30.yaml"), "r"), Loader=yaml.FullLoader)
        self.classifier= CNNClassifier(os.path.join(BASE_DIR, "out_datanewjun30", "cnn.pth"), self.config, device=device)
        
    def wav2bio(self, data, sr, device, class_weight=[1,5,5], scope=15):
        # resample to 16000
        # convert to numpy array
        if (type(data) is torch.Tensor):
            data = data.numpy()
        if (sr!=16000):
            data = librosa.resample(data, sr, 16000)
        lfcc = VectorDataSource(data=extract_lfcc(sig=data,**self.config['lfcc']),scope=scope)   
        lfcc.rewind()
        data = lfcc.read()
        data_list = []
        while (data is not None):
            data_list.append(data)
            data = lfcc.read()
        result2 = self.classifier.predict_batch2(data_list, device, class_weight=class_weight,batch_size=256)
        return result2