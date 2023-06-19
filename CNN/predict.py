from .preprocess import extract_lfcc, extract_mfcc
from .CNN_breath import BIOTYPE, CNNClassifier, VectorDataSource, FeatLoader
from torch.utils.data import Dataset
import os
import librosa
import torch
import time

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
print(device)
classifier = CNNClassifier(os.path.join(BASE_DIR, "out", "cnn.pth"), device=device)
print("Finished loading model")

def wav2bio(data, sr):
    # resample to 16000
    if (sr!=16000):
        data = librosa.resample(data, sr, 16000)
    lfcc = VectorDataSource(data=extract_lfcc(sig=data,sr=16000),scope=15)   
    # # tokenized:
    # lfcc.rewind()
    
    # # # Old code: loop and predict each frame
    # start_time = time.time()
    # data = lfcc.read()
    # result1 = []
    # while (data is not None):
    #     result1.append(classifier.predict(data))
    #     data = lfcc.read()
    
    # end_time = time.time()
    # running_time = end_time - start_time
    # print(f"Running time 1: {running_time:.2f} seconds")
        
    # New code: make a list of frames and predict all at once
    lfcc.rewind()
    # start_time = time.time()
    
    data = lfcc.read()
    data_list = []
    while (data is not None):
        data_list.append(data)
        data = lfcc.read()
    # print(len(data_list))
    # print(len(data_list))
    result2 = classifier.predict_batch(data_list, batch_size=256)
    # print("Finished predicting")
    # end_time = time.time()
    # running_time = end_time - start_time
    # print(f"Running time 2: {running_time:.2f} seconds")

    # assert result1 == result2
    return result2

# if __name__ == '__main__':
#     wav_path = "/dataa/Dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_5450704.flac"
#     data, sr = librosa.load(wav_path, sr=16000)
#     print(wav2bio(data, sr))