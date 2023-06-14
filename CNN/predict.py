from .preprocess import extract_lfcc, extract_mfcc
from .CNN_breath import BIOTYPE, CNNClassifier
from .hparams import *
import pickle
from auditok import DataValidator, ADSFactory, DataSource, StreamTokenizer, BufferAudioSource, player_for
import soundfile as sf
import os


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# if __name__ == '__main__':

# load models
classifier = CNNClassifier(os.path.join(BASE_DIR, "out", "cnn.pth"))

biotype = CNN_breath.BIOTYPE

def wav2bio(data, sr):
    # data, sr = sf.read(wav_path)
    lfcc = VectorDataSource(data=extract_lfcc(sig=data,sr=sr),scope=15)   
    # tokenized:
    lfcc.rewind()
    data = lfcc.read()
    # print(data)
    result = []
    while (data is not None):
        result.append(biotype[classfier.predict(data)[0][0]])
        data = lfcc.read()
    return result

# wav_path = "/root/dataset/ASVspoof/LA/ASVspoof2019_LA_train/flac/LA_T_5450704.flac"
# data, sr = sf.read(wav_path)
# wav2bio(data, sr)