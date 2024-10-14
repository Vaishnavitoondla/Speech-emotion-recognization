import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("C:\\Users\\Vaishnavi Patel\\Desktop\\ser 2\\speccch-emotion-recoginition-ravdess-data\\Actor_\\.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)
x_train,x_test,y_train,y_test=load_data(test_size=0.2)
x_train
print((x_train.shape[0], x_test.shape[0]))
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))
from sklearn.metrics import accuracy_score, f1_score
f1_score(y_test, y_pred,average=None)
import pandas as pd
import pickle 
df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df.head(20)
import pickle 
# Writing different model files to file
with open( 'modelForPrediction1.sav', 'wb') as f:
  pickle.dump("modelForPrediction1.sav",f)
filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename,'rb')) # loading the model file from the storage
feature=extract_feature("C:\\Users\\Vaishnavi Patel\\Desktop\ser 2\\speccch-emotion-recoginition-ravdess-data\\Actor_11\\03-01-02-02-02-01-11.wav", mfcc=True, chroma=True, mel=True)
feature=feature.reshape(1,-1)
prediction=loaded_model.predict(feature)
prediction
feature