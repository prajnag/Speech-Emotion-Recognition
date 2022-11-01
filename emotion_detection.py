import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras import regularizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Activation, Conv1D
from keras.models import Model

data_list= os.listdir('/Users/prajnagirish/Desktop/Emotion_Recognition/Dataset')

emotions=[]
for item in data_list:
    if item[6:-16]=='02':
        emotions.append('calm')
    elif item[6:-16]=='03':
        emotions.append('happy')
    elif item[6:-16]=='04':
        emotions.append('sad')
    elif item[6:-16]=='05':
        emotions.append('angry')


labels = pd.DataFrame(emotions)
df = pd.DataFrame(columns=['feature'])
bookmark=0
for index,y in enumerate(data_list):
    if data_list[index][6:-16]!='01' and data_list[index][6:-16]!='07' and data_list[index][6:-16]!='08':
        X, sample_rate = librosa.load('/Users/prajnagirish/Desktop/Emotion_Recognition/Dataset'+y, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                            sr=sample_rate, 
                                            n_mfcc=13), axis=0)
        feature = mfccs
        df.loc[bookmark] = [feature]
        bookmark=bookmark+1       

newdf = pd.concat([df3,labels], axis=1)
rnewdf = newdf.rename(index=str, columns={"0": "label"})
newdf1 = np.random.rand(len(rnewdf)) < 0.8
train = rnewdf[newdf1]
test = rnewdf[~newdf1]
trainfeatures = train.iloc[:, :-1]
trainlabel = train.iloc[:, -1:]
testfeatures = test.iloc[:, :-1]
testlabel = test.iloc[:, -1:]

x_train = np.array(trainfeatures)
y_train = np.array(trainlabel)
x_test = np.array(testfeatures)
y_test = np.array(testlabel)

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))
X_train.shape

x_traincnn =np.expand_dims(x_train, axis=2)
x_testcnn= np.expand_dims(x_test, axis=2)
model = Sequential()
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('softmax'))

preds = model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)

accuracy1 = getAccuracy(y_test, preds)
print(accuracy1)