from model_code.IRmodel import IRmodel

import os
import json

import keras.models as kmodels
import keras.layers as klayers
# from keras.layers.core import Dense, Dropout, Activation, TimeDistributed
from keras.layers import  Conv1D,Conv2D, Reshape, Flatten, Dropout, MaxPooling1D
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import load_model,model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint

class IR_CNN(IRmodel):
    def __init__(self,args,maxlen,word_size):
        self.data_name = args.data_name
        if args.train:
            self.model,self.earlystopping,self.checkpoint = self.get_model(maxlen,word_size)
        elif args.test:
            self.model = self.load_model()

    def get_model(self,maxlen,word_size):
        if not (os.path.exists(os.path.join("model",self.data_name))):
            os.makedirs(os.path.join("model",self.data_name))

        model = kmodels.Sequential()
        model.add(klayers.Embedding(word_size+1, 32, input_length = maxlen))

        # model.add(Conv1D(256, 3, padding='valid'))
        model.add(Conv1D(32, 3, padding='same', activation = "relu"))
        model.add(MaxPooling1D(2, padding='same'))
        model.add(Conv1D(64, 2, padding='same', activation = "relu"))

        model.add(Flatten())
        # model.add(Dropout(0.2))
        model.add(klayers.Dense(64, activation = "relu"))
        # model.add(Dropout(0.2))
        # model.add(klayers.Dense(128))
        # model.add(Dropout(0.2))
        model.add(klayers.Dense(13, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        with open(os.path.join("model",self.data_name,"CNN.json") , 'w') as f:
            json.dump(model.to_json(), f)

        model.summary()
        earlystopping = EarlyStopping(monitor='val_loss', patience = 3, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(filepath=os.path.join("model",self.data_name,"CNN_model_weight.hdf5"),
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss',
                                     mode='min')
        return model,earlystopping,checkpoint

    def load_model(self):
        model = model_from_json(json.load(os.path.join("model",self.data_name,"CNN.json")))
        model.load_weights(os.path.join("model",self.data_name,"CNN_model_weight.hdf5"))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


    def fit(self,X,Y,Xval,Yval):
        self.model.fit(X, Y, 
                    validation_data=(Xval, Yval),
                    epochs=500, 
                    batch_size=50,
                    callbacks=[self.earlystopping,self.checkpoint])
        self.model.load_weights(os.path.join("model",self.data_name,"CNN_model_weight.hdf5"))
        self.model.fit(Xval, Yval, 
                    validation_data=(X, Y),
                    epochs=500, 
                    batch_size=50,
                    callbacks=[self.earlystopping,self.checkpoint])
        self.model.load_weights(os.path.join("model",self.data_name,"CNN_model_weight.hdf5"))

    def predict(self,X,raw_Y):
        import csv

        if not (os.path.exists(os.path.join("result",self.data_name))):
            os.makedirs(os.path.join("result",self.data_name))
        f = open(os.path.join("result",self.data_name,"CNN.csv"),'w')
        wf = csv.writer(f)

        result = self.model.predict_classes(X)
        for i in range(0,len(raw_Y)):
            wf.writerow([raw_Y[i][0],raw_Y[i][1],result[i]])
        f.close()