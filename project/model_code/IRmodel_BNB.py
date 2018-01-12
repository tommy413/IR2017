from model_code.IRmodel import IRmodel

import os
import pickle

from sklearn.naive_bayes import BernoulliNB


class IR_BNB(IRmodel):
    def __init__(self,args):
        self.data_name = args.data_name
        if args.train:
            self.model = self.get_model()
        elif args.test:
            self.model = self.load_model()

    def get_model(self):
        if not (os.path.exists(os.path.join("model",self.data_name))):
            os.makedirs(os.path.join("model",self.data_name))
            
        clf = BernoulliNB()
        with open(os.path.join("model",self.data_name,"BNB.pkl") , 'wb') as f:
            pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
        return clf

    def load_model(self):
        with open(os.path.join("model",self.data_name,"BNB.pkl") , 'rb') as f:
            clf = pickle.load(f)
        return clf


    def fit(self,X,Y):
        self.model.fit(X, Y)
        with open(os.path.join("model",self.data_name,"BNB.pkl") , 'wb') as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
        

    def predict(self,X,raw_Y):
        import csv

        if not (os.path.exists(os.path.join("result",self.data_name))):
            os.makedirs(os.path.join("result",self.data_name))
        f = open(os.path.join("result",self.data_name,"BNB.csv"),'w')
        wf = csv.writer(f)

        result = self.model.predict(X)
        for i in range(0,len(raw_Y)):
            wf.writerow([raw_Y[i][0],raw_Y[i][1],result[i]])
        f.close()