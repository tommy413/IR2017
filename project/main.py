import argparse
from preprocess import get_data
from analysis import analysis

from model_code.IRmodel_RNN import IR_RNN
from model_code.IRmodel_CNN import IR_CNN

from model_code.IRmodel_GNB import IR_GNB
from model_code.IRmodel_BNB import IR_BNB
from model_code.IRmodel_MNB import IR_MNB

def parse():
    parser = argparse.ArgumentParser(description="IR final")
    parser.add_argument('--train', action='store_true', help='whether training')
    parser.add_argument('--test', action='store_true', help='whether testing')
    parser.add_argument('--data_name', type = str, help='dataset name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()

    X,Y,test_X,test_Y,dictionary,maxlen,wordsize = get_data(args.data_name,split = False, to_onehot = False)
    GNB_model = IR_GNB(args)
    GNB_model.fit(X,Y)
    GNB_model.predict(test_X,test_Y)
    analysis(args.data_name,"GNB")

    BNB_model = IR_BNB(args)
    BNB_model.fit(X,Y)
    BNB_model.predict(test_X,test_Y)
    analysis(args.data_name,"BNB")

    MNB_model = IR_MNB(args)
    MNB_model.fit(X,Y)
    MNB_model.predict(test_X,test_Y)
    analysis(args.data_name,"MNB")

    # X,Y,Xval,Yval,test_X,test_Y,dictionary,maxlen,wordsize = get_data(args.data_name,split = True, to_onehot = True)

    # CNN_model = IR_CNN(args,maxlen,wordsize)
    # CNN_model.fit(X,Y,Xval,Yval)
    # CNN_model.predict(test_X,test_Y)
    analysis(args.data_name,"CNN")

    # RNN_model = IR_RNN(args,maxlen,wordsize)
    # RNN_model.fit(X,Y,Xval,Yval)
    # RNN_model.predict(test_X,test_Y)
    analysis(args.data_name,"RNN")