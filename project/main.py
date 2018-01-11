import argparse
from preprocess import get_data
from model_code.IRmodel_RNN import IR_RNN


def parse():
    parser = argparse.ArgumentParser(description="IR final")
    parser.add_argument('--train', action='store_true', help='whether training')
    parser.add_argument('--test', action='store_true', help='whether testing')
    parser.add_argument('--data_name', type = str, help='dataset name')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    X,Y,Xval,Yval,raw_Y,dictionary,maxlen = get_data(args.data_name)

    RNN_model = IR_RNN(args,maxlen)
    RNN_model.fit(X,Y,Xval,Yval)
    RNN_model.predict(X,raw_Y)
