import csv
import os

def precision(true,pred):
    class_dict = {}
    acc_dict = {}
    p_sum = 0.0 
    macro_p = 0.0
    micro_p = 0.0

    for i in range(0,len(true)):
        l = class_dict.get(pred[i],[1.0,13.0])
        l[1] = l[1] + 1.0
        if pred[i] == true[i]:
            l[0] = l[0] + 1.0
        class_dict[pred[i]] = l

    for k in range(0,13):
        l = class_dict.get(k,[1.0,13.0]) 
        acc_dict[k] = [(l[0]/l[1]),l[1]]
        p_sum = p_sum + l[1]

    for k in range(0,13):
        macro_p = macro_p + acc_dict[k][0]
        micro_p = micro_p + acc_dict[k][0]*acc_dict[k][1]

    macro_p = macro_p / 13.0
    micro_p = micro_p / p_sum
    return macro_p,micro_p

def recall(true,pred):
    class_dict = {}
    acc_dict = {}
    p_sum = 0.0 
    macro_r = 0.0
    micro_r = 0.0

    for i in range(0,len(true)):
        l = class_dict.get(true[i],[1.0,13.0])
        l[1] = l[1] + 1.0
        if pred[i] == true[i]:
            l[0] = l[0] + 1.0
        class_dict[true[i]] = l

    for k in range(0,13):
        l = class_dict.get(k,[1.0,13.0]) 
        acc_dict[k] = [(l[0]/l[1]),l[1]]
        p_sum = p_sum + l[1]

    for k in range(0,13):
        macro_r = macro_r + acc_dict[k][0]
        micro_r = micro_r + acc_dict[k][0]*acc_dict[k][1]

    macro_r = macro_r / 13.0
    micro_r = micro_r / p_sum
    return macro_r,micro_r

def f1score(p,r):
    return 2*p*r/(p+r)

def plot_cm(true,pred,img_path):
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    array = [[0 for i in range(0,13)] for j in range(0,13)]

    for i in range(0,len(true)):
        array[true[i]][pred[i]] = array[true[i]][pred[i]] + 1

    df_cm = pd.DataFrame(array, range(13),
                  range(13))
    #plt.figure(figsize = (10,7))
    fig = None
    ax = None
    sn.set(font_scale=1.4)#for label size
    ax = sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
    fig = ax.get_figure()
    fig.savefig(img_path)
    plt.clf()
    plt.cla()

def analysis(data_name,model_name):
    result_path = os.path.join("result",data_name)
    rst_rf = csv.reader(open(os.path.join(result_path,"%s.csv" % model_name),'r'))

    meta_path = os.path.join(result_path,"%s_meta.csv" % model_name)
    meta_wf = csv.writer(open(meta_path,'w'))

    pred = []
    gtrue = []
    for line in list(rst_rf):
        gtrue.append(int(line[1]))
        pred.append(int(line[2]))

    macro_p,micro_p = precision(gtrue,pred)
    macro_r,micro_r = recall(gtrue,pred)
    macro_f = f1score(macro_p,macro_r)
    micro_f = f1score(micro_p,micro_r)
    meta_wf.writerow(["Macro_p" ,macro_p])
    meta_wf.writerow(["Micro_p" ,micro_p])
    meta_wf.writerow(["Macro_r" ,macro_r])
    meta_wf.writerow(["Micro_r" ,micro_r])
    meta_wf.writerow(["Macro_f" ,macro_f])
    meta_wf.writerow(["Micro_f" ,micro_f])

    plot_cm(gtrue,pred,os.path.join(result_path,"%s_confusion_matrix.png" % model_name))
