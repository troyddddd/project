import numpy as np
import json
import matplotlib.pyplot as plt
from evaluate import exact_match_score, f1_score

def read_file(fname):
    res = []
    with open(fname, "r") as f:
        data = json.load(f)
        for start, end, paragraph, answer in data:
            res.append((np.array(start[0]), np.array(end[0]),
                                 [str(i) for i in paragraph[0]], answer[0]))
    return res
    
def ensemble(results):
    res = []
    for i in range(len(results[0])):
        start = np.mean(np.array([r[i][0] for r in results]), axis=0)
        end = np.mean(np.array([r[i][1] for r in results]), axis=0)
        res.append((start, end, results[0][i][2], results[0][i][3]))
    return res

def argmax_eval(data):
    em = 0.
    f1 = 0.
    for start, end, paragraph, answer in data:
        text_p = " ".join(paragraph[np.argmax(start):np.argmax(end)+1])
        text_g = " ".join(paragraph[answer[0]:answer[1]+1])
        f1 += f1_score(text_p, text_g)
        em += exact_match_score(text_p, text_g)
    print("argmax EM: {:.5f}, F1: {:.5f}".format(em/len(data), f1/len(data)))
    
def search_eval(data, max_span=15, op="+"):
    em = 0.
    f1 = 0.
    for start, end, paragraph, answer in data:
        s, e, prob = 0, 0, 0
        for i in range(len(start)):
            for j in range(min(max_span, len(end)-i)):
                if op == "+":
                    if start[i] + end[i+j] > prob:
                        prob = start[i] + end[i+j]
                        s, e = i, i+j
                if op == "*":
                    if start[i] * end[i+j] > prob:
                        prob = start[i] * end[i+j]
                        s, e = i, i+j
        text_p = " ".join(paragraph[s:e+1])
        text_g = " ".join(paragraph[answer[0]:answer[1]+1])
        f1 += f1_score(text_p, text_g)
        em += exact_match_score(text_p, text_g)
    print("search EM: {:.5f}, F1: {:.5f} (max_span={}, op={})".format(
            em/len(data), f1/len(data), max_span, op))
   
def eval_all(fname):
    data = ensemble([read_file(f) for f in fname])
    argmax_eval(data)
    search_eval(data,15,"+")
    search_eval(data,15,"*")
    search_eval(data,10,"+")
    search_eval(data,10,"*")
    search_eval(data,12,"+")
    search_eval(data,12,"*")
    search_eval(data,18,"+")
    search_eval(data,18,"*")


def plot(fname, config="config.txt", legends_to_plot=set(list(range(18)))):
    legend = {}
    prameter = {}
    em = {}
    f1 = {}
    for fn in fname:
        with open(fn, "r") as fh:
            for line in fh:
                l = line.split()
                if l[0] in em:
                    em[l[0]].append(float(l[1][:-1]))
                else:
                    em[l[0]] = [float(l[1][:-1])]
                if l[0] in f1:
                    f1[l[0]].append(float(l[3]))
                else:
                    f1[l[0]] = [float(l[3])]
    for i in em:
        num = []
        for char in i:
            if char.isdigit():
                num.append(char)
        legend[i] = int("".join(num))
        
    with open(config, "r") as fh:
        for line in fh:
            l = line.split()
            if len(l) == 7:
                prameter[int(l[0][:-1])] = " ".join(l[2::2])
        prameter[19] = "0.5, 150, 200"
        prameter[20] = "0.6, 150, 200"
    
    plt.figure(figsize=(6,5))            
    for i in em:
        try:
            if legend[i] in legends_to_plot:
                plt.plot(em[i], label=prameter[legend[i]])
        except:
            continue
        
    plt.title("EM score vs. number of epochs (GRU pre-processing)")
    plt.xlabel("number of epochs (legend: dropout, hidden units, embedding)")
    plt.ylabel("EM score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("em.svg", bbox_inches='tight')
    
    plt.figure(figsize=(6,5))            
    for i in f1:
        try:
            if legend[i] in legends_to_plot:
                plt.plot(f1[i], label=prameter[legend[i]])
        except:
            continue
        
    plt.title("F1 score vs. number of epochs (GRU pre-processing)")
    plt.xlabel("number of epochs (legend: dropout, hidden units, embedding)")
    plt.ylabel("F1 score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("f1.svg", bbox_inches='tight')

eval_all(["./../model_12_15epoch/model_5_epoch_prediction.json",
          "./../model_12_15epoch/model_4_epoch_prediction.json",
          "./../model_12_15epoch/model_1_epoch_prediction.json",
          "./../18/model_5_epoch_prediction.json",
          "./../18/model_1_epoch_prediction.json",
          "./../transfer/model_4_epoch_prediction.json",
          "./../transfer/model_2_epoch_prediction.json"])

#plot(["./../LSTM_result_10epoch_original_1_6.txt",
#      "./../LSTM_7_12.txt",
#      "./../LSTM_result_13_18.txt"], legends_to_plot=set([10,11,12]))

#plot(["./../0.5.txt", "./../0.6.txt", "./../LSTM_7_12.txt"],
#     legends_to_plot=set([10,11,12,19,20]))

plot(["./../result_GRU_1-6.txt", "./../GRU_7_12.txt", "./../GRU-result_13-18.txt"])
