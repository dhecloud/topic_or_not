import datetime
import os
import numpy as np
import pandas as pd
import bcolz
import pickle 
import spacy
import json
from fastText import load_model
import random 
import string
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def set_default_args(args):
    if not args.name:   
        now = datetime.datetime.now()
        args.name = now.strftime("%Y-%m-%d-%H-%M")
    args.expr_dir = os.path.join(args.save_dir, args.name)

def random_word(N):
    return ''.join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase + string.digits,
            k=N
        )
    )

def save_plt(array, name, args):
    colors = ['blue','red','green','pink','purple']
    plt.cla()
    plt.clf()
    plt.close()
    for i in range(len(array)):
        np.savetxt(os.path.join(args.expr_dir,name[i]+'.txt'), array[i], fmt='%f')
        plt.plot(array[i],color=colors[i], label=name[i])
        plt.xlabel('epoch')
        plt.legend()
    plt.savefig(os.path.join(args.expr_dir, name[i]+'.png'))
    plt.cla()
    plt.clf()
    plt.close()

    
def txt_to_csv(path='data/', files=['questions_tokens', 'answers_tokens']):
    data_list = []
    for file in files:
        data = pd.read_csv(path+file+'.txt', sep="\t",header = None)
        data_list.append(data)
        data.to_csv(path + file + '.csv', header=False, index=False)
        
    data = pd.concat(data_list, axis=1)
    data.to_csv('data/combined.csv', header=False, index=False)

def normalizeStringInDF(s):
    s = s.str.normalize('NFC')
    s = s.str.lower()
    s = s.str.replace(r"([.!?])", r" \1")
    s = s.str.replace(r"[^a-zA-Z0-9.!?]+", r" ")
    s = s.str.replace(r"<a.*</a>", 'url')
    return s

def parseAugmentedPermutations():
    stime = time.time()
    augments = []
    files = ['output_success_'+ str(x) + '.txt' for x in range(0,18)]
    for file in files:
        with open('data/permutations/'+file, 'r', encoding='utf-8') as f:
            print("processing", file, "...")
            lines = f.readlines()
            start_idxs = []
            for i, line in enumerate(lines):
                # lines[i] = normalizeString(line)
                # lines[i] = unicodeToAscii(line.lower().strip())
                # lines[i] = re.sub(r"([.!?])", r" \1", lines[i])
                # lines[i] = re.sub(r"[^a-zA-Z.!?]+", r" ", lines[i])
                while lines[i][0] == ' ' or lines[i][0] == "'":
                    if len(lines[i]) > 2:
                        lines[i] = lines[i][1:]
                if 'Permutations of ' in lines[i]:
                    start_idxs.append(i)
            start_idxs.append(i+1)
            for i, idx in enumerate(start_idxs):
                if idx != start_idxs[-1]:
                    key = lines[idx][17:-3]
                    while key[0] == ' ' or key[0] == "'":
                        if len(key) > 2:
                            key = key[1:]
                    if "==========" not in lines[start_idxs[i+1]-1]:
                        continue
                    assert("==================" in lines[start_idxs[i+1]-1] )
                    augments += lines[idx+2:start_idxs[i+1]-1]
    etime = time.time() - stime 
    print("Took", etime, "to parse all augmented data!")
    # print(augments)
    return augments
    