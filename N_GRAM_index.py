"""
作者：LJH
日期：2021年08月05日
"""
import numpy as np
import matplotlib.pyplot as plt
from mido import MidiFile, tick2second
from pretty_midi import PrettyMIDI
import pickle
import joblib
import subprocess
import os
import os.path
import random
from ExtractBootlegFeatures1 import *
import multiprocessing
import time

def bootlegHash(arr):
    bitstring = ""
    for i in range(len(arr)):
        if arr[i]==1:
            bitstring+="1"
        else:
            bitstring +="0"
    bitstring = bitstring+"00"
    hashint = int(bitstring, 2)
    hashint = np.uint64(hashint)
    return hashint

def getTotalBscore(bscore_file):
    bscore_array = []
    with open (bscore_file,'rb') as f:
        bscore_array = pickle.load(f)
    f.close()
    total_bscore = np.array([]).reshape(62,0)
    page_array = []
    for page in bscore_array:
        total_page = np.array([]).reshape(62,0)
        for num in page:
            col = np.array(decodeColumn(num)).reshape(62,-1)
            total_page = np.concatenate((total_page,col),axis=1)
        total_bscore = np.concatenate((total_bscore,total_page),axis=1)
        page_array.append(total_page)
    return total_bscore,page_array

def decodeColumn(num):
    col = []
    for i in range(62):
        col.insert(0,num%2)
        num = int(num/2)
    return col

def Singular_DB(data, rindex):
    for colindex in range(len(data.T)):
        col = data.T[colindex]
        hashint = bootlegHash(col)
        if hashint == 0:
            continue
        pieceStr = curfile.split('/')[-1][:-4]

        if hashint in rindex:
            if pieceStr in rindex[hashint]:
                rindex[hashint][pieceStr].append(colindex)
            else:
                rindex[hashint][pieceStr]=[colindex]
        else:
            rindex[hashint]={}
            rindex[hashint][pieceStr]=[colindex]
    return rindex

def N_Gram_DB(data, rindex, N_Gram = 3):
    for colindex in range(len(data.T)):
        cols = []
        try:
            for i in range(N_Gram):
                cols.append(data.T[colindex+i])
        except IndexError:
            continue
        fp = []
        equals_Zero = True
        for column in cols:
            hashint = bootlegHash(column)
            fp.append(hashint)
            if hashint != 0:
                equals_Zero = False
        if equals_Zero == True:
            continue
        fp = tuple(fp)
        pieceStr = curfile.split('/')[-1][:-4]
        if fp in rindex:
            if pieceStr in rindex[fp]:
                rindex[fp][pieceStr].append(colindex)
            else:
                rindex[fp][pieceStr]=[colindex]
        else:
            rindex[fp]={}
            rindex[fp][pieceStr]=[colindex]
    return rindex

def Dynamic_N_Gram_DB(data, rindex, rindex_original, threshold_function):
    for colindex in range(len(data.T)):
        first_col = data.T[colindex]
        first_fp = bootlegHash(first_col)
        hits = len(rindex_original[fp].keys())
        N_Gram = int(threshold_function(hits))
        cols = []
        try:
            for i in range(N_Gram):
                cols.append(data.T[colindex+i])
        except IndexError:
            continue
        fp = []
        equals_Zero = True
        for col in cols:
            hashint = bootlegHash(col)
            fp.append(hashint)
            if hashint != 0:
                equals_Zero = False
        if equals_Zero == True:
            continue
        fp = tuple(fp)
        pieceStr = curfile.split('/')[-1][:-4]
        if fp in rindex:
            if pieceStr in rindex[fp]:
                rindex[fp][pieceStr].append(colindex)
            else:
                rindex[fp][pieceStr]=[colindex]
        else:
            rindex[fp]={}
            rindex[fp][pieceStr]=[colindex]
    return rindex

def createCountFile(outfile, rindex):
    rindex_count = {}
    out = "experiments/indices/Dynamic_N_GRAM_COUNT.pkl"
    for key in rindex:
        count = 0
        subDict = rindex[key]
        for key1 in subDict:
            count += len(subDict[key1])
        rindex_count[key] = count
    with open (outfile,"wb") as f:
        pickle.dump(rindex_count,f)
    f.close()

start = time.process_time()
rindex = {}
fpMap = {}
rindex_original = {}
filelist = 'cfg_files/db.list'
N=3
outfile = 'experiments/indices/N_Gram_{}_ALL.pkl'.format(N)
mode = "N_GRAM"
threshold_function = lambda x: 3*x/10000 + 1
with open(filelist, 'r') as f:
    failed = []
    for curfile in f:
        curfile = curfile.strip().strip('\n')
        #print("Processed:", count)
        try:
            num = curfile.split('/')[-1][0]
            if(num == 'd'):
                data, _ = getTotalBscore(curfile)
            else:
                with open(curfile, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
            if mode == "SINGULAR":
                rindex = Singular_DB(data, rindex)
            elif mode == "N_GRAM":
                rindex = N_Gram_DB(data, rindex, N_Gram=N)
            elif mode == "DYNAMIC_N_GRAM":
                rindex = Dynamic_N_Gram_DB(data, rindex, rindex_original, threshold_function)
        except:
            failed.append(curfile)
    print(failed)
with open(outfile,'wb') as f:
    pickle.dump(rindex,f)
if mode == "N_GRAM":
    outfile = os.path.splitext(outfile)[0][:-3]+"COUNT.pkl"
    createCountFile(outfile, rindex)
end = time.process_time()
print("the running time is {} seconds".format(end-start))

start = time.process_time()
rindex = {}
fpMap = {}
N=4
outfile = 'experiments/indices/N_Gram_{}_ALL.pkl'.format(N)
mode = "N_GRAM"
threshold_function = lambda x: 3*x/10000 + 1
with open(filelist, 'r') as f:
    failed = []
    for curfile in f:
        curfile = curfile.strip().strip('\n')
        #print("Processed:", count)
        try:
            num = curfile.split('/')[-1][0]
            if(num == 'd'):
                data, _ = getTotalBscore(curfile)
            else:
                with open(curfile, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
            if mode == "SINGULAR":
                rindex = Singular_DB(data, rindex)
            elif mode == "N_GRAM":
                rindex = N_Gram_DB(data, rindex, N_Gram=N)
            elif mode == "DYNAMIC_N_GRAM":
                rindex = Dynamic_N_Gram_DB(data, rindex, rindex_original, threshold_function)
        except:
            failed.append(curfile)
    print(failed)
with open(outfile,'wb') as f:
    pickle.dump(rindex,f)
if mode == "N_GRAM":
    outfile = os.path.splitext(outfile)[0][:-3]+"COUNT.pkl"
    createCountFile(outfile, rindex)
end = time.process_time()
print("the running time is {} seconds".format(end-start))