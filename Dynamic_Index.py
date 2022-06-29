"""
作者：LJH
日期：2021年08月10日
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


def Dynamic_N_Gram_DB(data, rindex, counts, num, threshold):
    pieceNum = num
    for colindex in range(len(data.T)):
        N_Gram = 1
        cols = []
        while(True):
            try:
                hashint = bootlegHash(data.T[colindex+N_Gram-1])
                if hashint == 0:
                    break
                cols.append(hashint)
            except IndexError:
                break
            fp = tuple(cols)
            numMatches = counts[N_Gram-1][fp]
            if numMatches < threshold or N_Gram == 4:
                if fp in rindex:
                    if pieceNum in rindex[fp]:
                        rindex[fp][pieceNum].append(colindex)
                    else:
                        rindex[fp][pieceNum] = [colindex]
                else:
                    rindex[fp] = {}
                    rindex[fp][pieceNum] = [colindex]
                break
            N_Gram+=1
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
counts = []
filelist = 'cfg_files/db.list'
outfile = 'experiments/indices/Dynamic_N_GRAM_ALL(5k).pkl'
threshold = 5000
for i in range(1,5):
    print("LOADING {}".format(i))
    count_file = 'experiments/indices/N_GRAM_{}_COUNT.pkl'.format(i)
    with open(count_file, 'rb') as f:
        counts.append(pickle.load(f))
        f.flush()
        f.close()
with open(filelist, 'r') as f:
    failed = []
    for i, curfile in enumerate(f):
        curfile = curfile.strip().strip('\n')
        print("Processed:", i)
        try:
            num = curfile.split('/')[-1][0]
            if(num == 'd'):
                data, _ = getTotalBscore(curfile)
            else:
                with open(curfile, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
            rindex = Dynamic_N_Gram_DB(data, rindex, counts, i, threshold)
        except:
            failed.append(curfile)
    print(failed)
with open(outfile,'wb') as f:
    pickle.dump(rindex,f)
end = time.process_time()
print("the running time is {} seconds".format(end-start))
