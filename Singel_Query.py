"""
作者：LJH
日期：2021年08月10日
"""
import numpy as np
import copy
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import cv2
from skimage import filters, measure
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from sklearn.cluster import KMeans
import matplotlib.patches as mpatches
from scipy.signal import convolve2d
from scipy.spatial import KDTree
import seaborn as sns
import pickle
import librosa as lb
import time
import cProfile
import os
import os.path
import pyximport; pyximport.install()
import multiprocessing
from ExtractBootlegFeatures import *

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

def showHistograms(arr,numBins):
    plt.hist(arr,bins=numBins)
    plt.show()

def getOffsetDelta(bscore_query, rindex):
    offsetDict = {}
    for index in range(len(bscore_query.T)):
        hashkey = bootlegHash(bscore_query.T[index])
        if hashkey ==0 or not hashkey in rindex:
            continue
        rindex_hash = rindex[hashkey]
        #print(len(rindex_hash))
        for key in rindex_hash:
            #DONT USE np.ARRAY
            offset = [i - index for i in rindex_hash[key]]
            if key in offsetDict:
                offsetDict[key].extend(offset)
            else:
                offsetDict[key]=offset
    return offsetDict

def getOffsetDeltaNGram(bscore_query, rindex, N_Gram = 3):
    offsetDict = {}
    for index in range(len(bscore_query.T)):
        hashkey = []
        try:
            for i in range(N_Gram):
                hashkey.append(bootlegHash(bscore_query.T[index+i]))
        except IndexError:
            continue
        hashkey = tuple(hashkey)
        if hashkey == 0 or not hashkey in rindex:
            continue
        rindex_hash = rindex[hashkey]
        for key in rindex_hash:
            #DONT USE np.ARRAY
            offset = [i - index for i in rindex_hash[key]]
            if key in offsetDict:
                offsetDict[key].extend(offset)
            else:
                offsetDict[key]=offset
    return offsetDict

def getOffsetDeltaDynamicStaticN_GRAM(bscore_query, rindex, Max_N = 4):
    offsetDict = {}
    for index in range(len(bscore_query.T)):
        N_Gram = 1
        while(N_Gram <= Max_N):
            hashkey = []
            try:
                for i in range(N_Gram):
                    hashkey.append(bootlegHash(bscore_query.T[index+i]))
            except IndexError:
                break
            hashkey = tuple(hashkey)
            if hashkey in rindex:
                rindex_hash = rindex[hashkey]
                for key in rindex_hash:
                    #DONT USE np.ARRAY
                    offset = [i - index for i in rindex_hash[key]]
                    if key in offsetDict:
                        offsetDict[key].extend(offset)
                    else:
                        offsetDict[key]=offset
                break
            N_Gram+=1
    return offsetDict



def rankHistograms(offsetDict, bin_size=5):
    histograms = {}
    pieceScores = []
    for key in offsetDict:
        h = offsetDict[key]
        maxh = max(h)
        minh = min(h)
        if (maxh > minh + bin_size):
            hist = [0 for i in range(int((maxh - minh) / bin_size) + 1)]
            for i in h:
                hist[int((i - minh) / bin_size)] += 1
            score = max(hist)
            pieceScores.append((key, score))
            histograms[key] = (h, len(set(h)))

    pieceScores = sorted(pieceScores, key=lambda x: x[1], reverse=True)
    return pieceScores, histograms

def displayHist(pieceScores, pieceNum):
    pieceScores1 = sorted(pieceScores, key = lambda x:x[0][1:])
    print(pieceScores1[pieceNum-1][0])
    h,numBins = histograms[pieceScores1[pieceNum-1][0]]
    showHistograms(h,numBins)


def processSingleQuery(imagefile, rindex, mode="N_GRAM"):
    profileStart = time.time()

    # Get Bootleg Score
    bscore_query = processQuery(imagefile)
    print("process image time:",time.time()-profileStart)
    # Generate and rank histograms
    if mode == "NORMAL":
        offsetDict = getOffsetDelta(bscore_query, rindex)
    elif mode == "N_GRAM":
        offsetDict = getOffsetDeltaNGram(bscore_query, rindex, N_Gram=3)
    elif mode == "Dynamic_Static":
        offsetDict = getOffsetDeltaDynamicStaticN_GRAM(bscore_query, rindex=rindex, Max_N=4)

    pieceScores, histograms = rankHistograms(offsetDict)

    # Profile & save to file
    profileEnd = time.time()
    profileDur = profileEnd - profileStart
    # print("total time:", profileDur)

    # return pieceScores, histograms
    return pieceScores, profileDur


pklfile = "piece_to_num.pkl"
with open (pklfile,'rb')as f:
    piece_to_num = pickle.load(f)
pickle_file = 'experiments/indices/Dynamic_N_GRAM_ALL(2k).pkl'
print('start loading')
with open(pickle_file, 'rb') as f:
    rindex = pickle.load(f)
imagefile = "p018.png"
input("Press enter to continue:")
pieceScores, histograms = processSingleQuery(imagefile, rindex, "Dynamic_Static")
for i in range(20):
    print(piece_to_num[pieceScores[i][0]],' ',pieceScores[i][1])
print(len(pieceScores))

