from mmdet.apis import init_detector, inference_detector
import numpy as np
from numpy.matlib import repmat
from PIL import Image, ImageFilter, ImageChops
import cv2
from sklearn.cluster import KMeans
from scipy.signal import convolve2d
import os
import os.path
import json
from IPython import embed
import time
from utils import opt
from draw import drawDetectResult, visualizeEstStaffLines
from skimage.filters import threshold_sauvola


def sauvola(img, windowsize):
    sauvolaImage = threshold_sauvola(img, window_size=windowsize, k=0.25)
    binary_sauvola = (img > sauvolaImage) * 255
    binary_sauvola = binary_sauvola.astype(img.dtype)
    return binary_sauvola


def detect(model, imagefile):
    # build the model from a config file and a checkpoint file
    img = cv2.imread(imagefile)
    # img = sauvola(img, 35)
    # cv2.imwrite('sauvola.jpg', img)
    result = inference_detector(model, img)

    jsonData = model.save_result(img,
                                 result,
                                 score_thr=0.25,
                                 save_pic=False,
                                 out_file=None)
    # model.show_result(img, result, score_thr=0.25, out_file='/home/cy/llj/SheetMidiRetrieval/result/photo_0/test.jpg')

    return jsonData


# Pre-processing
def removeBkgdLighting(pimg, filtsz=5, thumbnailW=100, thumbnailH=100):
    '''
    去除图片中的背景光线，消除光线不均匀的影响
    '''
    tinyimg = pimg.copy()
    # thumbnail会覆盖原对象，进行等比例缩放
    tinyimg.thumbnail([thumbnailW, thumbnailH])  # resize to speed up
    # 高斯模糊
    shadows = tinyimg.filter(ImageFilter.GaussianBlur(filtsz)).resize(
        pimg.size)
    # remove background lighting from the image
    result = ImageChops.invert(ImageChops.subtract(shadows, pimg))
    return result


def getPenalizedCombFilter(linesep):
    '''
    构造1D梳状滤波器，用于估计五线谱间距
    length: 
        (linesep * 5)，5代表五线谱的5根线
    组成：
        1、对linesep的0.5、1.5、2.5、3.5、4.5倍索引及其前后1位，置1.0 （正峰值）
        2、对linesep的1、2、3、4倍索引及其前后1位，以及索引0、1、len(linesep)-1处，置-1.0（负峰值）
        3、其余位置，置0
    特点：三连1和三连-1穿插，中间由0补齐，间距由linesep控制
    意义：
        1、linesep的取值决定了滤波器的长度，以及相邻1、-1的间距（间距为linesep * 0.5 - 2)
        2、卷积结果中，每一列的的最大值反应了五线谱的位置；对于不同的linesep，最大值越大，标明当前linesep越接近五线谱的线间距离
    举例：
        假设linesep=16且正好为五线谱线间距，则滤波器filt的长度为16*5=80，滤波器在(7、8、9)、(23、24、25)、(39、40、41)、(55、56、57)、(71、72、73)处为1.0，
        在(0、1)、(15、16、17)、(31、32、33)、(47、48、49)、(63、64、65)、(79)处
        为-1.0
        假设五线谱理想（无符号），那么在包含五线谱的列中必有x个最大值，x为当前图片中五线谱的个数
        例如滤波器滑窗到最上方的五线谱，索引8、24、40、56、72刚好对应五线谱的横线位置（对于灰度翻转图片，白色的五线代表像素255），索引16、32、48、64刚好对应相邻五线谱的中心（黑底代表像素0），此时取得最大卷积（卷积值与线高相关）
        只要linesep小于或大于五线谱线间距，当前列获得的最大卷积必然小于上述情况，因此我们能用linesep来估计五线谱的谱间距
    '''
    filt = np.zeros(int(np.round(linesep * 5)))

    # positive spikes
    for i in range(5):
        offset = int(np.round(.5 * linesep + i * linesep))
        filt[offset - 1:offset + 2] = 1.0

    # negative spikes
    for i in range(6):
        center = int(np.round(i * linesep))
        startIdx = max(center - 1, 0)
        endIdx = min(center + 2, len(filt))
        filt[startIdx:endIdx] = -1.0

    return filt


def estimateLineSep(pim, ncols, lrange, urange, delta):
    '''
    五线谱间距估算：
    1、将图像按照固定的列数等间距分割（不包括最后一列）
    2、计算分割列的每一行像素值的中位数
    3、将行像素值的中位数与一组梳状滤波器进行卷积，这些梳状滤波器对应于不同的谱间距
    4、找到所有分割列中累积响应最强的梳状滤波器，即为所估计的五线谱间距
    '''
    # break image into columns, calculate row medians for inner columns (exclude outermost columns)
    img = 255 - np.array(pim)
    imgHeight, imgWidth = img.shape
    rowMedians = np.zeros((imgHeight, ncols))
    colWidth = imgWidth // (ncols + 2)
    for i in range(ncols):
        rowMedians[:,
                   i] = np.median(img[:,
                                      (i + 1) * colWidth:(i + 2) * colWidth],
                                  axis=1)

    # print('rowMedians shape: ', rowMedians.shape)

    # apply comb filters
    lineseps = np.arange(lrange, urange, delta)
    responses = np.zeros((len(lineseps), imgHeight, ncols))
    for i, linesep in enumerate(lineseps):
        filt = getPenalizedCombFilter(linesep).reshape((-1, 1))
        response = convolve2d(rowMedians, filt, mode='same')
        responses[i, :, :] = response
        # np.savetxt('{}responses_{}.txt'.format(resultFile, i), response)
    # find comb filter with strongest response
    scores = np.sum(np.max(responses, axis=1), axis=1)
    bestIdx = np.argmax(scores)
    estLineSep = lineseps[bestIdx]

    return estLineSep, scores


def calcResizedDimensions(pim, estimatedLineSep, desiredLineSep):
    '''
    调整图片尺寸大小，使得谱线间距为10
    '''
    curH, curW = pim.height, pim.width
    scale_factor = 1.0 * desiredLineSep / estimatedLineSep
    targetH = int(curH * scale_factor)
    targetW = int(curW * scale_factor)
    return targetH, targetW, scale_factor


# Staff Line Features
def getNormImage(img):
    '''
    归一化图像到(-1, 1)的范围
    '''
    X = 1 - np.array(img) / 255.0
    return X


def morphFilterRectangle(arr, kernel_height, kernel_width):
    '''
    形态过滤器,用于过滤特定大小与方向的像素
    针对图像的白色（高亮）区域而言
    先腐蚀再膨胀为开操作，开操作一般会平滑物体的轮廓，断开较窄的狭颈并消除细的突出物
    代码作用：
    1、过滤水平线外全部内容。先腐蚀过滤非水平线，再膨胀将水平线的长度还原
    2、获取水平粗横梁。先腐蚀过滤水平线（线高小于3），在膨胀平滑水平粗横梁
    '''
    kernel = np.ones((kernel_height, kernel_width), np.uint8)
    result = cv2.erode(arr, kernel, iterations=1)
    result = cv2.dilate(result, kernel, iterations=1)
    return result


# 41，3，0.9
def isolateStaffLines(arr, kernel_len, notebarfilt_len, notebar_removal):
    '''
    提取五线谱
    1、使用短（1像素高）脂肪形态过滤器对图像进行腐蚀和膨胀，过滤水平线外全部内容，过滤结束后，图片将只剩下五线谱和音符的水平横梁
    2、删除符的水平横梁。根据横梁比五线谱粗的特性，再次通过形态过滤器隔离
    3、将（1）的结果减去（2），删除符的水平横梁
    '''
    lines = morphFilterRectangle(arr, 1,
                                 kernel_len)  # isolate horizontal lines
    notebarsOnly = morphFilterRectangle(lines, notebarfilt_len,
                                        1)  # isolate thick notebars
    result = np.clip(lines - notebar_removal * notebarsOnly, 0,
                     None)  # subtract out notebars
    return result, notebarsOnly


def getCombFilter(lineSep):
    # generate comb filter of specified length
    # e.g. if length is 44, then spikes at indices 0, 11, 22, 33, 44
    # e.g. if length is 43, then spikes at 0 [1.0], 10 [.25], 11 [.75], 21 [.5], 22 [.5], 32 [.75], 33 [.25], 43 [1.0]
    stavelen = int(np.ceil(4 * lineSep)) + 1
    combfilt = np.zeros(stavelen)
    for i in range(5):
        idx = i * lineSep
        idx_below = int(idx)
        idx_above = idx_below + 1
        remainder = idx - idx_below
        combfilt[idx_below] = 1 - remainder
        if idx_above < stavelen:
            combfilt[idx_above] = remainder
    return combfilt, stavelen


# ncols=10, lrange=8.5, urange=11.75, delta=0.25
def computeStaveFeatureMap(img, ncols, lrange, urange, delta):
    '''
    计算五线谱特征图
    输入img: 仅保留五线的图片
    1、将图像按照固定的列数等间距分割
    2、计算分割列的每一行像素值的和
    3、将行像素值的和与一组梳状滤波器进行卷积，这些梳状滤波器对应于不同的谱间距
    注：由于我们已经对图像执行了行间归一化，因此梳状滤波器的大小范围比图像预处理阶段要窄得多

    Returns:
        featmap:五线谱特征图
        stavelens:五线谱高度数组
        colWidth:五线谱按列切割后，每列的宽度
    '''
    # break image into columns, calculate row medians
    imgHeight, imgWidth = img.shape
    rowSums = np.zeros((imgHeight, ncols))
    colWidth = int(np.ceil(imgWidth / ncols))
    for i in range(ncols):
        startCol = i * colWidth
        endCol = min((i + 1) * colWidth, imgWidth)
        rowSums[:, i] = np.sum(img[:, startCol:endCol], axis=1)
    # rowSums: (imgHeight, ncols=10)

    # apply comb filters
    # (8.5, 11.75, 0.25)
    lineseps = np.arange(lrange, urange, delta)
    # 4 * 11.5 + 1 = 47
    maxFiltSize = int(np.ceil(4 * lineseps[-1])) + 1
    # (13, imgHeight - 46, 10)
    featmap = np.zeros((len(lineseps), imgHeight - maxFiltSize + 1, ncols))
    # (13,)
    stavelens = np.zeros(len(lineseps), dtype=int)
    for i, linesep in enumerate(lineseps):
        # filt: (4 * linesep + 1, )
        # stavelen=4 * linesep + 1
        filt, stavelen = getCombFilter(linesep)
        # (47, 1)
        padded = np.zeros((maxFiltSize, 1))
        padded[0:len(filt), :] = filt.reshape((-1, 1))
        # rowSums与padded互相关计算
        featmap[i, :, :] = convolve2d(rowSums,
                                      np.flipud(np.fliplr(padded)),
                                      mode='valid')
        stavelens[i] = stavelen
    # 卷积操作后, 得到K * (H - maxFiltSize + 1) * C的特征张量，其中K是梳状滤波器的数量，H是图像的高度（以像素为单位），C是列数
    return featmap, stavelens, colWidth


# Notehead Detection
# 解析note位置
def paraseJson(dic, scale_factor):
    notes = []
    braces = []
    clefs = []
    keys = []
    accs = []
    others = []
    for v in dic.values():
        prefix = v['name'][:5]
        suffix = v['name'][-4:]
        confidence = float(v['confidence'])
        a1, a2, b1, b2 = resizeSymbols(v, scale_factor)
        if prefix == 'noteh':
            # 先y后x
            if suffix == 'Line':
                # 在线上 0
                notes.append((a2, a1, b2, b1, confidence, int(0)))
            else:
                # 在线间 1
                notes.append((a2, a1, b2, b1, confidence, int(1)))
        elif prefix == 'brace':
            braces.append((int(a2), int(a1), int(b2), int(b1), confidence))
        elif prefix == 'clefG':
            # 高音谱号0
            clefs.append(
                (int(a2), int(a1), int(b2), int(b1), confidence, int(0)))
        elif prefix == 'clefF':
            # 低音谱号1
            clefs.append(
                (int(a2), int(a1), int(b2), int(b1), confidence, int(1)))
        elif prefix == 'keySh':
            # 升号 0
            keys.append(
                (int(a2), int(a1), int(b2), int(b1), confidence, int(0)))
        elif prefix == 'keyFl':
            # 降号 1
            keys.append(
                (int(a2), int(a1), int(b2), int(b1), confidence, int(1)))
        elif prefix == 'accid':
            acc = v['name'][:11]
            if acc == 'accidentalS':
                # 升号 0
                accs.append(
                    (int(a2), int(a1), int(b2), int(b1), confidence, int(0)))
            elif acc == 'accidentalN':
                # 还原号 2
                accs.append(
                    (int(a2), int(a1), int(b2), int(b1), confidence, int(2)))
            elif acc == 'accidentalF':
                # 降号 1
                accs.append(
                    (int(a2), int(a1), int(b2), int(b1), confidence, int(1)))
        else:
            others.append((int(a2), int(a1), int(b2), int(b1), confidence))
    return notes, clefs, keys, accs, braces, others


def filterAllSymbols(notes, clefs, keys, accs, braces, others, vpadding):
    # 去重
    if len(notes) > 1:
        notes = mergeOverlapRect(notes)
    if len(clefs) > 1:
        clefs = mergeOverlapRect(clefs)
    if len(keys) > 1:
        keys = mergeOverlapRect(keys)
    if len(accs) > 1:
        accs = mergeOverlapRect(accs)
    if len(braces) > 1:
        braces = mergeOverlapRect(braces)
    if len(others) > 1:
        others = mergeOverlapRect(others)

    # print('clefs: ', clefs)
    # 去除异常大小符头
    # clefs = removeUnNormalSymbol(clefs, int(vpadding * 3))
    # 去除边缘位置符头
    clefs = np.array(clefs)
    clef_y_top = [clef[0] for clef in clefs]
    clef_y_bottom = [clef[2] for clef in clefs]

    # print('clef_y: ', clef_y)
    clef_y_min = min(clef_y_top) - vpadding * 4
    clef_y_max = max(clef_y_bottom) + vpadding * 8
    notes_len = len(notes)
    notes = list(notes)
    for i in range(notes_len - 1, -1, -1):
        note = notes[i]
        if note[0] < clef_y_min or note[2] > clef_y_max:
            notes.pop(i)

    return np.array(notes), np.array(clefs), np.array(keys), np.array(
        accs), np.array(braces), np.array(others)


def removeUnNormalSymbol(symbols, padding):
    symLens = [(symbol[2] - symbol[0]) for symbol in symbols]
    symbols = list(symbols)
    for i in range(len(symLens) - 1, -1, -1):
        symLen = symLens[i]
        if symLen < padding:
            symbols.pop(i)
    symbols = np.array(symbols)
    return symbols


def mergeOverlapRect(bboxes):
    ''''''
    # if len(bboxes) == 0: return []
    for i in range(len(bboxes)):
        if i == (len(bboxes) - 1): break
        for j in range(i + 1, len(bboxes)):
            rect1, rect2 = bboxes[i], bboxes[j]
            isoverlap = isRectangleOverlap(rect1, rect2)
            if len(rect1) == 6:
                flag1, flag2 = rect1[-1], rect2[-1]
                conf1, conf2 = rect1[-2], rect2[-2]
                if isoverlap:
                    min_y = min(rect1[0], rect2[0])
                    min_x = min(rect1[1], rect2[1])
                    max_y = max(rect1[2], rect2[2])
                    max_x = max(rect1[3], rect2[3])
                    flag = flag1 if conf1 > conf2 else flag2
                    conf = conf1 if conf1 > conf2 else conf2
                    rect = [min_y, min_x, max_y, max_x, conf, flag]
                    bboxes[i], bboxes[j] = rect, rect
            else:
                if isoverlap:
                    min_y = min(rect1[0], rect2[0])
                    min_x = min(rect1[1], rect2[1])
                    max_y = max(rect1[2], rect2[2])
                    max_x = max(rect1[3], rect2[3])
                    conf1, conf2 = rect1[-1], rect2[-1]
                    conf = conf1 if conf1 > conf2 else conf2
                    rect = [min_y, min_x, max_y, max_x, conf]
                    bboxes[i], bboxes[j] = rect, rect
        bboxes = np.array(bboxes)
        rects = np.unique(bboxes, axis=0)

    return rects


def isRectangleOverlap(rec1, rec2):
    # 判断相交
    iou = bb_intersection_over_union(rec1, rec2)
    if iou < 0.15:
        return False
    else:
        return True


def bb_intersection_over_union(boxA, boxB):
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def resizeSymbols(v, scale_factor):
    a1 = int(v['leftTopX']) * scale_factor
    a2 = int(v['leftTopY']) * scale_factor
    b1 = int(v['right_bottomX']) * scale_factor
    b2 = int(v['right_bottomY']) * scale_factor
    return a1, a2, b1, b2


def getNoteheadInfo(bboxes):
    '''
    获取符头的大小与位置信息

    Args:
        bboxes(tuple,list): 符头左上和右下坐标
    Returns:
        nhlocs(list): 符头中心坐标(y,x)数组
        nhlen_est(int): 符头平均高度
        nhwidth_est(int): 符头平均宽度
    '''
    nhlocs = [(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]))
              for bbox in bboxes]
    nhlens = [(bbox[2] - bbox[0]) for bbox in bboxes]
    nhwidths = [(bbox[3] - bbox[1]) for bbox in bboxes]
    nhlen_est = int(np.ceil(np.mean(nhlens)))
    nhwidth_est = int(np.ceil(np.mean(nhwidths)))
    return nhlocs, nhlen_est, nhwidth_est


def getOtherSymbolsCenter(clefs, keys, accs, braces, others):
    '''
    获取其他符号中心坐标
    '''
    clocs = [(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]))
             for bbox in clefs]
    klocs = [(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]))
             for bbox in keys]
    alocs = [(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]))
             for bbox in accs]
    olocs = [(.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3]))
             for bbox in others]
    return clocs, klocs, alocs, olocs


def getBracesLength(braces):
    '''
    获取花括号的长度
    '''
    brLens = [(brace[2] - brace[0]) for brace in braces]
    brlen_est = int(np.ceil(np.mean(brLens)))
    return brlen_est


# Infer Note Values
# 1. deltaRowMax=50 globalOffset=-20
# 2. deltaRowMax=15 globalOffset=nhRowOffsets-20
def getEstStaffLineLocs(featmap,
                        nhlocs,
                        stavelens,
                        colWidth,
                        deltaRowMax,
                        globalOffset=0):
    '''
    确定每个符头所在列的五线谱的起始和终止高度
    Args:
        featmap(numpy):五线谱特征图 K * (H - maxFiltSize + 1) * C
        nhlocs(list): 符头中心坐标(y,x)数组
        stavelens(list):五线谱高度数组
        colWidth:(int)五线谱按列切割后，每列的宽度
    Returns:
        preds:预测结果数组(谱线起始高度、终止高度、符头中心坐标x值、符头中心坐标y值、局部梳状滤波器的索引)
        sfiltlen:全部符头对应的五线谱间距的中位数（取整）
    '''
    preds = []
    if np.isscalar(globalOffset):
        globalOffset = [globalOffset] * len(nhlocs)  # [-20 -20 ... -20]
    for i, nhloc in enumerate(nhlocs):
        # 对每一个符头
        r = int(np.round(nhloc[0]))  # y值
        c = int(np.round(nhloc[1]))  # x值
        # 高度截取
        # 初始: rupper=r+31, rlower=r-70
        # 调整: rupper=r+16+globalOffset[i], rlower=r-15+globalOffset[i],
        # 即: rupper=符头对应聚类中心y坐标-4， rlower=符头对应聚类中心y坐标-35
        rupper = min(r + deltaRowMax + 1 + globalOffset[i], featmap.shape[1])
        rlower = max(r - deltaRowMax + globalOffset[i], 0)
        featmapIdx = c // colWidth  # featmap对应的第3维
        # (K, rupper-rlower) 符头局部特征图
        regCurrent = np.squeeze(featmap[:, rlower:rupper, featmapIdx])
        # 获取局部特征图中最大值所在位置（mapidx, roffset）
        # mapidx为对应梳状滤波器的索引；roffset为卷积值，反应了在局部特征图中最大响应的高度
        mapidx, roffset = np.unravel_index(regCurrent.argmax(),
                                           regCurrent.shape)
        rstart = rlower + roffset
        # 一行五线谱间距
        rend = rstart + stavelens[mapidx] - 1
        # 谱线所在行起始高度、谱线所在行终止高度、符头中心坐标x值、符头中心坐标y值、局部梳状滤波器的索引
        preds.append((rstart, rend, c, r, mapidx))

    # 全部符头对应的五线谱间距的中位数（取整）
    sfiltlen = int(np.round(np.median([stavelens[tup[4]] for tup in preds])))
    return preds, sfiltlen


# minNumStaves = 2, maxNumStaves = 12, minStaveSeparation = 60.0
def estimateStaffMidpoints(preds, clustersMin, clustersMax, threshold):
    # clustersMin = 2, clustersMax = 12, threshold = 60.0
    '''
    用k-Means算法将所有符头的高度进行聚类，获取聚类中心

    Args:
        preds(list):预测结果数组(谱线起始高度、终止高度、符头中心坐标x值、符头中心坐标y值、局部梳状滤波器的索引)
        clustersMin(int): k-Means聚类的最小簇数
        clustersMax(int): k-Means聚类的最大簇数
        threshold(float): 相邻五线谱最小间距, 作为k-Means算法的终止条件
    Returns:
        staffMidpts(list): 聚类中心坐标数组（一维），升序排列
    '''
    r = np.array([.5 * (tup[0] + tup[1])
                  for tup in preds])  # midpts of estimated stave locations
    models = []
    for numClusters in range(clustersMin, clustersMax + 1):
        kmeans = KMeans(n_clusters=numClusters, n_init=1,
                        random_state=0).fit(r.reshape(-1, 1))
        # 升序排列聚类中心
        sorted_list = np.array(sorted(np.squeeze(kmeans.cluster_centers_)))
        mindiff = np.min(sorted_list[1:] - sorted_list[0:-1])
        # 样本数numClusters>最小簇数clustersMin, 且相邻聚类中心<相邻五线谱最小间距threshold时，退出循环
        if numClusters > clustersMin and mindiff < threshold:
            # print('numClusters, mindiff: ', numClusters, mindiff)
            break
        models.append(kmeans)
    staffMidpts = np.sort(np.squeeze(models[-1].cluster_centers_))
    # print('staffMidpts: ', staffMidpts)
    return staffMidpts


def assignNoteheadsToStaves(nhlocs, staveCenters):
    '''
    将符头分配给最近的全局五线谱系统
    
    Args:
        nhlocs(list): 符头中心坐标(y,x)数组
        staveCenters(list): 聚类中心y坐标数组（一维），升序排列
    Returns:
        staveIdxs(numpy): 每个符头对应的聚类中心y坐标数组的索引数组
        offsets(numpy): [符头对应的聚类中心y坐标 - 符头中心y坐标]
    '''
    # (len(staveCenters), len(nhlocs)) 每行为符头中心y坐标数组
    nhrows = repmat([tup[0] for tup in nhlocs], len(staveCenters), 1)
    # (len(staveCenters), len(nhlocs)) 每列为聚类中心y坐标数组
    centers = repmat(staveCenters.reshape((-1, 1)), 1, len(nhlocs))
    # (len(nhlocs), )
    # 每个符头对应的聚类中心y坐标数组的索引
    staveIdxs = np.argmin(np.abs(nhrows - centers), axis=0)
    # 符头对应的聚类中心y坐标 - 符头中心y坐标
    offsets = staveCenters[staveIdxs] - nhrows[
        0, :]  # row offset between note and staff midpoint
    return staveIdxs, offsets


def assignOtherSymbolsToStaves(staveCenters, clocs, klocs, alocs, olocs,
                               blocs):
    '''
    将其他符号分配给最近的全局五线谱系统

    Args:
        staveCenters(list): 符头聚类中心y坐标数组（一维），升序排列
        clocs(list): 谱号中心坐标
        klocs(list): 总升降音谱号中心坐标
        alocs(list): 局部升降还原音谱号中心坐标
        olocs(list): 其他符号中心坐标
        blocs(list): 小节线中心坐标
    '''
    # 获取中心y坐标, 然后重复len(nhlocs)行
    # 高音、低音谱号
    crows = repmat([clef[0] for clef in clocs], len(staveCenters), 1)
    # 升、降、还原符号
    krows = repmat([key[0] for key in klocs], len(staveCenters), 1)
    arows = repmat([acc[0] for acc in alocs], len(staveCenters), 1)
    orows = repmat([oth[0] for oth in olocs], len(staveCenters), 1)
    brows = repmat([bar[0] for bar in blocs], len(staveCenters), 1)
    # 将staveCenters转置成1列，然后重复对应符号数量列
    ccenters = repmat(staveCenters.reshape((-1, 1)), 1, crows.shape[1])
    kcenters = repmat(staveCenters.reshape((-1, 1)), 1, krows.shape[1])
    acenters = repmat(staveCenters.reshape((-1, 1)), 1, arows.shape[1])
    ocenters = repmat(staveCenters.reshape((-1, 1)), 1, orows.shape[1])
    bcenters = repmat(staveCenters.reshape((-1, 1)), 1, brows.shape[1])
    # 每个符头对应的聚类中心y坐标数组的索引
    cIdxs = np.argmin(np.abs(crows - ccenters), axis=0)
    kIdxs = np.argmin(np.abs(krows - kcenters), axis=0)
    aIdxs = np.argmin(np.abs(arows - acenters), axis=0)
    oIdxs = np.argmin(np.abs(orows - ocenters), axis=0)
    bIdxs = np.argmin(np.abs(brows - bcenters), axis=0)
    return cIdxs, kIdxs, aIdxs, oIdxs, bIdxs


def estimateNoteLabels(preds, notes):
    nhvals = []  # estimated note labels
    for i, (rstart, rend, c, r, filtidx) in enumerate(preds):
        # if a stave has height L, there are 8 stave locations in (L-1) pixel rows
        note_loc_type = notes[i][-1]
        staveMidpt = .5 * (rstart + rend)
        noteStaveLoc = -1.0 * (r - staveMidpt) * 8 / (rend - rstart)
        nhval_ceil = int(np.ceil(noteStaveLoc))
        nhval_floor = int(np.floor(noteStaveLoc))
        if nhval_ceil - nhval_floor == 0:
            nhval = nhval_ceil
        else:
            if note_loc_type == 0:
                # 在线上, nhval偶数
                if nhval_ceil % 2 == 0:
                    nhval = nhval_ceil
                else:
                    nhval = nhval_floor
            else:
                # 在线间, nhval奇数
                if nhval_ceil % 2 == 0:
                    nhval = nhval_floor
                else:
                    nhval = nhval_ceil
        # nhval = int(np.round(noteStaveLoc))
        nhvals.append(nhval)
    return nhvals


def filterNoteBeyondValue(nhvals, notes, staveIdxs, value=8):
    nhvals = np.array(nhvals)
    nhvals_idx = np.where(nhvals <= value)
    nhvals = nhvals[nhvals_idx]
    notes = notes[nhvals_idx]
    staveIdxs = staveIdxs[nhvals_idx]
    return nhvals, notes, staveIdxs


# Cluster staves & noteheads
# morphFilterVertLineLength=35, morphFilterVertLineWidth=10, maxBarlineWidth=40
def isolateBarlines(im, morphFilterVertLineLength, morphFilterVertLineWidth,
                    maxBarlineWidth):
    hkernel = np.ones((1, morphFilterVertLineWidth),
                      np.uint8)  # dilate first to catch warped barlines
    vlines = cv2.dilate(im, hkernel, iterations=1)
    vlines = morphFilterRectangle(vlines, morphFilterVertLineLength,
                                  1)  # then filter for tall vertical lines
    nonbarlines = morphFilterRectangle(vlines, 1, maxBarlineWidth)
    vlines = np.clip(vlines - nonbarlines, 0, 1)
    return vlines


# 符号分组排序
def get_symbols_group(symbols, symbolIdxs, isYFirst=True):
    symbol_groups = {}
    if len(symbolIdxs) == 0: return {}
    idx_min = symbolIdxs.min()
    idx_max = symbolIdxs.max()
    # print('symbolIdxs: ', symbolIdxs)
    for idx in range(idx_min, idx_max + 1):
        # 获取当前行的小节
        symbols_row = symbols[np.where(symbolIdxs == idx)]
        if len(symbols_row) == 0: continue
        if isYFirst:
            symbols_row_x = symbols_row[:, 1]
        else:
            symbols_row_x = symbols_row[:, 0]
        # 按照x值排序
        symbols_argsort = np.argsort(symbols_row_x)
        symbols_row = symbols_row[symbols_argsort]
        # 添加一列组号列
        symbols_row = np.insert(symbols_row,
                                len(symbols_row[0]),
                                int(idx),
                                axis=1)
        symbol_groups[int(idx)] = symbols_row
    return symbol_groups


def separatedNotesGroup(notes_group: dict, barlines_group: dict):
    group_nums = len(barlines_group)
    notes_barline_group = {}
    notes_group_addIdx = {}
    # print('barlines_group nums, notes_group nums: ', len(barlines_group), len(notes_group))

    for i in range(group_nums):
        nb_indexes = [-1]
        notes_barline = {}
        barlines, notes = barlines_group[i], notes_group[i]
        barlines_x = np.array([barline[0]
                               for barline in barlines]).reshape(1, -1)
        barlines_x = barlines_x.repeat(len(notes), axis=0)
        notes_x = np.array([note[1] for note in notes]).reshape(-1, 1)
        notes_x = notes_x.repeat(len(barlines), axis=1)
        nb_x = (notes_x - barlines_x).T
        for nb in nb_x:
            nb_index_arr = np.where(nb < 0)[0]
            if len(nb_index_arr) == 0: continue
            nb_index = nb_index_arr[-1]
            nb_indexes.append(nb_index)
        # embed()
        # 列末无小节线时，需补上构成小节
        if nb_indexes[-1] != len(notes) - 1:
            nb_indexes.append(len(notes) - 1)
        barline_index_row = np.zeros((len(notes), ), dtype=int)
        for j in range(len(nb_indexes) - 1):
            s, e = nb_indexes[j], nb_indexes[j + 1]
            notes_current_barline = notes[(s + 1):(e + 1)]
            r, _ = notes_current_barline.shape
            # 加上一列小节索引，从0开始
            index_col = np.full((r, 1), j)
            notes_current_barline = np.hstack(
                (notes_current_barline, index_col))
            notes_barline[j] = notes_current_barline
            # 原notes_group最后列加上索引
            barline_index_row[(s + 1):(e + 1)] = j
        notes_barline_group[i] = notes_barline
        barline_index_row = barline_index_row.reshape(-1, 1)
        notes_group_addIdx[i] = np.hstack((notes, barline_index_row))
    return notes_barline_group, notes_group_addIdx


def get_Scientific_pitch_notation(notes_barline_group, notes_group,
                                  clefs_group, keys_group, accs_group):
    '''
    计算科学音调记号表示的音高
    '''
    pitches = []
    rows = len(notes_barline_group)
    for i in range(rows):
        # 获取当前行谱号，默认只有一个
        clef_current = clefs_group[i][0][-2]
        # 当前行pitch
        pitches_current = []
        # key
        notes_barline = notes_barline_group[i]
        if i in keys_group:
            current_key = keys_group[i]
            current_key_num, current_key_class = len(
                current_key), current_key[0][-2]
            current_key_letters = getKeyGlobal(current_key_class,
                                               current_key_num)
            pitches_current = []
            for j in range(len(notes_barline)):
                notes = notes_barline[j]
                pitches_barline = []
                for note in notes:
                    pitch = cal_pitch_nonation(clef=clef_current,
                                               pitch=int(note[-3]))
                    pitch = changePitchGlobal(pitch, current_key_letters,
                                              current_key_class)
                    pitches_barline.append(pitch)
                pitches_current.append(pitches_barline)
        else:
            for j in range(len(notes_barline)):
                notes = notes_barline[j]
                pitches_barline = []
                for note in notes:
                    pitch = cal_pitch_nonation(clef=clef_current,
                                               pitch=int(note[-3]))
                    pitches_barline.append(pitch)
                pitches_current.append(pitches_barline)
        # accidental
        notes_row = notes_group[i]
        nlocs = np.array([[.5 * (bbox[0] + bbox[2]), .5 * (bbox[1] + bbox[3])]
                          for bbox in notes_row])
        nlocs_x = np.array([nloc[1] for nloc in nlocs])
        # (y1, x1, y2, x2, flag, row)
        # flag = 0,1,2 #、b、还原号
        if i in accs_group:
            accs = accs_group[i]
            for acc in accs:
                acc_type = acc[-2]
                aloc = np.array(
                    [.5 * (acc[0] + acc[2]), .5 * (acc[1] + acc[3])])
                aloc = aloc.reshape(1, -1)
                alocs = aloc.repeat(len(nlocs), axis=0)
                alocs_norm = np.linalg.norm(alocs - nlocs, axis=1).squeeze()
                # 过滤掉变音符号前的
                aloc_x = np.array([.5 * (acc[1] + acc[3])]).reshape(1, -1)
                alocs_x = aloc_x.repeat(len(nlocs), axis=0).reshape(-1, )
                nalocs = nlocs_x - alocs_x
                naIdx = np.where(nalocs > 0)[0].min()
                alocs_norm = alocs_norm[naIdx:]
                # 最近符头索引
                aloc_min = np.argmin(alocs_norm) + naIdx
                # 小节索引，小节内索引
                bar_idx = notes_row[aloc_min][-1]
                bar_item_idx = aloc_min - np.sum(notes_row[:, -1] < bar_idx)
                # 找到所在位置
                pitches_barline = pitches_current[int(bar_idx)]
                pitch = pitches_barline[bar_item_idx]
                # 临时音高处理
                pitches_barline_len = len(pitches_barline)
                for j in range(bar_item_idx, pitches_barline_len):
                    temp_pitch = pitches_barline[j]
                    if temp_pitch == pitch:
                        # 根据类型修改音高
                        temp_pitch = changePitchLocal(temp_pitch, acc_type)
                        pitches_barline[j] = temp_pitch
                pitches_current[int(bar_idx)] = pitches_barline

        pitches.append(pitches_current)

    return pitches


def changePitchLocal(pitch: str, type: int):
    if type == 0:
        # 临时升
        return pitch[0] + '#' + pitch[-1]
    elif type == 1:
        # 临时降
        return pitch[0] + 'b' + pitch[-1]
    else:
        # 还原
        return pitch[0] + pitch[-1]


def changePitchGlobal(pitch: str, keys: list, type: int):
    letter, num = pitch[0], pitch[1]
    pitch_series = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    if letter in keys:
        if type == 0:
            return letter + '#' + num
        else:
            return letter + 'b' + num
    else:
        return pitch


def getKeyGlobal(key: int, num: int):
    if key == 0:
        letters = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
    else:
        letters = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
    return letters[:num]


def cal_pitch_nonation(clef, pitch):
    # 将相对音高转换为科学音调记号
    pitch_series = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    q, r, num = 0, 0, 0
    if clef == 1:
        # 低音谱号，pitch=-1表示C2
        q = int(pitch + 1) // 7
        r = int(pitch + 1) % 7
        num = 2 + q
    else:
        # 高音谱号，pitch=1表示C4
        q = int(pitch - 1) // 7
        r = int(pitch - 1) % 7
        num = 4 + q
    letter = pitch_series[r]
    # 从1开始
    return letter + str(num + 1)


def get_others_symbol(clefs, keys, accs, others):
    if len(keys) == 0:
        if len(accs) == 0:
            if len(others) == 0:
                others_symbol = clefs[:, 0:4]
            else:
                others_symbol = np.concatenate((clefs[:, 0:4], others[:, 0:4]))
        else:
            if len(others) == 0:
                others_symbol = np.concatenate((clefs[:, 0:4], accs[:, 0:4]))
            else:
                others_symbol = np.concatenate(
                    (clefs[:, 0:4], accs[:, 0:4], others[:, 0:4]))
    else:
        if len(accs) == 0:
            if len(others) == 0:
                others_symbol = np.concatenate((clefs[:, 0:4], keys[:, 0:4]))
            else:
                others_symbol = np.concatenate(
                    (clefs[:, 0:4], keys[:, 0:4], others[:, 0:4]))
        else:
            if len(others) == 0:
                others_symbol = np.concatenate(
                    (clefs[:, 0:4], keys[:, 0:4], accs[:, 0:4]))
            else:
                others_symbol = np.concatenate(
                    (clefs[:, 0:4], keys[:, 0:4], accs[:, 0:4], others[:,
                                                                       0:4]))
    return others_symbol


def barlineDectect(img,
                   morphFilterVertLineLength,
                   morphFilterVertLineWidth,
                   maxBarlineWidth,
                   notes,
                   nhlen_est,
                   nhwidth_est,
                   others_symbol,
                   staveMidpts,
                   line_width,
                   scale=1):
    vlines = isolateBarlines(img, morphFilterVertLineLength,
                             morphFilterVertLineWidth, maxBarlineWidth)
    vlines_black = (1 - vlines) * 255
    vlines_removes = whiteAllSymbolsArea(vlines_black, notes, nhlen_est,
                                         nhwidth_est, others_symbol,
                                         staveMidpts, line_width)
    # cv2.imwrite('{}{}_vlines_removes.jpg'.format(resultFile, name), vlines_removes)
    barlines = findVlineRect(vlines_removes)
    # print('barlines len: ', len(barlines))
    if len(barlines) == 0:
        return [], []
    barlines_h = barlines[:, -1]
    barlines = barlines[np.where(
        barlines_h >= np.ceil(int(np.max(barlines_h) * 0.4)))]
    barlines_w = barlines[:, -2]
    # print('barlines_w: ', barlines_w)
    barlines = barlines[np.where(barlines_w >= morphFilterVertLineWidth * 0.6)]
    # print('barlines len: ', len(barlines))
    blocs = [(.5 * (bbox[1] + bbox[3]), .5 * (bbox[0] + bbox[2]))
             for bbox in barlines]
    return barlines, blocs


def whiteAllSymbolsArea(img,
                        notes,
                        notesH_est,
                        notesW_est,
                        symbols_without_notes,
                        staveMidpts,
                        line_width,
                        scale=1):
    ''''''
    H, W = img.shape
    tempH = int(8 * line_width + notesH_est / 4.0)
    _, img = cv2.threshold(img.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for i, note in enumerate(notes):
        # 切割note局部区域
        # 假设乐符在上加二线-下架二线间
        center_y = int((note[0] + note[2]) / 2.0 * scale)
        center_x = int((note[1] + note[3]) / 2.0 * scale)
        delta_y = tempH
        delta_x = int(notesW_est)

        tempMinY = max(0, center_y - delta_y)
        tempMinX = max(0, center_x - delta_x)
        tempMaxY = min(H, center_y + delta_y)
        tempMaxX = min(W, center_x + delta_x)

        # notesH_est_half = int(notesH_est / 2)
        # notesW_est_half = int(notesW_est / 2)
        img[tempMinY:tempMaxY, tempMinX:tempMaxX] = int(255)

    for symbol in symbols_without_notes:
        symbol_minX = int(symbol[1] * scale)
        symbol_minY = int(symbol[0] * scale)
        symbol_maxX = int(symbol[3] * scale)
        symbol_maxY = int(symbol[2] * scale)
        symbol_width = symbol_maxX - symbol_minX
        # symbol_height = symbol_maxY - symbol_minY

        tempMinX = max(0, int(symbol_minX - symbol_width / 2))
        tempMinY = max(0, int(symbol_minY - tempH / 2))
        tempMaxX = min(W, int(symbol_maxX + symbol_width / 2))
        tempMaxY = min(H, int(symbol_maxY + tempH / 2))
        # img[tempMinY:tempMaxY, tempMinX: tempMaxX] = np.ones((tempMaxY - tempMinY, tempMaxX - tempMinX), dtype=int) * 255
        img[tempMinY:tempMaxY, tempMinX:tempMaxX] = int(255)

    miny = max(0, int(staveMidpts.min() - tempH / 2))
    maxy = min(H, int(staveMidpts.max() + tempH / 2))
    img[0:miny, :] = int(255)
    img[maxy:H, :] = int(255)
    # clocs_y = np.array(clocs)[:,0]
    # clocs_y_min, clocs_y_max = max(0, int(clocs_y.min() - tempH / 2)), min(H, int(clocs_y.max() + tempH / 2))
    # img[clocs_y_min: clocs_y_max, :] = int(255)

    return img


def findVlineRect(img):

    H, W = img.shape
    # 扩宽图片，以便边缘框检测
    newImg = np.ones((H, W + 100), dtype=np.int8) * 255
    newImg[0:H, 0:W] = img

    _, binary = cv2.threshold(newImg.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 轮廓检测
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_NONE)

    # 验证上面的图片有多少个图像轮廓
    rects = []
    # 删除图片外边框
    for cnt in contours[1:]:
        x, y, w, h = cv2.boundingRect(cnt)
        x1, y1 = min(x + w, W), min(y + h, H)
        rects.append([int(x), int(y), int(x + w), int(y + h), int(w), int(h)])
    rects = np.array(rects)

    return rects


def convertPitchToNum(clef, pitch):
    '''
    音高转midi
    '''
    letter_dict = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}
    clefG_4_list = [60, 62, 64, 65, 67, 69, 71]
    clefF_3_list = [48, 50, 52, 53, 55, 57, 59]
    letter, num = pitch[0], int(pitch[-1])
    letter_idx = int(letter_dict[letter])
    midi = 0
    if clef == 0:
        # 高音谱号
        midi = clefG_4_list[letter_idx] + (num - 4) * 12
    else:
        # 低音谱号
        midi = clefF_3_list[letter_idx] + (num - 3) * 12
    # 临时升降号
    if len(pitch) > 2:
        tmp_symbol = pitch[1:-1]
        # 1个#/b，或2个b
        tmp_symbol_type = tmp_symbol[0]
        if tmp_symbol_type == '#':
            midi += len(tmp_symbol)
        else:
            midi -= len(tmp_symbol)
    return midi


def generatePitchJson(clefs_group, notes_barline_group, pitches, scale,
                      originW, originH, linesep):
    pitchJsonDict = {}
    noteInfo = []
    pictureInfo = {}
    clefY_list = []
    rows = len(notes_barline_group)
    for i in range(rows):
        # 获取当前行谱号，默认只有一个
        clef_row = clefs_group[i][0]
        # 谱号坐标
        cy0, cx0, cy1, cx1 = clef_row[:4]
        # 谱号中心坐标
        cy, cx = int((cy0 + cy1) / 2 * scale), int((cx0 + cx1) / 2 * scale)
        clefY_list.append(cy)
        # 0高音，1低音
        clef_current = clef_row[-2]
        pitchJsonDict_row = {}
        measureCount = len(pitches[i])
        pitchJsonDict_row['measureCount'] = measureCount
        midi_list = []
        note_list = []
        for j in range(measureCount):
            pitches_measure = pitches[i][j]
            notes_barline_group_measure = notes_barline_group[i][j]
            measure = j + 1
            for k in range(len(pitches_measure)):
                note_dict = {}
                pitch = pitches_measure[k]
                midi = convertPitchToNum(clef_current, pitch) - 20
                y0, x0, y1, x1 = notes_barline_group_measure[k][:4]
                x, y = int((x0 + x1) / 2 * scale), int((y0 + y1) / 2 * scale)
                w, h = int((x1 - x0) * scale), int((y1 - y0) * scale)
                note_dict['pitch'] = int(midi)
                note_dict['measure'] = int(measure)
                note_dict['x'] = x
                note_dict['y'] = y
                note_dict['w'] = w
                note_dict['h'] = h
                note_list.append(note_dict)
                midi_list.append(midi)
        pitchJsonDict_row['lastNotePitch'] = int(midi_list[-1])
        pitchJsonDict_row['maxPitch'] = int(max(midi_list))
        pitchJsonDict_row['minPitch'] = int(min(midi_list))
        pitchJsonDict_row['startY'] = cy
        pitchJsonDict_row['startX'] = cx
        pitchJsonDict_row['noteList'] = note_list
        noteInfo.append(pitchJsonDict_row)
    # 音符信息
    pitchJsonDict['noteInfo'] = noteInfo
    # 分割位置
    segmentation_list = []
    if len(clefY_list) <= 1:
        segmentation_list = [int(originH)]
    else:
        clefY_list = np.array(clefY_list)
        clefY_list = (clefY_list[:-1] + clefY_list[1:]) / 2
        segmentation_list = list(clefY_list)
    # 图片信息
    pictureInfo['pictureWidth'] = int(originW)
    pictureInfo['pictureHeight'] = int(originH)
    pictureInfo['size'] = int(linesep)
    pictureInfo['segmentation'] = segmentation_list
    pitchJsonDict['pictureInfo'] = pictureInfo
    pitchJson = json.dumps(pitchJsonDict)
    return pitchJson


#### main ####
def processQuery(imagefile, jsonData, name=None, resultFile=None):

    if len(jsonData) == 0:
        print('符号检测结果为空')
        return []

    # pre-processing
    pim1 = Image.open(imagefile).convert(
        'L')  # pim indicates PIL image object, im indicates raw pixel values
    originW, originH = pim1.width, pim1.height
    pim2 = removeBkgdLighting(pim1, opt.thumbnailFilterSize, opt.thumbnailW,
                              opt.thumbnailH)

    linesep, scores = estimateLineSep(pim2, opt.estLineSep_NumCols,
                                      opt.estLineSep_LowerRange,
                                      opt.estLineSep_UpperRange,
                                      opt.estLineSep_Delta)
    # print('linesep: ', linesep)

    targetH, targetW, scale_factor = calcResizedDimensions(
        pim2, linesep, opt.targetLineSep)
    # 调整图片尺寸大小,使得五线谱线间距为10
    pim2 = pim2.resize((targetW, targetH))

    # staff line features
    X2 = getNormImage(pim2)

    # morphFilterHorizLineSize=41 notebarFiltLen=3 notebarRemoval=0.9
    hlines, notebarsOnly = isolateStaffLines(X2, opt.morphFilterHorizLineSize,
                                             opt.notebarFiltLen,
                                             opt.notebarRemoval)

    # cv2.imwrite(resultFile + name + '_hlines.jpg', (1 - hlines) * 255)
    # showGrayscaleImage(notebarsOnly, name='notebarsOnly', isSaved = True, resultFile=resultFile)

    # 计算五线谱特征图
    # 卷积操作后, 得到K * (H - maxFiltSize + 1) * C的特征张量，其中K是梳状滤波器的数量，H是图像的高度（以像素为单位），C是列数
    # 10， 8.5，11.75，0.25
    featmap, stavelens, columnWidth = computeStaveFeatureMap(
        hlines, opt.calcStaveFeatureMap_NumCols,
        opt.calcStaveFeatureMap_LowerRange, opt.calcStaveFeatureMap_UpperRange,
        opt.calcStaveFeatureMap_Delta)

    # 各个符号位置解析
    notes, clefs, keys, accs, braces, others = paraseJson(
        jsonData, scale_factor)
    # 过滤位置不合理符号
    # vpadding = int(linesep * 2)
    notes, clefs, keys, accs, braces, others = filterAllSymbols(
        notes, clefs, keys, accs, braces, others, opt.targetLineSep)
    # 符头的位置（y,x）、平均高度、平均宽度
    nhlocs, nhlen_est, nhwidth_est = getNoteheadInfo(notes)
    # 其他符号中心坐标（y,x）
    clocs, klocs, alocs, olocs = getOtherSymbolsCenter(clefs, keys, accs,
                                                       braces, others)
    # 获取花括号的平均长度
    # brlen_est = getBracesLength(braces)

    # infer note values
    '''
    可能的问题：
        1、对于钢琴谱，当符头处于第二行时，当前符头的中心距离可能离第一行更近，从而造成误判
    '''
    # 估计每一个符头所在的局部区域五线谱的位置与谱间距
    estStaffLineLocs, sfiltlen = getEstStaffLineLocs(
        featmap, nhlocs, stavelens, columnWidth, opt.maxDeltaRowInitial,
        int(-2 * opt.targetLineSep))

    # visualizeEstStaffLines(estStaffLineLocs, hlines, name='start', resultFile=resultFile, isSaved=True)

    # minNumStaves = 2, maxNumStaves= 12, minStaveSeparation = 60.0
    # 聚类中心坐标数组, 数量为五线谱行数
    staveMidpts = estimateStaffMidpoints(estStaffLineLocs, opt.minNumStaves,
                                         opt.maxNumStaves,
                                         opt.minStaveSeparation)
    # 将符头分配给最近的全局五线谱系统，并使用以该系统位置为中心的更小的上下文区域
    staveIdxs, nhRowOffsets = assignNoteheadsToStaves(nhlocs, staveMidpts)

    # # 根据全局五线谱位置的估计，重新估计每一个符头所在的局部区域五线谱的位置与谱间距
    estStaffLineLocs, _ = getEstStaffLineLocs(
        featmap, nhlocs, stavelens, columnWidth, opt.maxDeltaRowRefined,
        (nhRowOffsets - 2 * opt.targetLineSep).astype(int))
    # visualizeEstStaffLines(estStaffLineLocs, hlines, name='end', resultFile=resultFile, isSaved=True)
    # embed()
    # 使用预测的每个符头临近的五线谱位置和间距，估计符头在五线谱的垂直位置（通过线性插值完成）
    nhvals = estimateNoteLabels(estStaffLineLocs, notes)
    # 删除符头位置超出上架2线的值
    # nhvals, notes, staveIdxs = filterNoteBeyondValue(nhvals, notes, staveIdxs, value=8)

    others_symbol = get_others_symbol(clefs, keys, accs, others)
    # 小节线检测
    # 35，10，40
    barlines, blocs = barlineDectect(X2, opt.morphFilterVertLineLength,
                                     opt.morphFilterVertLineWidth,
                                     opt.maxBarlineWidth, notes, nhlen_est,
                                     nhwidth_est, others_symbol, staveMidpts,
                                     opt.targetLineSep)
    if len(barlines) == 0:
        print('当前图片小节线无法检测')
        return []
    # 符号分组
    # 将其余符号分配给最近的全局五线谱系统，以定位其所在行数
    cIdxs, kIdxs, aIdxs, oIdxs, bIdxs = assignOtherSymbolsToStaves(
        staveMidpts, clocs, klocs, alocs, olocs, blocs)

    nhvals = np.array(nhvals).reshape(-1, 1)
    notes = np.hstack((notes, nhvals))
    notes_group = get_symbols_group(notes, staveIdxs)
    clefs_group = get_symbols_group(clefs, cIdxs)
    keys_group = get_symbols_group(keys, kIdxs)
    accs_group = get_symbols_group(accs, aIdxs)
    # others_group = get_symbols_group(others, oIdxs)
    barlines_group = get_symbols_group(barlines, bIdxs, isYFirst=False)

    if list(range(len(barlines_group))) != list(barlines_group.keys()):
        print('当前图片小节线漏检过多')
        return []
    # 符头按小节分组
    notes_barline_group, notes_group = separatedNotesGroup(
        notes_group, barlines_group)

    # 音高计算
    pitches = get_Scientific_pitch_notation(notes_barline_group, notes_group,
                                            clefs_group, keys_group,
                                            accs_group)

    scale = linesep / opt.targetLineSep * 1.0

    # 生成结果json
    pitchJson = generatePitchJson(clefs_group, notes_barline_group, pitches,
                                  scale, originW, originH, linesep)

    if resultFile != None:
        # 绘制结果
        drawDetectResult(notes_group, pitches, others_symbol, barlines,
                         imagefile, resultFile, name, scale)
    #     # 音高保存
    #     with open(resultFile + 'pitches.txt', 'w') as fh:
    #         fh.write(str(pitches))

    # oriimage = cv2.imread(imagefile, 0)
    # drawVlineArea(oriimage, barlines, resultFile, name, scale)
    return pitchJson


def test():
    # /home/cy/llj/SheetMidiRetrieval/config/dsv2_fcos_hrnet_w18_tiny.py
    # /home/cy/llj/SheetMidiRetrieval/config/dsv2_fcos_hrnet_w18.py
    # config_file = '/home/cy/llj/SheetMidiRetrieval/config/dsv2_fcos_hrnet_w18_tiny.py'
    config_file = './config/dsv2_fcos_hrnet_w18_tiny.py'
    # /home/data/cy/llj/mmDetection/class34_tiny/latest.pth
    # /home/data/cy/llj/mmDetection/focs_w18_mmdet-2.19/epoch_400.pth
    # checkpoint_file = '/home/data/cy/llj/mmDetection/class34_tiny/latest.pth'
    checkpoint_file = './config/epoch_240.pth'
    # /home/cy/llj/mmdetection-2.17/mmdetection/OMR
    # /home/data/cy/llj/dataset/mmdetection-test/TestData/ErrorImages
    imagePath = 'data/queries'
    save_path = 'result'
    name = 'p1_q1'
    imgType = 'jpg'
    imageFile = imagePath + '/{}.'.format(name) + imgType
    resultFile = save_path + '/{}/'.format(name)
    if not os.path.exists(resultFile):
        os.makedirs(resultFile)

    start_time = time.perf_counter()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # raise ValueError(type(model))
    next_time = time.perf_counter()
    print('init_detector use time {} s'.format(next_time - start_time))
    # 目标检测
    jsonData = detect(model, imageFile)
    detect_time = time.perf_counter()
    print('detect use time {} s'.format(detect_time - next_time))

    # with open(resultFile + name + '.json', 'w') as f:
    #     jsData = json.dumps(jsonData)
    #     f.write(jsData)

    # 音高定位
    pitchJson = processQuery(imageFile, jsonData, name, resultFile)
    end_time = time.perf_counter()
    print('processQuery use time {} s'.format(end_time - detect_time))
    # 保存到本地
    # if pitchJson != []:
    #     with open(resultFile + name + '_result.json', 'w') as f:
    #         f.write(pitchJson)
    with open(os.path.join(resultFile,name+"_result.json"),'w')as f:
        json.dump(jsonData,f)


def run_batch():
    start_time = time.perf_counter()

    file_path = '/home/data/cy/llj/dataset/性能测试'
    save_path = '/home/data/cy/llj/dataset/camera-DataBase/result/test'
    config_file = '/home/cy/llj/SheetMidiRetrieval/config/dsv2_fcos_hrnet_w18_tiny.py'
    checkpoint_file = '/home/data/cy/llj/mmDetection/class34_tiny/latest.pth'
    imgType = 'png'

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    next_time = time.perf_counter()
    print('preDetect use time {} s'.format(next_time - start_time))

    pathList = os.listdir(file_path)
    for path in pathList:
        start_time = time.perf_counter()

        imgType_len = len(imgType)
        if path[-imgType_len: ] != imgType: 
            continue
        name = path[:-(imgType_len + 1)]
        print('name: ', name)
        # 'IMG_1227'
        # jsonFile = save_path + '/{}/{}.json'.format(name, name)
        imagefile = file_path + '/{}.'.format(name) + imgType
        resultFile = save_path + '/{}/'.format(name)
        if not os.path.exists(resultFile):
            os.makedirs(resultFile)

        # 目标检测
        jsonData = detect(model, imagefile)

        # with open(resultFile + name + '.json', 'w') as f:
        #     jsData = json.dumps(jsonData)
        #     f.write(jsData)

        # 音高定位
        pitchJson = processQuery(imagefile, jsonData, name, resultFile)
        end_time = time.perf_counter()
        print('one OMR use time {} s'.format(end_time - start_time))

        # # 保存到本地
        # if pitchJson != []:
        #     with open(resultFile + name + '_result.json', 'w') as f:
        #         f.write(pitchJson)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # run_batch()
    test()
    