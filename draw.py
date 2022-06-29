from IPython.terminal.embed import embed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.cluster import KMeans
import cv2
from scipy.signal import convolve2d
from skimage.filters import threshold_sauvola

def sauvola(img, windowsize):
    sauvolaImage = threshold_sauvola(img, window_size=windowsize, k=0.25)
    binary_sauvola = (img > sauvolaImage) * 255
    binary_sauvola = binary_sauvola.astype(img.dtype)
    return binary_sauvola

def showGrayscaleImage(X, sz = (10,10), maxval = 1, inverted = True, name = None, isSaved = False):
    # by default assumes X is a normalized image between 0 (white) and 1 (black)
    plt.figure(figsize = sz)
    if inverted:
        plt.imshow(maxval-X, cmap='gray')
    else:
        plt.imshow(X, cmap='gray')
    if isSaved:
        plt.savefig('{0}{1}.jpg'.format(resultFile, name))

def visualizeLabels(img, bboxes):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(1 - img, cmap='gray')
    
    for (minr, minc, maxr, maxc) in bboxes:
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

def visualizeEstStaffLines(preds, arr, name='', resultFile='', isSaved=False):
    showGrayscaleImage(arr, (15,15))
    rows1 = np.array([int(pred[0]) for pred in preds]) # top staff line
    rows2 = np.array([int(pred[1]) for pred in preds]) # bottom staff line
    cols = np.array([int(pred[2]) for pred in preds]) # nh col
    rows3 = np.array([int(pred[3]) for pred in preds]) # nh row
    plt.scatter(cols, rows1, c = 'r', s = 12)
    plt.scatter(cols, rows2, c = 'b', s = 12)
    plt.scatter(cols, rows3, c = 'y', s = 12)
    if isSaved:
        plt.savefig('{0}EstStaffLines-{1}.jpg'.format(resultFile, name))

def visualizeClusterEst(preds, arr, staffMidpts, staveIdxs, name='', isSaved=False):
    showGrayscaleImage(arr, (15,15))
    cols = np.array([pred[2] for pred in preds]) # nh col
    for i in range(len(cols)):
        plt.scatter(cols[i], staffMidpts[staveIdxs[i]], c = 'b', s=3)
    if isSaved:
        plt.savefig('{0}ClusterEst-{1}.jpg'.format(resultFile, name))

def debugStaffMidpointClustering(preds):
    r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
    inertias = []
    mindiffs = []
    clusterRange = np.arange(2,12)
    for numClusters in clusterRange:
        kmeans = KMeans(n_clusters=numClusters, n_init=1, random_state = 0).fit(r.reshape(-1,1))
        inertias.append(kmeans.inertia_)
        sorted_list = np.array(sorted(np.squeeze(kmeans.cluster_centers_)))
        diffs = sorted_list[1:] - sorted_list[0:-1]
        mindiffs.append(np.min(diffs))
    plt.subplot(211)
    plt.plot(clusterRange, np.log(inertias))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.subplot(212)
    plt.plot(clusterRange, mindiffs)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Min Centroid Separation')
    plt.axhline(60, color='r')

def visualizeStaffMidpointClustering(preds, centers):
    r = np.array([.5*(tup[0] + tup[1]) for tup in preds]) # midpts of estimated stave locations
    plt.plot(r, np.random.uniform(size = len(r)), '.')
    for center in centers:
        plt.axvline(x=center, color='r')

# X2, nhlocs, otherlocs, staveIdxs, otherIdxs, 'ClustersAll', True
def visualizeClustersAll(arr, nhlocs, otherests, staveIdxs, otherIdxs, name = None, isSaved = False):
    '''
    可视化聚类分类结果（全部符号）
    '''
    showGrayscaleImage(arr)
    # 符头
    rows = np.array([tup[0] for tup in nhlocs])
    cols = np.array([tup[1] for tup in nhlocs])
    plt.scatter(cols, rows, s=10, c=staveIdxs)
    for i in range(len(staveIdxs)):
        plt.text(cols[i], rows[i] - 15, str(staveIdxs[i]), fontsize = 8, color='red')
    # 其他符号
    ax = plt.gca()
    for i in range(len(otherIdxs)):
        est = otherests[i]
        rect = mpatches.Rectangle((est[1], est[0]), est[3] - est[1], est[2] - est[0], edgecolor = 'r', facecolor='none')
        ax.add_patch(rect)
        plt.text(.5*(est[1]+est[3]), est[0] - 12, str(otherIdxs[i]), fontsize= 8, color = 'blue')
    if isSaved:
        plt.savefig('{0}{1}.jpg'.format(resultFile, name))

def visualizeClusters(arr, nhlocs, clusters, name = None, isSaved = False):
    '''
    可视化聚类分类结果
    '''
    showGrayscaleImage(arr)
    rows = np.array([tup[0] for tup in nhlocs])
    cols = np.array([tup[1] for tup in nhlocs])
    plt.scatter(cols, rows, c=clusters)
    for i in range(len(clusters)):
        plt.text(cols[i], rows[i] - 15, str(clusters[i]), fontsize = 12, color='red')
    if isSaved:
        plt.savefig('{0}{1}.jpg'.format(resultFile, name))

def visualizeNoteLabels(arr, vals, locs, name = None, isSaved = False):
    showGrayscaleImage(arr)
    rows = np.array([loc[0] for loc in locs])
    cols = np.array([loc[1] for loc in locs])
#     plt.scatter(cols, rows, color='blue')
    for i in range(len(rows)):
        plt.text(cols[i], rows[i] - 20, str(vals[i]), fontsize = 12, color='red')
    if isSaved:
        plt.savefig('{0}{1}.jpg'.format(resultFile, name))
        # cv2.imwrite('{0}{1}.jpg'.format(resultFile, name))
        np.savetxt('{}{}.txt'.format(resultFile, name), vals, fmt='%i')

def determineStaveGrouping(staveMidpts, vlines):
    '''
    Args:
        staveMidpts(list): 聚类中心y坐标数组（一维），升序排列
        vlines(numpy): 小节线特征图
    '''
    N = len(staveMidpts)
    # 各行相加
    rowSums = np.sum(vlines, axis=1)
    
    # grouping A: 0-1, 2-3, 4-5, ...
    elems_A = []
    map_A = {}
    for i, staveIdx in enumerate(np.arange(0, N, 2)):
        if staveIdx+1 < N:
            startRow = int(staveMidpts[staveIdx])
            endRow = int(staveMidpts[staveIdx+1]) + 1
            elems_A.extend(rowSums[startRow:endRow])
            map_A[staveIdx] = staveIdx
            map_A[staveIdx+1] = staveIdx + 1
        else:
            map_A[staveIdx] = -1 # unpaired stave
    
    # grouping B: 1-2, 3-4, 5-6, ...
    elems_B = []
    map_B = {}
    map_B[0] = -1 
    for i, staveIdx in enumerate(np.arange(1, N, 2)):
        if staveIdx+1 < N:
            startRow = int(staveMidpts[staveIdx])
            endRow = int(staveMidpts[staveIdx+1]) + 1
            elems_B.extend(rowSums[startRow:endRow])
            map_B[staveIdx] = staveIdx - 1
            map_B[staveIdx + 1] = staveIdx
        else:
            map_B[staveIdx] = -1
    
    if N > 2:
        evidence_A = np.median(elems_A)
        evidence_B = np.median(elems_B)
        if evidence_A > evidence_B:
            mapping = map_A
        else:
            mapping = map_B
    else:
        evidence_A = np.median(elems_A)
        evidence_B = 0
        mapping = map_A
    
    return mapping, (evidence_A, evidence_B, elems_A, elems_B)

def debugStaveGrouping(vlines, staveCenters):
    plt.plot(np.sum(vlines, axis=1))
    for m in staveCenters:
        plt.axvline(m, color = 'r')

def clusterNoteheads(staveIdxs, mapping):
    clusterIdxs = [mapping[staveIdx] for staveIdx in staveIdxs]
    maxClusterIdx = np.max(np.array(clusterIdxs))
    clusterPairs = []
    for i in range(0, maxClusterIdx, 2):
        clusterPairs.append((i,i+1))
    return clusterIdxs, clusterPairs

def barlineFeatureMap(img, brlen_est):
    img = cv2.imread('result/IMG_1003/Barlines.jpg', 0)
    img = 255 - img
    imgHeight, imgWidth = img.shape
    barlineFilter = np.ones((brlen_est, 1), dtype=int)
    print('brlen_est: ', brlen_est)
    featureMap = np.zeros((imgHeight - brlen_est + 1, imgWidth))
    featureMap = convolve2d(img, barlineFilter, mode='valid')
    mapidx = np.argmax(featureMap, axis=1)   
    featureMap_flat = featureMap.flatten()
    plt.plot(featureMap_flat)
    plt.savefig('barlineFeatureMap.jpg')

    return featureMap, mapidx

def flatten(a):
    for each in a:
        if not isinstance(each, list):
            yield each
        else:
            yield from flatten(each)

def drawVlineArea(img, rects, resultFile, name, scale=1):
    img = img.copy()
    newImg = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_GRAY2BGR)
    # 绘制mote
    for rect in rects:
        left_top = (int(rect[0] * scale), int(rect[1] * scale))
        right_bottom = (int(rect[2] * scale), int(rect[3] * scale))
        width, height = int(rect[-2] * scale), int(rect[-1] * scale)
        cv2.rectangle(newImg, left_top, right_bottom, (220,123,12), thickness=3)
        cv2.putText(newImg, '{},{}'.format(height, width), (left_top[0], left_top[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
    cv2.imwrite('{}{}_vlines.jpg'.format(resultFile, name), newImg) 

def drawDetectResult(notes_group, pitches, others_symbol, barlines, imagefile, resultFile, name, scale):
    oriimage = cv2.imread(imagefile)
    # origin_notes = []
    rows = len(notes_group)

    for symbol in others_symbol:
        left_top = (int(symbol[1] * scale), int(symbol[0] * scale))
        right_bottom = (int(symbol[3] * scale), int(symbol[2] * scale))
        cv2.rectangle(oriimage, left_top, right_bottom, (30,105,210), thickness=2)

    for barline in barlines:
        left_top = (int(barline[0] * scale), int(barline[1] * scale))
        right_bottom = (int(barline[2] * scale), int(barline[3] * scale))
        cv2.rectangle(oriimage, left_top, right_bottom, (220,123,12), thickness=2)
    for i in range(rows):
        notes = notes_group[i]
        current_row_pitches = list(flatten(pitches[i]))
        for i in range(len(current_row_pitches)):
            note = notes[i]
            y0, x0, y1, x1 = note[:4]
            y0, y1 = int(y0 * scale), int(y1 * scale)
            x0, x1 = int(x0 * scale), int(x1 * scale)
            center_x = int((x0 + x1) * 0.5)
            center_y = int((y0 + y1) * 0.5)
            current_row_pitch = current_row_pitches[i]
            cv2.rectangle(oriimage, (x0, y0), (x1, y1), (0,90,139), thickness=2)
            cv2.putText(oriimage, '{}'.format(current_row_pitch), (center_x - 10, center_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite('{}{}_result.jpg'.format(resultFile, name), oriimage)

def drawNoteNhvals(notes, imagefile, resultFile, name, scale, title="nhvals"):
    oriimage = cv2.imread(imagefile)
    for note in notes:
        y0, x0, y1, x1 = note[:4]
        nhval = note[-1]
        y0, y1 = int(y0 * scale), int(y1 * scale)
        x0, x1 = int(x0 * scale), int(x1 * scale)
        center_x = int((x0 + x1) * 0.5)
        center_y = int((y0 + y1) * 0.5)
        cv2.rectangle(oriimage, (x0, y0), (x1, y1), (0,90,139), thickness=2)
        cv2.putText(oriimage, '{}'.format(int(nhval)), (center_x - 10, center_y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite('{}{}_{}.jpg'.format(resultFile, name, title), oriimage)


def drawRectangle(oriimage, rects, savePath):
    # oriimage = cv2.imread(img)
    for rect in rects:
        left_top = (int(rect[0]), int(rect[1]))
        right_bottom = (int(rect[2]), int(rect[3]))
        cv2.rectangle(oriimage, left_top, right_bottom, (30,105,210), thickness=2)
    cv2.imwrite(savePath, oriimage)

def drawNoteAndStaffRect(oriimage, rects, points, savePath):
    for rect in rects:
        left_top = (int(rect[0]), int(rect[1]))
        right_bottom = (int(rect[2]), int(rect[3]))
        cv2.rectangle(oriimage, left_top, right_bottom, (30,105,210), thickness=2)
    for point in points:
        cv2.circle(oriimage, (int(point[1]), int(point[0])), 1, (0, 0, 255), 2)
    cv2.imwrite(savePath, oriimage)

def drawStaveMidptsAndStaffRect(oriimage, rects, staveMidpts, savePath):
    H, W = oriimage.shape[:2]
    for rect in rects:
        left_top = (int(rect[0]), int(rect[1]))
        right_bottom = (int(rect[2]), int(rect[3]))
        cv2.rectangle(oriimage, left_top, right_bottom, (30,105,210), thickness=2)
    for y in staveMidpts:
        pstart, pend, pcolor = (0, int(y)), (W, int(y)), (0, 0, 255)
        cv2.line(oriimage, pstart, pend, pcolor, 2, 8)
    cv2.imwrite(savePath, oriimage)


if __name__ == '__main__':
    name = 'IMG_1219'
    imagefile = '/home/data/cy/llj/dataset/mmdetection-test/TestData/compress/{}.jpeg'.format(name)
    jsonFile = '/home/cy/llj/SheetMidiRetrieval/result/{}/{}.json'.format(name, name)
    resultFile = '/home/cy/llj/SheetMidiRetrieval/result/{}/'.format(name)

    oriimage = cv2.imread(imagefile, 0)
    img_ = sauvola(oriimage, 25)
    cv2.imwrite('{}sauvola_{}.jpg'.format(resultFile, name), img_)