from numpy import *
import re

def loadSingleImage(f):
    import scipy.misc
    image = mat(scipy.misc.imread(f))
    m, n = shape(image)
    ret = mat(zeros((m, n)))
    # tried to do treate the image as binary, too much noise
    for i in range(m):
        for j in range(n):
            #ret[i, j] = image[i, j] / 255.0
            if image[i, j] >= 128:
                ret[i, j] = 1.0
            else:
                ret[i, j] = 0.0
    return reshape(ret, m*n), m, n

def loadImages(targetDir = 'train/', DEBUG = False):
    import os
    if DEBUG:
        fileNumber = 100
    else:
        fileNumber = sum([len(files) \
            for root,targetDirs,files in os.walk(targetDir)])

    retTmp = []
    for i in range(fileNumber):
        image, m, n = loadSingleImage(targetDir + str(i) + '.jpg')
        retTmp.append(image)
    ret = mat(zeros((fileNumber, m*n)))
    for i in range(fileNumber):
        ret[i, :] = retTmp[i]
    return ret

def plotImage(imageMat, f, m, n):
    import scipy.misc
    from math import log
    #cm = log(max(imageMat.tolist()[0]))
    image = reshape(imageMat, (m, n))
    #for i in range(m):
    #    for j in range(n):
    #        if image[i, j] != 0.0:
    #            image[i, j] = log(image[i , j]) / cm

    scipy.misc.toimage(image, cmin = 0.0, cmax = 1.0).save(f)

def plotClusterImages(centList, m = 28, n = 28, numClusters = 10):
    import matplotlib.pyplot as plt

    fCL = open(centList)
    cnt = 0
    cenlists = mat(zeros((numClusters, m * n)))
    pre = False
    i = 0
    for line in fCL.readlines():
        if len(line.strip()) == 0:
            i = 0
            if pre == False:
                pre = True
            else:
                pre = False
                cnt += 1
        else:
            cenlists[cnt, i] = float(line.strip())
            i += 1

    for i in range(numClusters):
        plotImage(cenlists[i, :], str(i) + '.jpg', m, n)

def getCheetSheetFromFile(f):
    # lenCS = len(open(f).readlines())
    cheetsheet = mat(zeros((1, 42000)))
    fCS = open(f)
    for line in fCS.readlines():
        pattern = re.compile(r'(\d+)\.jpg (\d+)')
        match = pattern.match(line)
        cheetsheet[0, int(match.group(1))] = int(match.group(2))
    return cheetsheet

def getClassifyFromFile(f):
    lenC = len(open(f).readlines())
    classify = mat(zeros((1, lenC)))
    cnt = 0
    fCluster = open(f)
    for line in fCluster.readlines():
        classify[0, cnt] = int(line.strip().split(' ')[0])
        cnt += 1
    return classify

def getCentListFromFile(f, num = 10, length = 28*28):
    fCL = open(f)
    cnt = 0
    cenlist = mat(zeros((num, length)))
    pre = False
    i = 0
    for line in fCL.readlines():
        if len(line.strip()) == 0:
            i = 0
            if pre == False:
                pre = True
            else:
                pre = False
                cnt += 1
        else:
            cenlist[cnt, i] = float(line.strip())
            i += 1
    return cenlist


def plotDistribution(cluster, cheetsheet, numClusters = 10):
    import matplotlib.pyplot as plt

    numInCluster = len(open(cluster).readlines())
    numInCS = len(open(cheetsheet).readlines())
    if numInCS < numInCluster:
        print('number of images not match in cluster and cheetsheet')
        return -1

    cs = mat(zeros((1, numInCS)))
    fCS = open(cheetsheet)
    for line in fCS.readlines():
        pattern = re.compile(r'(\d+)\.jpg (\d+)')
        match = pattern.match(line)
        cs[0, int(match.group(1))] = int(match.group(2))

    cnt = 0
    fCluster = open(cluster)
    ret = mat(zeros((numClusters, 10)))
    for line in fCluster.readlines():
        i = int(line.strip().split(' ')[0])
        ret[i, int(cs[0, cnt])] += 1
        cnt += 1

    # print matrix
    for i in range(numClusters):
        print('\t'.join([str(x) for x in ret[i, :].tolist()[0]]))
    for i in range(numClusters):
        tmp = ret[i, :].tolist()[0]
        if float(max(tmp)) / sum(tmp) < 0.9:
            continue
        print('cluster ' + str(i) + ' indicate ' + str(tmp.index(max(tmp)))\
                + ' with probability ' + str(float(max(tmp) / sum(tmp))))

    # plot bar
    # for idx  in range(numClusters):
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #     plt.bar([i for i in range(10)], [ret[idx, j] for j in range(10)])
    #     plt.savefig('bar_' + str(idx) + '.jpg')

if __name__ == '__main__':
    #plotDistribution('eclude.txt', 'train.txt')
    # plotDistribution('./255_normalize/cluster1.txt', 'train.txt')
    # plotClusterImages('./20_cluster/centList1.txt', numClusters = 20)
    plotDistribution('./cluster1.txt', 'train.txt', numClusters = 100)
    #plotDistribution('./cluster1.txt', 'train.txt')
    #plotClusterImages('./centList1.txt')
