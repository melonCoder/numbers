from numpy import *
from utils import *
from cluster import *

def TestKMeans(measDist = 0, numClusters = 10):
    trainSet = loadImages('train/')
    # testSet = loadImages('test/')
    # valSet = loadImages('val/')

    if measDist == 0:
        meas = projectionDist
    else:
        meas = ecludDist
    centList, clusterAssment = biKmeans(trainSet, numClusters, meas)
    print(shape(centList))
    f1 = open('cluster' +str(measDist) + '.txt', 'w')
    f2 = open('centList' + str(measDist) + '.txt', 'w')
    imageLen = shape(centList)[1]
    recordCnt = shape(clusterAssment)[0]

    for i in range(numClusters):
        for j in range(imageLen):
            print(centList[i, j], file = f2)
        print('\n', file = f2)

    for i in range(recordCnt):
        print('%d %f' % (clusterAssment[i, 0], clusterAssment[i, 1]), file = f1)
    return centList, clusterAssment

def testResult(targetDir = './', numClusters = 100, threshold = 0.9):
    import os
    import re

    classification = getClassifyFromFile('./cluster1.txt')
    centList = getCentListFromFile('./centList1.txt', numClusters)
    cheetsheet = getCheetSheetFromFile('./train.txt')
    distribution = calcDistribution(cheetsheet, classification, numClusters)
    validCents = {}
    for i in range(numClusters):
        tmp = distribution[i, :].tolist()[0]
        if float(max(tmp)) / sum(tmp) < threshold:
            continue
        validCents[i] = tmp.index(max(tmp))

    ret = 0
    cheetsheet = getCheetSheetFromFile(targetDir + '.txt')
    for root, dirs, files in os.walk(targetDir):
        fileCnt = len(files)
        for name in files:
            f = os.path.join(root, name)
            i = int(re.match(r'(\d+).jpg', name).group(1))
            if i < 40000 and i >= 30000:
                continue
            x, m, n = loadSingleImage(f)
            dist = inf
            for key in validCents.keys():
                if ecludDist(centList[key], x) < dist:
                    dist = ecludDist(centList[key], x)
                    curCent = key
            if validCents[curCent] == cheetsheet[0, i]:
                ret += 1
    print('corretness = ', float(ret)/ fileCnt)
    print('corrent number is ', ret)

def Test(f = 'test/0.jpg'):
    x, m, n = loadSingleImage(f)
    classification = getClassifyFromFile('./cluster1.txt')
    centList = getCentListFromFile('./centList1.txt')
    cheetsheet = getCheetSheetFromFile('./train.txt')
    distribution = calcDistribution(cheetsheet, classification)
    ret = calcProbability(distribution, x, centList, ecludDist).tolist()[0]
    print(ret)
    print(ret.index(max(ret)))
