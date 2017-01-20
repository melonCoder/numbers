from numpy import *
from utils import *
from cluster import *

def TestKMeans(measDist = 0):
    trainSet = loadImages('train/')
    # testSet = loadImages('test/')
    # valSet = loadImages('val/')

    if measDist == 0:
        meas = projectionDist
    else:
        meas = ecludDist
    centList, clusterAssment = biKmeans(trainSet, 10, meas)
    print(shape(centList))
    f1 = open('cluster' +str(measDist) + '.txt', 'w')
    f2 = open('centList' + str(measDist) + '.txt', 'w')
    imageLen = shape(centList)[1]
    recordCnt = shape(clusterAssment)[0]

    for i in range(10):
        for j in range(imageLen):
            print(centList[i, j], file = f2)
        print('\n', file = f2)

    for i in range(recordCnt):
        print('%d %f' % (clusterAssment[i, 0], clusterAssment[i, 1]), file = f1)
    return centList, clusterAssment
