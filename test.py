from numpy import *
from loader import *
from cluster import *

def TestKMeans():
    trainSet = loadImages('train/')
    # testSet = loadImages('test/')
    # valSet = loadImages('val/')

    centList, clusterAssment = biKmeans(trainSet, 10)
