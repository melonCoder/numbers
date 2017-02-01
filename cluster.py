from numpy import *

def projectionDist(k, j):
    print((k * j.T)[0, 0])
    return abs((k * j.T)[0, 0])

def ecludDist(k, j):
    return sqrt(sum(power(k - j, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

def kMeans(dataSet, k, distMeas = projectionDist, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis = 0)

    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas = projectionDist):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]

    for j in range(m):
        clusterAssment[j, 1] = distMeas(mat(centroid0), dataSet[j, :])**2

    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = \
                    dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            # if len(ptsInCurrCluster) == 0:
            #     continue
            centroidMat, splitClustAss = \
                    kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = \
                    sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i), 1])
            print("sseSplit, and sseNotSplit", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseNotSplit + sseSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print('the bestCentToSplit is ',bestCentToSplit)
        print('the len of bestClustAss is ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:, 0].A == \
                bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment

def calcDistribution(cheetsheet, classification, m = 10, n = 10):
    distribution = mat(zeros((m, n)))
    for i in range(len(classification.T)):
        distribution[int(classification[0, i]), int(cheetsheet[0, i])] += 1
    return distribution

def calcObservedProbability(x, centList, distMeas = projectionDist):
    dist = []
    for i in range(len(centList)):
        dist.append(distMeas(x, centList[i, :]))
    m = min(dist)
    pmin = 1.0 / sum([m / d for d in dist])
    return [m / d * pmin for d in dist]

def calcProbability(distribution, x, centList, distMeas = projectionDist):
    PxiOnci = mat(zeros((10, 10)))
    PciOnxi = mat(zeros((10, 10)))
    for i in range(10):
        numInci = sum(distribution[i, :])
        for j in range(10):
            PxiOnci[i, j] = float(distribution[i, j]) / numInci

    for i in range(10):
        numInxi = sum(distribution[:, i])
        for j in range(10):
            PciOnxi[j, i] = float(distribution[j, i]) / numInxi

    Pci = calcObservedProbability(x, centList, distMeas)
    return Pci * PxiOnci
