from numpy import *

def loadSingleImage(f):
    import scipy.misc
    image = mat(scipy.misc.imread(f))
    m, n = shape(image)
    ret = mat(zeros((m, n)))
    # tried to do treate the image as binary, too much noise
    for i in range(m):
        for j in range(n):
            ret[i, j] = image[i, j] / 255.0
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
    image = reshape(imageMat, (m, n))
    scipy.misc.toimage(image, cmin = 0.0, cmax = 1.0).save(f)

def plotDistribution(cluster, cheetsheet):
    import re
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
    ret = mat(zeros((10, 10)))
    for line in fCluster.readlines():
        i = int(line.strip().split(',')[0])
        ret[i, int(cs[0, cnt])] += 1
        cnt += 1

    # print matrix
    for i in range(10):
        print('%4d\t%4d\t%4d\t%4d\t%4d\t%4d\t%4d\t%4d\t%4d\t%4d' %\
                (ret[i, 0], ret[i, 1], ret[i, 2], ret[i, 3], ret[i, 4], \
                ret[i, 5], ret[i, 6], ret[i, 7], ret[i, 8], ret[i, 9]))

    # plot bar
    fig = plt.figure()
    ax0 = fig.add_subplot(251)
    plt.bar([i for i in range(10)], [ret[0, j] for j in range(10)])
    ax1 = fig.add_subplot(252)
    plt.bar([i for i in range(10)], [ret[1, j] for j in range(10)])
    ax2 = fig.add_subplot(253)
    plt.bar([i for i in range(10)], [ret[2, j] for j in range(10)])
    ax3 = fig.add_subplot(254)
    plt.bar([i for i in range(10)], [ret[3, j] for j in range(10)])
    ax4 = fig.add_subplot(255)
    plt.bar([i for i in range(10)], [ret[4, j] for j in range(10)])
    ax5 = fig.add_subplot(256)
    plt.bar([i for i in range(10)], [ret[5, j] for j in range(10)])
    ax6 = fig.add_subplot(257)
    plt.bar([i for i in range(10)], [ret[6, j] for j in range(10)])
    ax7 = fig.add_subplot(258)
    plt.bar([i for i in range(10)], [ret[7, j] for j in range(10)])
    ax8 = fig.add_subplot(259)
    plt.bar([i for i in range(10)], [ret[8, j] for j in range(10)])
    ax9 = fig.add_subplot(2, 5, 10)
    plt.bar([i for i in range(10)], [ret[9, j] for j in range(10)])
    plt.show()

if __name__ == '__main__':
    #imageMat, m, n = loadSingleImage('train/0.jpg')
    #plotImage(imageMat, 'hahaha.jpg', m, n)
    plotDistribution('eclude.txt', 'train.txt')
