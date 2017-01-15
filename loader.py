from numpy import *

def loadSingleImage(f):
    import scipy.misc
    image = mat(scipy.misc.imread(f))
    m, n = shape(image)
    for i in range(m):
        for j in range(n):
            if image[i, j] > 0:
                image[i, j] = 1.0
            else:
                image[i, j] = 0.0
    return reshape(image, m*n)

def loadImages(targetDir = 'train/'):
    import os
    # fileNumber = 100
    fileNumber = sum([len(files) \
        for root,targetDirs,files in os.walk(targetDir)])

    retTmp = []
    for i in range(fileNumber):
        image = loadSingleImage(targetDir + str(i) + '.jpg')
        retTmp.append(image)
    if fileNumber > 0:
        n = shape(retTmp[0])[1]
    ret = mat(zeros((fileNumber, n)))
    for i in range(fileNumber):
        ret[i, :] = retTmp[i]
    return ret
