from numpy import *

def loadSingleImage(f):
    import scipy.misc
    image= mat(scipy.misc.imread(f))
    m, n = shape(image)
    return reshape(image, m*n)

def loadImages(total, dir = 'test'):
    ret = []
    for i in range(total):
        ret.append(dir + i + ".jpg")
    return ret
