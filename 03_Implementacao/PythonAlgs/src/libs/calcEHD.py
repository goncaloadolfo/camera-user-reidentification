import numpy as np
import cv2

def dist(pat, un):
    pattern = np.mat(pat)
    unit = np.mat(un)
    dummy, Np = pattern.shape
    dummy, Nu = unit.shape
    p2 = np.sum(np.power(pattern,2),0)
    u2 = np.sum(np.power(unit,2),0).T
    up = np.dot(unit.T,pattern)
    elem1 = np.tile(p2,(Nu,1))
    elem2 = np.tile(u2,(1,Np))
    return  elem1 -2*up + elem2

def procBlock(block):
    aH = np.array([(1,-1,1,-1)])
    aV = np.array([(1,1,-1,-1)])
    a45 = np.array([(np.sqrt(2),0,0,-np.sqrt(2))])
    a135 = np.array([(0,+np.sqrt(2),-np.sqrt(2),0)])
    aND = np.array([(2,-2,-2,+2)])
    tresh = 20
    
    nlin, ncol = block.shape
#    print nlin,ncol
    dimX = np.int(np.round(nlin/2))
    nlinAux = 2*dimX
    dimY = np.int(np.round(ncol/2))
    ncolAux = 2*dimY
    J = np.array([])

    for x in range(0,nlinAux,dimX):
        for y in range(0,ncolAux,dimY):
            subBlock = block[x:x+dimX, y:y+dimY]
            J = np.append(J,subBlock.mean())
    filt = np.abs(np.array([tresh,np.sum(J*aV),\
                np.sum(J*aH),np.sum(J*a45),\
                np.sum(J*a135),np.sum(J*aND)]))
    return filt.argmax()                
        

def procSubImage(subImage):
    nlin, ncol = subImage.shape
    n = np.round(np.sqrt(1100*nlin/ncol))
    blockNlin = np.int(np.round(nlin/n))
    sIp = np.array([])
    
    maxLin = np.int(np.floor(nlin/blockNlin)*blockNlin)
    maxCol = np.int(np.floor(ncol/blockNlin)*blockNlin)

#    print n,blockNlin,nlin,maxLin,ncol,maxCol
    for x in range(0, maxLin, blockNlin):
        for y in range(0, maxCol, blockNlin):
#            print 'X - > ',x,':',x+blockNlin,' Y - >',y,':',y+blockNlin
            block = subImage[x:x+blockNlin, y:y+blockNlin]
            sIp = np.append(sIp,procBlock(block))
    lhist,bins = np.histogram(sIp,np.arange(-0.5,6,1))
    return np.array(lhist[1:],dtype=np.float64)/np.sum(lhist)


def compEHD(gray):
    quatizeTable= np.array([[0.010867,0.057915,0.099526,0.144849,0.195573,0.260504,0.358031,0.530128],
                            [0.012266,0.069934,0.125879,0.182307,0.243396,0.314563,0.411728,0.564319],
                            [0.004193,0.025852,0.046860,0.068519,0.093286,0.123490,0.161505,0.228960],
                            [0.004174,0.025924,0.046232,0.067163,0.089655,0.115391,0.151904,0.217745],
                            [0.006778,0.051667,0.108650,0.166257,0.224226,0.285691,0.356375,0.450972]])
    
    nlin, ncol = gray.shape
    deltaX = np.int(np.round(nlin/4))
    nlinAux = 4*deltaX
    deltaY = np.int(np.round(ncol/4))
    ncolAux = 4*deltaY
    
    ehd = np.array([])
    
    for x in range(0, nlinAux, deltaX):
        for y in range(0, ncolAux, deltaY):
            subImage = gray[x:x+deltaX, y:y+deltaY]
            ehd = np.append(ehd,procSubImage(subImage))
    
    ehdQuant = np.zeros(ehd.shape,dtype=np.int)
    for k in range(5):
        ehdQuant[k:-1:5] = np.argmin(dist(ehd[k:-1:5],quatizeTable[k,:]),axis=0)
    
    return ehdQuant
