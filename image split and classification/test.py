import numpy as np
import pylab
import mahotas as mh
import image_split as split

image = mh.imread('D:\Documents\GitHub\image split and classification\matching-game2.png')
#mg stands for "matching game"

position,info = split.locate(image)
print position,info

label = split.split(image,position,info)

for i in range(len(label)):
    print label[i]