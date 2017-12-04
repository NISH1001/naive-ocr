#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageOps

BASEWIDTH = 28

def resize(img):
    wpercent = ( BASEWIDTH / float(img.size[0]) )
    hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((BASEWIDTH,BASEWIDTH), Image.ANTIALIAS)

def binarize(img):
    return img.convert('L')

def load_image(filename):
    return Image.open(filename)

def featurize(filename):
    img = resize(load_image(filename))
    img = binarize(img)
    img = ImageOps.invert(img)
    #img.save("data/converted.jpg")
    data = np.copy(np.asarray(img))
    data[data<25] = 0
    data = data.reshape( (1, BASEWIDTH*BASEWIDTH) )
    return data

def one_hot_encoder(a):
    length = len(a)
    b = np.zeros( (length, 10) )
    b[np.arange(length), a] = 1
    return b

def main():
    test = featurize("data/test.jpg")
    print(test, test.shape)
    print(np.array(test).shape)

if __name__ == "__main__":
    main()

