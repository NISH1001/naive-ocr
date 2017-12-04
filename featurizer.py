#!/usr/bin/env python3

import numpy as np
from PIL import Image, ImageOps

BASEWIDTH = 28
BASE_PATH = "data/"
THRESHOLD_VALUE = 254

def resize(img):
    wpercent = ( BASEWIDTH / float(img.size[0]) )
    hsize = int((float(img.size[1])*float(wpercent)))
    return img.resize((BASEWIDTH,BASEWIDTH), Image.ANTIALIAS)

def binarize(img):
    return img.convert('L')

def convert_img_to_array(img):
    img_data = np.asarray(img)
    threshold_data = (img_data > THRESHOLD_VALUE ) * 1.0
    return threshold_data.flatten()

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


def main():
    test = featurize("data/test.jpg")
    print(test, test.shape)
    print(np.array(test).shape)

if __name__ == "__main__":
    main()

