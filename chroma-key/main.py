## Alunos: Otávio Bertucini e Isadora Chandelier

import sys
import timeit
import numpy as np
import cv2
from matplotlib.pyplot import boxplot
import matplotlib.pyplot as plt
import math

INPUT_IMAGE = './images/8.bmp'
GREEN = [0, 255, 0]
R = 0
G = 1
B = 2

## metodo retidado de https://www.compuphase.com/cmetric.htm
def colorDelta(color):

    red = (color[R] + GREEN[R]) / 2
    delta_red = color[R] - GREEN[R]
    delta_green = color[G] - GREEN[G]
    delta_blue = color[B] - GREEN[B]

    delta = math.sqrt(((2 + (red / 255)) * delta_red**2) + (4 * delta_green ** 2) + ((2 + (255 - red) / 255) * delta_blue ** 2))

    return delta

def removeBackground(img):

    matte = np.zeros((img.shape[0], img.shape[1]))

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            color = img[x][y]
            delta = colorDelta(color)
            matte[x][y] = delta

    return matte


def main ():

    # Abre a imagem em escala de cinza.
    original = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    matte = removeBackground(img)
    matte = cv2.normalize(matte, matte, 0, 1, cv2.NORM_MINMAX)
    print(matte)
    cv2.imwrite('matte.bmp', matte*255)


if __name__ == '__main__':
    main ()

#===============================================================================