# Alunos: Otávio Bertucini e Isadora Chandelier

import sys
import numpy as np
import cv2
import math

INPUT_IMAGE = './images/0.bmp'
BACK_IMAGE = './back.bmp'

GREEN = [0, 255, 0]
R = 0
G = 1
B = 2

GAMMA = 2

THRESHOLD = 20

# Correção Gama, baseado em https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
def contrast(img):

    img = img.astype(np.uint8)

    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, GAMMA) * 255, 0, 255)

    new_image = cv2.LUT(img, lookUpTable)

    return new_image

# metodo baseado em https://www.compuphase.com/cmetric.htm
def colorDelta(color):

    red = (color[R] + GREEN[R]) / 2
    delta_red = color[R] - GREEN[R]
    delta_green = color[G] - GREEN[G]
    delta_blue = color[B] - GREEN[B]

    delta = math.sqrt(((2 + (red / 255)) * delta_red**2) + (4 * delta_green ** 2) + ((2 + (255 - red) / 255) * delta_blue ** 2))
    # delta = math.sqrt((delta_red**2) + (delta_green**2) + (delta_blue**2))

    return delta


def generateMatte(img):

    matte = np.zeros((img.shape[0], img.shape[1]))

    # img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            color = img[y][x]
            # print(color)
            delta = colorDelta(color)
            matte[y][x] = delta

    return matte


def removeBackground(img, back, matte):

    result = np.zeros((img.shape[0], img.shape[1], 3))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if(matte[y][x] > THRESHOLD):
                result[y][x] = img[y][x]
                continue
            result[y][x] = ( 1 - (matte[y][x] / 255)) * back[y][x]

    print(result.shape)
    return result


def main():

    # Abre a imagem em escala de cinza.
    original = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    cv2.imwrite('original.bmp', original)
    img = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    back = cv2.imread(BACK_IMAGE, cv2.IMREAD_COLOR)
    if back is None:
        print('Erro abrindo o back.\n')
        sys.exit()

    resized = cv2.resize(
        back, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('resized.bmp', resized)

    matte = generateMatte(original)
    matte = cv2.normalize(matte, matte, 0, 1, cv2.NORM_MINMAX)
    matte = matte * 255
    
    kernel = np.ones((4,4),np.uint8)
    matte = cv2.morphologyEx(matte, cv2.MORPH_OPEN, kernel)

    matte = cv2.normalize(matte, matte, 0, 1, cv2.NORM_MINMAX)
    matte = matte * 255
    # equ = cv2.equalizeHist(matte)
    # print(matte)
    cv2.imwrite('matte.bmp', matte)

    with_contrast = contrast(matte)
    # with_contrast = cv2.convertScaleAbs(matte*255, alpha=ALPHA, beta=0)
    cv2.imwrite('contrasted.bmp', with_contrast)

    result = removeBackground(original, resized, with_contrast)
    cv2.imwrite('result.bmp', result)


if __name__ == '__main__':
    main()

# ===============================================================================
