#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  './images/205.bmp'

NEGATIVO = False
THRESHOLD = 0.60
ALTURA_MIN = 100
LARGURA_MIN = 100
N_PIXELS_MIN = 50

sys.setrecursionlimit(1500)

#===============================================================================


def binariza(img, threshold):

    return 255 - cv2.adaptiveThreshold(255 - img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 3)

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):

    # Use a abordagem com flood fill recursivo.

    component_count = 1
    components = []

    check_matrix = np.zeros(img.shape)

    def flood(x, y, component):
        if(x < 0 or y < 0 or x > img.shape[0] - 1 or y > img.shape[1] - 1):
            return None

        if(img[x][y][0] == 0 or check_matrix[x][y][0] == 1):
            return None

        check_matrix[x][y][0] = 1
        component['n_pixels'] += 1

        if(component['B'] is None or x > component['B']):
            component['B'] = x
        
        if(component['T'] is None or x < component['T']):
            component['T'] = x

        if(component['L'] is None or y < component['L']):
            component['L'] = y

        if(component['R'] is None or y > component['R']):
            component['R'] = y

        flood(x+1, y, component)
        flood(x-1, y, component)
        flood(x, y+1, component)
        flood(x, y-1, component)

        return None

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):

            if(check_matrix[x][y][0] == 1):
                continue

            if(img[x][y][0] == 1):
                component = {
                    'label': component_count,
                    'n_pixels': 0,
                    'T': None,
                    'L': None,
                    'B': None,
                    'R': None
                }
                flood(x, y, component)
                components.append(component)
                component_count += 1

    return list(filter(lambda component: component['n_pixels'] > n_pixels_min, components))

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    original = cv2.imread (INPUT_IMAGE, cv2.IMREAD_COLOR)
    img = original.copy()
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.

    img = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    # Mantém uma cópia colorida para desenhar a saída.
    img_out = img


    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img    

    ## binariza via limiar adaptativo
    img = binariza (img, THRESHOLD)

    ## tirar ruído via abertura (morfologia)
    kernel = np.ones((6,6),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    cv2.imwrite ('01 - binarizada.png', img)

    kernel = np.ones((3,3),np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=2)
    cv2.imwrite ('01 - erodida.png', img_erosion)

    duo = np.zeros(img.shape)

    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            if(img[y][x][0] == 255):
                duo[y][x][0] = original[y][x][0]

    cv2.imwrite ('01 - duo.png', duo)

    a,ret = cv2.threshold(duo,200,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    ret = cv2.erode(ret, kernel, iterations=1)

    cv2.imwrite ('01 - ret.png', ret)

# 
    # output = np.zeros(original.shape)
    # output = cv2.normalize(duo, output, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite ('01 - normalized.png', duo)

    # start_time = timeit.default_timer ()
    # componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    # n_componentes = len (componentes)
    # print ('Tempo: %f' % (timeit.default_timer () - start_time))
    # print ('%d componentes detectados.' % n_componentes)

    # # Mostra os objetos encontrados.
    # for c in componentes:
    #     cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    # # cv2.imshow ('02 - out', img_out)
    # cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================