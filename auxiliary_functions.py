import cv2
import numpy as np
from matplotlib import pyplot as plt

def rotateImage(image_list, bounding_box):
    x,y,w,h = bounding_box
    new_image_list = []

    for image in image_list:
        if w < h and y == 0:
            image  = cv2.rotate(image, cv2.ROTATE_180)
        elif w > h and x == 0:
            image  = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif w > h:
            image  = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        
        new_image_list.append(image)

    return new_image_list


def findCentroid(contours)->tuple:
    m = cv2.moments(contours)
    
    cx = int(m['m10']/m['m00'])
    xy = int(m['m01']/m['m00'])
    return (cx,xy)


def findContours(image):
    # Encontra o maior valor de camada da imagem
    num_labels, labeled_image = cv2.connectedComponents(image)
    component_mask = np.uint8(labeled_image == 2) * 255
    maior=0
    labelMaior=0
    for component in range(1,num_labels+1):
        area = np.uint8(labeled_image==component)*255
        cv2.countNonZero(area)
        if(cv2.countNonZero(area)>maior):
            maior = cv2.countNonZero(area)
            labelMaior=component

    component_mask= np.uint8(labeled_image == labelMaior) * 255
    # show_image(component_mask)

    # Encontra os contornos do componente
    contours= cv2.findContours(component_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    return contours


def createImageData(image):
    
    # Encontra os contornos do componente
    contours = findContours(image)

    # Encontra o centroide e a caixa da imagem
    cx,cy = findCentroid(contours[0])
    bounding_box = cv2.boundingRect(contours[0])
    x,y,w,h = bounding_box

    # Criando uma imagem binária somente com o contorno
    binary_image_contours = image.copy()
    binary_image_contours.fill(0)
    binary_image_contours = cv2.drawContours(binary_image_contours, contours, -1, 255, thickness=2)

    # Achar os picos dos dedos
    flag = 0
    peaks = []
    binary_image_contours_above_centroid = binary_image_contours[y: cy, x: x+w]
    for index_row, row in enumerate(binary_image_contours_above_centroid):
        if flag == 1:
            flag = 0
            slice_bin_img_contours = binary_image_contours_above_centroid[:index_row, :]
            ret, labels = cv2.connectedComponents(slice_bin_img_contours)
            for label in range(1,ret):
                mask = np.array(labels, dtype=np.uint8)
                mask[labels == label] = 255
            if len(peaks) + 2 == ret:    
                for i_r, r in enumerate(mask):
                    for i_c, c in enumerate(r):
                        if c == 255:
                            peaks.append((x+i_c, y+i_r))
                            break
            if len(peaks) == 4:
                break

        flag += 1
    
    peaks = sorted(peaks, key=lambda tupla: tupla[0])
    peaks.reverse()

    image_data = {
        "fingers_peaks" : peaks,
        "bounding_box"  : bounding_box,
        "centroid"      : (cx, cy),
        "contours"      : contours
        
    }

    return(image_data)


def baseProcessing(src_img, threshold):
    image = src_img.copy()

    # Operação de Binarização
    binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    
    kernel = np.ones((9, 9), np.uint8) 
    # Operação de Abertura
    binary_open_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    # Operação de Fechamento
    binary_close_image = cv2.morphologyEx(binary_open_image, cv2.MORPH_CLOSE, kernel)

    return binary_close_image


def thumbDetector(processed_image, source_image, bounding_box, detection_area: int=30) -> bool:
    has_thumb = False
    src_img = source_image.copy()
    binary_image = processed_image.copy()
    x,y,w,h = bounding_box
    green = (0, 255, 0)
    blue = (0, 0, 255)
    image_with_box = cv2.rectangle(src_img, (x,y), (x+detection_area,y+h), green, 2)
    image_with_box = cv2.rectangle(image_with_box, (x+w-detection_area,y), (x+w,y+h), blue, 2)

    binary_image_left_section = binary_image[y: y+h, x: x+detection_area]
    binary_image_right_section = binary_image[y : y+h, x+w-detection_area : x+w]

    binary_left_sum = int(binary_image_left_section.sum())//255
    binary_right_sum = int(binary_image_right_section.sum())//255
    binary_7percent_sum = int(binary_image.sum()//255*0.07)
    # print(f"binary_7percent_sum = {binary_7percent_sum}, binary_left_sum = {binary_left_sum}, binary_right_sum = {binary_right_sum}")

    if (binary_left_sum < binary_7percent_sum or binary_right_sum < binary_7percent_sum) and (binary_left_sum > binary_7percent_sum or binary_right_sum > binary_7percent_sum):
        has_thumb = True
    
    return has_thumb


def calcEuclidianDistance(peaks_list, centroid, thumb):
    gesture_list = [thumb]
    distance_list = []

    for finger_peak in peaks_list:
        distance = ((finger_peak[0] - centroid[0])**2 + (finger_peak[1] - centroid[1])**2)**(1/2)
        distance_list.append(distance)
    
    max_distance = max(distance_list)
    threshold = max_distance * 0.75

    for peak in distance_list:
        if peak > threshold:
            gesture_list.append(True)
        else:
            gesture_list.append(False)
    
    if len(gesture_list) < 5:
        for i in range(5 - len(gesture_list)):
            gesture_list.append(False)

    return gesture_list