import cv2
import numpy as np
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture("/dev/video0")
threshold = 150
frame=None

def pontoMaisEsquerda(imagem):
    for linha in range(20, 640):
        indices_brancos = np.where(imagem[linha, 25:480] == 255)[0]
        if indices_brancos.size > 0:
            coluna = indices_brancos[0] + 25
            return (linha, coluna)
            
def pontoMaisDireita(imagem):
    for linha in range(639, 20, -1):
        indices_brancos = np.where(imagem[linha, 25:480] == 255)[0]
        if indices_brancos.size > 0:
            coluna = indices_brancos[-1] + 25
            return (linha, coluna)
            
def pontoMaisAlto(imagem):
    for coluna in range(25, 480):
        indices_brancos = np.where(imagem[20:640, coluna] == 255)[0]
        if indices_brancos.size > 0:
            linha = indices_brancos[0] + 20
            return (linha, coluna)

def preparacaoImagemEContornos():
    global frame
    global imgOriginal
    cv2.threshold(frame,threshold,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,frame)
    kernel = np.ones((3, 3), np.uint8) 
    cv2.erode(frame, kernel,frame)
    cv2.dilate(frame,kernel,frame)
    
    #fiz isso para poder tratar a mao como um componnet
    frame = cv2.bitwise_not(frame)
    cv2.imshow("imagem binarizada ", frame)
    num_labels, labeled_image = cv2.connectedComponents(frame)

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
    # Encontra os contornos do componente
    contours, hierarchy = cv2.findContours(component_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #menor forma para representar os contornos
    contornoOrdendo = cv2.convexHull(contours[0])
    cnt = contours[0]
    cx,cy = acharCentroidEPicos(cnt)
    
    contorno_filtrado = np.array([ponto for ponto in contornoOrdendo if ponto[0][1] < cy])
    Valorpico =0
    indicePico =0
    


    
    
    
    x,y,w,h = cv2.boundingRect(cnt)
    
    

    imgOriginal = cv2.drawContours(imgOriginal, contours, -1,(0,255,0),thickness=2)
    orientacao = "Horizontal" if w>h else "Vertical"
    cv2.putText(imgOriginal,orientacao,(188,513),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),1)
    
    # print(pontoMaisAlto(frame))
    # cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)

    
def acharCentroidEPicos(coutornos)->None:
    m = cv2.moments(coutornos)
    
    cx = int(m['m10']/m['m00'])
    xy = int(m['m01']/m['m00'])
    return (cx,xy)
    

    


while True:
    # Captura o quadro
    ret, frame = cap.read()
    imgOriginal  = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame  = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)

    preparacaoImagemEContornos()
    esquerda = pontoMaisEsquerda(frame)
    direita = pontoMaisDireita(frame)
    alto = pontoMaisAlto(frame)
    
    
    # Exibe o componente usando cv2.imshow
    
    cv2.imshow(f'Img Original', imgOriginal)
    
    if cv2.waitKey(1)=="27":
        
        break
    
# Libera os recursos
cap.release()
cv2.destroyAllWindows()
