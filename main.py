# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 11:01:11 2022

@author: Alessio Procelli
"""
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import math


#Dataset
#train
IMG_TRAIN='train-images.idx3-ubyte'
LABEL_TRAIN='train-labels.idx1-ubyte'
#test
IMG_TEST='t10k-images.idx3-ubyte'
LABEL_TEST='t10k-labels.idx1-ubyte'
#Scelta tipologia capi dal set di dati N da 0 a 9
SCELTA_1=1
SCELTA_2=6
#Selta Valori
SCELTA_1_VAL=0
SCELTA_2_VAL=1
#Grandezza mlp
IN_LAY=784
HID_LAY_1=10
HID_LAY_2=10
EPOCHE=10


def inizializza_Dati(images_path,labels_path):
    # Carica dati
    X, y = loadlocal_mnist(
            images_path, 
            labels_path)
  
    # Crea un nuovo dataset filtrando in base alle etichette che mi interessano
    n_examples = len(y)
    n_elementi=0
    for el in range(0,n_examples-1):
        if(y[el]==SCELTA_1 or y[el]==SCELTA_2 ):
            n_elementi+=1
            
    xRidotta=np.ones((n_elementi,len(X[1])))
    yRidotta=np.ones(n_elementi)
    index=0
    
    #Prende solo gli elementi scelti e la loro rappresentazione
    for el in range(0,n_examples-1):
       if(y[el]==SCELTA_1 or y[el]==SCELTA_2 ):
           xRidotta[index]=X[el]
           if(y[el]==SCELTA_1):
               yRidotta[index]=SCELTA_1_VAL
           else:
               yRidotta[index]=SCELTA_2_VAL
           index+=1
    
    labels = np.unique(yRidotta) # restituisce elementi univoci di controllo
    
    return xRidotta, labels, yRidotta


def sigmoid_function(x, der=False):
    if (der==True) : #derivata sigmoide
        f = 1/(1+ np.exp(- x))*(1-1/(1+ np.exp(- x)))
    else : # sigmoide
        f = 1/(1+ np.exp(- x))
    
    return f


def ReLU_function(x, der=False):

    if (der == True): #derivata ReLU
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    
    return f

def train(X_train, Y_train, ):
    eta=0.0095 #arbitrario fisso circa 10^-2
    alfa=0.9 # arbitrario peso momentum
    #Inizializzazione Random pesi e bias
    scala=(np.sqrt(1/(len(X_train[0]+1))))
    w1 = (2*scala)*np.random.rand(HID_LAY_1 , X_train.shape[1]) -scala # Strato 1
    w1_old=np.zeros((HID_LAY_1 , X_train.shape[1])) # per momentum
    b1 = scala*np.random.rand(HID_LAY_1)

    w2 = (2*scala)*np.random.rand(HID_LAY_1 , HID_LAY_2) - scala  # Strato 2
    w2_old=np.zeros((HID_LAY_1 , HID_LAY_2)) #per momentum
    b2 = scala* np.random.rand(HID_LAY_2)

    wOut = (2*scala)*np.random.rand(HID_LAY_2) - scala  # Output 
    wOut_old=np.zeros(HID_LAY_2)
    bOut = scala* np.random.rand(1)
    absloss=[]
    vec_y = []

    # Loop su i pixel
    
    for e in range(0,EPOCHE):
        mu=np.zeros(X_train.shape[0]-1)
        for I in range(0,X_train.shape[0]-1): 
            
            # Prende immagine riga
            x = X_train[I]/255

            # Feed forward
            z1 = ReLU_function(np.matmul(w1, x) + b1) # output layer 1 
            
            z2 = ReLU_function(np.matmul(w2, z1) + b2) # output layer 2
            
            y = sigmoid_function(np.matmul(wOut, z2) + bOut) # Output of the Output layer
            
            #Errore Croos Entropy

            if(Y_train[I]==1):
                loss=-math.log(y)
            else:
                loss=-math.log(1-y)
            
            delta_Out = y-Y_train[I] 
    
            #2.3: Backpropagate
            delta_2 = delta_Out * wOut * ReLU_function(z2,True) #Layer 2 Errore
            delta_1 = np.dot(delta_2, w2) * ReLU_function(z1,True) #Layer 1 Errore
            
            # 3: Gradient descent : Aggiornamento pesi
            current_wOut=eta*delta_Out*z2 #risultato presente per momentum
            wOut = wOut - current_wOut-(alfa*wOut_old)
            wOut_old=current_wOut
            bOut = bOut - eta*delta_Out
            
            current_w2=eta*np.kron(delta_2, z1).reshape(HID_LAY_2,HID_LAY_1)
            w2 = w2 - current_w2-(alfa*w2_old) #Layer 2
            w2_old=current_w2
            b2 = b2 - eta*delta_2
            
            current_w1=eta*np.kron(delta_1, x).reshape(HID_LAY_1, x.shape[0])
            w1 = w1 - current_w1-(alfa*w1_old) #Layer 1
            w1_old=current_w1
            b1 = b1 - eta*delta_1
            
            # 4. Registra loss
            mu[I]=(loss)
            
        absloss.append(np.mean(mu)) # andamento epoca i-esima
        
       #Disegna grafico
   
    plt.scatter(np.arange(1,EPOCHE+1),absloss)
    plt.title('Loss in epoche', fontsize=20)
    plt.xlabel('Epoche', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()
    
    return w1, b1,w2,b2, wOut, bOut, mu
 

def prediction(X_test, w1, b1,w2,b2, wOut, bOut):
    
    pred = []
    
    for I in range(0, X_test.shape[0]): #loop 
        #input the data 
        x =X_test[I]/255
        #Feed forward
        z1 = ReLU_function(np.matmul(w1, x) + b1) # output layer 1 
        z2 = ReLU_function(np.matmul(w2, z1) + b2) # output layer 2
        y = sigmoid_function(np.matmul(wOut, z2) + bOut) # Output of the Output l
        # if y < 0.5 l'output è zero, altrimenti è 1
        if(y<0.5):
            pred.append(0)
        else:
            pred.append(1)
         
    return np.array(pred);


# Estrapola dataset
xTrain,labelsTrain, yTrain  = inizializza_Dati(IMG_TRAIN,LABEL_TRAIN)
xTest,labelsTest, yTest  = inizializza_Dati(IMG_TEST,LABEL_TEST)
# Allena
w1,b1,w2,b2,wOut,bOut,mu=train(xTrain, yTrain)
# Predice
result=prediction(xTest,w1,b1,w2,b2,wOut,bOut)

#valuta accuratezza
giuste=0
sbagliate=0
for i in range(0,len(yTest)-1):
   if(yTest[i]==result[i]):
       giuste+=1
   else:
       sbagliate+=1

print("giuste :"+ str(giuste))
print("Sbagliate: "+str(sbagliate))

risultati = {'Giuste':giuste, 'Sbagliate':sbagliate}
plt.bar(list(risultati.keys()), list(risultati.values()),color=['green','red',])
plt.title("Risultati")
plt.show()

