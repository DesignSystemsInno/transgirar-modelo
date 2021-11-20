# Load dependencies
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
from random import  randint
from matplotlib import pyplot
from tensorflow.keras.optimizers import SGD
import os
epocas = 100
flag = 0

os.environ['KMP_DUPLICATE_LIB_OK']='True'
path_prediction1 = "C:/Users/Diego/OneDrive - Universidad EAFIT/Semestre 7-2/Proyecto Integrador 2/Transgirar/red_neuronal_ejemplo/Perceptron/prueba/prediction2.txt"
dataset = "C:/Users/Diego/OneDrive - Universidad EAFIT/Semestre 7-2/Proyecto Integrador 2/Transgirar/red_neuronal_ejemplo/Perceptron/prueba/prediction3.txt"
dataset = np.loadtxt(dataset, delimiter=",",usecols=(0,1,2,3,4))
np.random.shuffle(dataset)
# Separate features and targets
dataset = np.array(dataset)
X = dataset[:20000, 0:4]
Y = dataset[:20000, 4]
print(X)
print(Y)    
sgd = SGD(learning_rate=0.001, momentum=0.9)
model = Sequential()
model.add(Dense(100, input_dim=4, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error',
            optimizer=sgd,
            metrics=['binary_accuracy'])

history = model.fit(X, Y, epochs=epocas)
X = dataset[20000:, 0:4]
Y = dataset[20000:, 4]
loss = model.evaluate(X, Y, verbose=0)
print(loss)
print(history.history['loss'])
# pyplot.title('Learning Curves')
# pyplot.xlabel('Epoch')
# pyplot.ylabel('Cross Entropy')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.legend()
# pyplot.show()

path_comprobar = "C:/Users/Diego/OneDrive - Universidad EAFIT/Semestre 7-2/Proyecto Integrador 2/Transgirar/red_neuronal_ejemplo/Perceptron/prueba/comprobar.txt"
path_comprobar = np.loadtxt(path_comprobar, delimiter=",",usecols=(0,1,2,3))
np.random.shuffle(path_comprobar)
dataset = np.array(path_comprobar)
X = dataset[:, 0:4]
yhat = model.predict(X).round()
for i in yhat:
    f = open (path_prediction1, 'a')
    datos = str(int(i[0]))
    f.write(datos+'\n') 
    f.close()


print(yhat)
model.save('model2.h5')
path_prediction = "C:/Users/Diego/OneDrive - Universidad EAFIT/Semestre 7-2/Proyecto Integrador 2/Transgirar/red_neuronal_ejemplo/Perceptron/prueba/prediction2.txt"
comprobar = "C:/Users/Diego/OneDrive - Universidad EAFIT/Semestre 7-2/Proyecto Integrador 2/Transgirar/red_neuronal_ejemplo/Perceptron/prueba/comprobar.txt"
comprobar = np.loadtxt(comprobar, delimiter=",", skiprows=0, usecols=(4))
path_prediction = np.loadtxt(path_prediction, usecols=(0))
# Separate features and targets
comprobar = np.array(comprobar)
path_prediction = np.array(path_prediction)
contador = 0
print(comprobar)
for i in range(len(comprobar)):
    if comprobar[i] == path_prediction[i]:
        contador +=1

print(contador)
# print(epocas)
# if(contador >flag):
#     flag = contador
#     epocas = epocas + 100
#     f = open(path_prediction1, 'w')
#     f.close()
# elif(contador<flag):
#     epocas = epocas-10
#     f = open (path_prediction1, 'w')
#     f.close()
# else:
#     break
