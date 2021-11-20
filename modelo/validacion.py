

from tensorflow.keras.models import load_model
model = load_model('model.h5')
while True:
    try:
        datos = []
        datos_in = input("Ingrese el arreglo de datos")
        sp = datos_in.split(",")
        for i in sp:
            datos.append(int(i))
        print(datos)
        yhat = model.predict([datos])
        print('Predicted: %.3f' % yhat[0])
    except Exception:
        print("Error en los datos intentalo de nuevo")
        pass