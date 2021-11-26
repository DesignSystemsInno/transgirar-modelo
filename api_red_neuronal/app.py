
from fastapi import FastAPI
from tensorflow.keras.models import  load_model
model = load_model('model2.h5')


app = FastAPI()
origins = [
 "http://18.116.59.243",
 "http://18.191.58.131",
 "http://p2transredneuronal.ml"
]
app.add_middleware(
 CORSMiddleware,
 allow_origins=origins,
 allow_credentials=True,
 allow_methods=["*"],
 allow_headers=["*"],
)
@app.get('/')
def welcome():
    return {
        "message":"welcome!"
    }

@app.post('/perceptron/{x1}/{x2}/{x3}/{x4}')
def procces(x1:int,x2:int,x3:int,x4:int):
    """ FastAPI 
    
    Arg:
    
        x1: Recibe 1 si se abrio la tapa o 0 si no se abrio la tapa
        x2: Recibe 1 si la tapa fue abierta fuera de una estación de gasolina o 0 si la tapa se abrio dentro de una estación de gasolina
        x3: Recibe 1 si tenia que tanquear o 0 si no tenia que tanquear
        x4: Recibe 1 si llega con la cantidad de gasolina aprox que deberia de haber llegado o 0 Si no llego con la cantidad de gasolina que deberia de haber llegado 
    Returns:
    
        predicted: Probabilidad de que hayan robado gasolina
    """
    datos = [x1,x2,x3,x4]
    yhat = model.predict([datos])
    return {
        "predicted":'%.3f' % yhat[0]
    }
    