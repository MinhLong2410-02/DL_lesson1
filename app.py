from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from fastapi.middleware.cors import CORSMiddleware



class CustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def train(self, X_train, y_train, X_test, y_test):
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return y_pred
    
    def f1_score(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='macro')
    
    def accuracy_score(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)
    
    def confusion_matrix(self, y_true, y_pred):
        return metrics.confusion_matrix(y_true, y_pred)
    
class Flower(BaseModel):
    sepal_length : float
    sepal_width : float
    petal_length : float
    petal_width : float
    model: str

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This should be restricted to your actual frontend's domain
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/prediction/")
def operate(input:Flower):
    
    input_data = [[input.sepal_length, input.sepal_width, input.petal_length, input.petal_width]]
    model = joblib.load(f'model/{input.model}.pkl')
    X_new = np.array(input_data)
    X_new = X_new.reshape(1,-1)
    prediction = model.predict(X_new)
    if prediction == 0:
        return {
            'target': "setosa",
            'msg': 'This is Iris-setosa',
            'img': 'https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
        }
    elif prediction == 1:
        return {
            'target': "versicolor",
            'msg': 'This is Iris-versicolor',
            'img': 'https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
        }
    else:
        return {
            'target': "virginica",
            'msg': 'This is Iris-virginica',
            'img': 'https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
        }


if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)