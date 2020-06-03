from flask import Flask, request, jsonify
import numpy as np
from json import JSONEncoder
import json
from flask_cors import CORS
import requests
import matplotlib.pyplot as plt
import pickle


app = Flask(__name__)
CORS(app)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def prepareImage(img):
    img = np.array(img, dtype=np.float32)
    img = img/255.0
    rgb_weights = [0.2989, 0.5870, 0.1140]
    img = np.dot(img, rgb_weights)
    img = img.reshape(1, -1)
    return img

def getpred(img):
     session = requests.Session()
     r = session.post("https://yzxed7zf71.execute-api.us-east-1.amazonaws.com/default/retinopathybin",json=img)
     return json.dumps(r.json())

@app.route('/',methods=["POST"])
def hello_world():
    img = plt.imread(request.files['file'])
    mode  = request.form['mode']
    img = prepareImage(img)
    data = {"data": img,"mode":mode} 
    encodedNumpyData = json.dumps(data, cls=NumpyArrayEncoder)
    return getpred(encodedNumpyData)
@app.route('/heart', methods=["POST"])
def heartfunction():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    cp = int(request.form['cp'])
    trestbps = int(request.form['trestbps'])
    chol = int(request.form['chol'])
    fbs = int(request.form['fbs'])
    restecg = int(request.form['restecg'])
    thalach = int(request.form['thalach'])
    exang = int(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = int(request.form['slope'])
    ca = int(request.form['ca'])
    thal = int(request.form['thal'])
    model = pickle.load(open('model1.pkl', 'rb'))
    print("yahan tak aagaye")
    labels = {0: 'No Heart Disease', 1: "Heart Disease Present"}
    y = model.predict([[age, sex, cp, trestbps, chol, fbs,restecg, thalach, exang, oldpeak, slope, ca, thal]])
    return json.dumps(labels[int(y)])

if __name__ == "__main__":
    app.run()
