import os
from flask import Flask, request, send_from_directory
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/')
def index():
    return "Welcome!"


@app.route('/random_forest_classification', methods=['GET'])
def callModelOne():
    xValue = request.args.get('x')
    value = xValue.split(" ")
    sc = pickle.load(open('standard_scaler.pkl', 'rb'))
    value = sc.transform([value])
    modelOne = pickle.load(open('random_forest_classifier.pkl', 'rb')) # load model1 to the server
    return str(modelOne.predict(value)[0])


@app.route('/random_forest_regression', methods=['GET'])
def callModelTwo():
    xValue = request.args.get('x')
    value = xValue.split(" ")
    modelTwo = pickle.load(open('Random_Forest_Outcome.pkl', 'rb')) # load model2 to the server
    print(modelTwo.predict([value])[0])
    return str(modelTwo.predict([value])[0])


@app.route('/text_processing', methods=['GET'])
def callModelThree():
    xValue = request.args.get('x')
    preprocessing = pickle.load(open('transpose_text_input.pkl', 'rb'))
    xValue = preprocessing(xValue)
    value = [xValue]
    cv = pickle.load(open('count_vectorizer.pkl', 'rb'))
    value = cv.transform(value).toarray()
    modelThree = pickle.load(open('text_processing.pkl', 'rb'))
    return str(modelThree.predict(value)[0])


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
        'favicon.ico', mimetype='image/seed_logo.png')
