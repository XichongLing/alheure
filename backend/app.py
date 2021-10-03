import math
from flask import current_app, flash, jsonify, make_response, redirect, request, url_for, Flask
import pickle
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return 'Hello World!'

@app.route('/predict', methods=['GET'])
def predict():
    day = request.args.get('day')
    depblock = request.args.get('depblock')
    carrier = request.args.get('carrier')
    depairport = request.args.get('depairport')
    prevairport = request.args.get('prevairport')
    model = pickle.load(open('model.pkl', 'rb'))
    enc = pickle.load(open('enc.pkl', 'rb'))
    x = enc.transform([[day, depblock, carrier, depairport, prevairport]]).toarray()
    [res] = model.predict(x).tolist()
    response = make_response(jsonify({"delay": res}), 200,)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response