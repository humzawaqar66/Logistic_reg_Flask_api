from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

with open('IRIS (1).pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    
    json_data = request.get_json()

    
    data = pd.DataFrame(json_data, index=[0])
    
    prediction = loaded_model.predict(data)
    
    prediction = prediction.tolist()

    response = {'prediction': prediction}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
