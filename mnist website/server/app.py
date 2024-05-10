# flask backend server
from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load the ml model
model = pickle.load(open('./mnist_model.pkl', 'rb'))


# default api
@app.route('/')
def home():
    return {}


# main route
@app.route('/predict', methods=['POST'])
def make_prediction():
    try:
        # convert file to np array
        uploaded_file = request.files['file']
        image_stream = uploaded_file.read()
        nparr = np.frombuffer(image_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) 
        img_resized = cv2.resize(img, (28, 28))       
        img = np.array(img_resized)
        img = img.reshape((1, 28, 28, 1)).astype('float32') / 255  
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        print(predicted_class)

        return jsonify({'predicted_class': int(predicted_class)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)

