from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import mediapipe as mp

app = Flask(__name__)

# Load the model and define actions
model = load_model('realtimev7.h5')
actions = ['aku', 'apa', 'bagaimana', 'berapa', 'di', 'F', 'halo', 'I', 'J', 'K', 'kamu', 'kapan', 'ke', 'mana', 'mereka', 'nama', 'saya', 'siapa', 'Y', 'yang']

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)


@app.route('/translate', methods=['POST'])
def translate():
    try:
        # Get the input keypoints array from the request
        data = request.json['keypoints']
        sequence = np.array(data, dtype=np.float32)
        sequence = sequence.reshape((1, 30, 126))
        translation_index = np.argmax(model.predict(sequence))
        translation = actions[translation_index]
        response = {
            'translation': translation
        }
        return jsonify(response)

    except KeyError as e:
        error_message = f"KeyError: '{e.args[0]}' is missing in the request data"
        response = {
            'error': error_message
        }
        return jsonify(response), 400

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        response = {
            'error': error_message
        }
        return jsonify(response), 500

@app.route('/')
def index():
    return 'Hello, Flask!'

if __name__ == '__main__':
    app.run(debug=True)
