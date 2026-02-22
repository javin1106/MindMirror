import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class EmotionDetector:
    def __init__(self, model_path: str, tokenizer_path: str, label_path: str):
        self.max_len = 50
        self.model = tf.keras.models.load_model(model_path)

        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(label_path, 'rb') as f:
            self.label_classes = pickle.load(f)
        
    def preprocess(self, text: str):
        text = text.lower().strip()
        seq = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, padding = "post", maxlen = self.max_len)
        return padded

    def predict(self, text: str) -> dict:
        if not text or len(text.strip()) == 0:
            return {
                "label": "unknown",
                "confidence": 0.0
            }
        
        padded = self.preprocess(text)
        proba = self.model.predict(padded, verbose = 1)[0]
        max_index = int(np.argmax(proba))
        confidence = float(proba[max_index])
        label = str(self.label_classes[max_index])

        return {"label": label, "confidence": confidence}
    
#------------
# For testing
#------------

# detector = EmotionDetector(
#     model_path="artifacts/model.keras",
#     tokenizer_path="artifacts/tokenizer.pkl",
#     label_path="artifacts/label_classes.pkl"
# )

# print(detector.predict("I feel completely drained and hopeless."))
# print(detector.predict("I am so excited about my future!"))
# print(detector.predict("I can't believe this happened."))
