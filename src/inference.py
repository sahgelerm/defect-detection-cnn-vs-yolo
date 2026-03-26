import argparse
import numpy as np
import tensorflow as tf
import cv2

CLASS_NAMES = ["chern_toch", "vozd_puz", "nedoliv", "non_defects", "prigar"]

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    model = tf.keras.models.load_model("models/my_model.keras")

    img = preprocess_image(args.image)
    preds = model.predict(img)

    class_id = np.argmax(preds)
    print(f"Predicted class: {CLASS_NAMES[class_id]}")
    print(f"Confidence: {preds[0][class_id]:.4f}")

if __name__ == "__main__":
    main()

