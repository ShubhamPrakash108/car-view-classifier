import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

TFLITE_MODEL_PATH = "best_efficientnetb0_car_view.tflite"
TEST_FOLDER = "test"                    
IMG_SIZE = 224

class_names = ['Front', 'Front_Left', 'Front_Right', 'Rear', 'Rear_Left', 'Rear_Right', 'Unknown']

print(f" Loading model: {TFLITE_MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!\n")


def predict_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img.numpy())
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])

    return predicted_class, confidence

if __name__ == "__main__":
    test_dir = Path(TEST_FOLDER)

    if not test_dir.exists():
        print(f"Error: Test folder '{TEST_FOLDER}' not found!")
        print("Please put your test images in a folder named 'test' and try again.")
        exit(1)

    image_files = [f for f in test_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]

    if not image_files:
        print(f"No images found in '{TEST_FOLDER}' folder!")
        exit(1)

    print(f"Found {len(image_files)} images. Starting prediction...\n")

    results = []
    for img_file in sorted(image_files):
        pred_class, conf = predict_image(str(img_file))
        
        results.append({
            "image_name": img_file.name,
            "prediction": pred_class
        })

        print(f"{img_file.name:40} → {pred_class}")

    df = pd.DataFrame(results)
    df.to_csv("predictions.csv", index=False)

    print("\n" + "="*85)
    print(" SUCCESS! Inference completed.")
    print(f"   Processed {len(df)} images")
    print(f"   Results saved to → predictions.csv")
    print(df.head(10))
