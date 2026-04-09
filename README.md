# Car View Classifier

This repository provides code and models to classify the view/angle of a car in an image. The supported classes are: `Front`, `Front_Left`, `Front_Right`, `Rear`, `Rear_Left`, `Rear_Right`, and `Unknown`.

## Project Structure

- **Training Notebooks**: 
  - `EfficientNetB0.ipynb`
  - `MobileNetV3Large.ipynb`
- **Pre-trained TFLite Models**:
  - `best_efficientnetb0_car_view.tflite` (Default model used for inference)
  - `best_mobilenetv3large_car_view.tflite`
- **Inference Script**: 
  - `test_run.py` - Script to run batch predictions on images in a test directory.

## Requirements

Ensure you have Python installed along with the following packages:
- `tensorflow`
- `numpy`
- `pandas`

You can install the required packages using pip:
```bash
pip install tensorflow numpy pandas
```

## How to Run

1. **Prepare your test images**: Create a folder named `test` in the root directory (same level as `test_run.py`) and place the car images you want to classify inside this folder. Recomended image formats: `.jpg`, `.jpeg`, `.png`.

   ```
   car-view-classifier/
   │
   ├── test/
   │   ├── car1.jpg
   │   ├── car2.png
   │   └── ...
   │
   ├── best_efficientnetb0_car_view.tflite
   ├── test_run.py 
   └── ...
   ```

2. **Run the inference script**:
   ```bash
   python test_run.py
   ```

3. **View the results**: 
   The script will process each image in the `test` folder and display the prediction in the terminal. Once complete, it will save the results to a file named `predictions.csv` in the current directory.

## Output Format

The `predictions.csv` will contain:
- `image_name`: Name of the processed image file.
- `prediction`: Suggested car view class (e.g., `Front_Right`, `Rear`, etc.).
