# Gender-&-Age-Prediction

This project uses OpenCV and deep learning models to detect faces in images and predict their age and gender. The results are displayed on the image and saved in a designated folder.

## Features

- **Face Detection**: Detects faces in images using a pre-trained OpenCV DNN model.
- **Age Prediction**: Predicts the age range of the detected face using a Caffe-based deep learning model.
- **Gender Prediction**: Predicts the gender (Male or Female) of the detected face using a Caffe-based deep learning model.
- **Image Processing**: Processes each image in a specified folder and saves the output with detected faces, predicted age, and gender.

## Technologies/Tools

* Python 3.x
* OpenCV `pip install opencv-python`
* numpy `pip install numpy`

![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=FFFF00)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)

## Setup

1. **Clone the repository**:

    ```bash
    git clone https://github.com/LasithaAmarasinghe/Gender-and-age-prediction.git
    ```

2. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the required pre-trained models**:

    - **Face Detection Model**: `opencv_face_detector.pbtxt` and `opencv_face_detector_uint8.pb`
    - **Age Model**: `age_deploy.prototxt` and `age_net.caffemodel`
    - **Gender Model**: `gender_deploy.prototxt` and `gender_net.caffemodel`

4. **Prepare your image folder**:

    - Create a folder named `images` and place the images you want to process in this folder.
    - Supported image formats: `.png`, `.jpg`, `.jpeg`.

5. **Create the output folder**:

    - The processed images will be saved in the `detected_images` folder. This folder will be created automatically if it does not exist.
  
## How This Works

1. **Step 1**: The script loads pre-trained deep learning models for face detection, age prediction, and gender prediction.
2. **Step 2**: It processes each image in the `images` folder:
    - Detects faces using the OpenCV DNN model.
    - For each detected face, it uses the age and gender models to predict the corresponding values.
    - Annotates the image with the predicted age and gender.
3. **Step 3**: The script saves the processed image in the `detected_images` folder.

