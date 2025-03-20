# Gender-&-Age-Prediction 👦👧👩‍🦳🧔

This project uses OpenCV and deep learning models to detect faces in images and predict their age and gender.

## Features ✨

- **Face Detection**: Detects faces in images using a pre-trained OpenCV DNN model. 
- **Age Prediction**: Predicts the age range of the detected face using a Caffe-based deep learning model. 
- **Gender Prediction**: Predicts the gender (Male or Female) of the detected face using a Caffe-based deep learning model. 
- **Image Processing**: Processes each image in a specified folder and saves the output with detected faces, predicted age, and gender. 

## Technologies/Tools 🔧

* Python 3.x 
* OpenCV (`pip install opencv-python`) 
* numpy (`pip install numpy`) 

![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=FFFF00)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)

## Setup 🚀

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
    - Supported image formats: `.png`, `.jpg`, `.jpeg`. 🖼️

5. **Create the output folder**:

    - The processed images will be saved in the `detected_images` folder. This folder will be created automatically if it does not exist. 

## How This Works 🔍

1. **Step 1**: The script loads pre-trained deep learning models for face detection, age prediction, and gender prediction. 
2. **Step 2**: It processes each image in the `images` folder:
    - Detects faces using the OpenCV DNN model. 
    - For each detected face, it uses the age and gender models to predict the corresponding values. 
    - Annotates the image with the predicted age and gender. 
3. **Step 3**: The script saves the processed image in the `detected_images` folder. 

## Models Used 🏋️‍♂️

This project uses several pre-trained models to perform face detection, age prediction, and gender prediction.

### 1. **Face Detection Model** 
- **`opencv_face_detector.pbtxt`**: This is the **configuration file** that defines the architecture of the face detection model. It specifies the layers, input sizes, and other necessary parameters to initialize the face detection network. 
- **`opencv_face_detector_uint8.pb`**: This is the **pre-trained model file** containing the weights and parameters of the face detection network. It was trained on a large dataset and is capable of detecting faces in images. 

### 2. **Age Prediction Model** 
- **`age_deploy.prototxt`**: This is the **configuration file** that defines the architecture of the age prediction network. It includes details about the layers and the input structure. 
- **`age_net.caffemodel`**: This is the **pre-trained model file** that contains the learned weights for the age prediction network. It was trained on a dataset that classifies ages into different ranges. 

### 3. **Gender Prediction Model** 
- **`gender_deploy.prototxt`**: This is the **configuration file** for the gender prediction network. It contains the model's architecture, including layer definitions and input configurations. 
- **`gender_net.caffemodel`**: This is the **pre-trained model file** containing the learned weights for the gender prediction network. It classifies faces into two categories: Male and Female. 

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
