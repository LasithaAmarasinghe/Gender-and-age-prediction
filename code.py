import cv2
import math
import os

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes

def process_image(image_path, faceNet, ageNet, genderNet, detected_folder):
    print(f"Processing image: {image_path}")
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image '{image_path}'")
        return
    
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    if not faceBoxes:
        print("No face detected")
        
        # Save the image even if no faces are detected
        output_path = os.path.join(detected_folder, "no_face_" + os.path.basename(image_path))
        cv2.imwrite(output_path, resultImg)
        print(f"Saved result to: {output_path}")
    else:
        face_count = 0
        for faceBox in faceBoxes:
            face_count += 1
            face = frame[max(0, faceBox[1]-padding):
                        min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):
                        min(faceBox[2]+padding, frame.shape[1]-1)]
            
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            
            print(f'Face #{face_count} - Gender: {gender}, Age: {age[1:-1]} years')
            
            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        
        # Save the processed image
        output_path = os.path.join(detected_folder, "detected_" + os.path.basename(image_path))
        cv2.imwrite(output_path, resultImg)
        print(f"Saved result to: {output_path}")
        print(f"Found {face_count} faces in the image")
    
    print("----------------------------------------")

# Path to models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Check if images folder exists
images_folder = "images"
if not os.path.exists(images_folder):
    print(f"Error: '{images_folder}' folder not found. Please create it and add images.")
    exit()

# Create detected_images folder if it doesn't exist
detected_folder = "detected_images"
if not os.path.exists(detected_folder):
    os.makedirs(detected_folder)
    print(f"Created '{detected_folder}' folder")

# Get list of images in the folder
image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    print(f"Error: No images found in '{images_folder}' folder. Please add images.")
    exit()

print(f"Found {len(image_files)} images to process")
print("----------------------------------------")

# Process each image
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    process_image(image_path, faceNet, ageNet, genderNet, detected_folder)

print("All images processed successfully!")