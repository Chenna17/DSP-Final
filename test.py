import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera and models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Constants
offset = 20
imgSize = 300
folder = "Data/D"
counter = 0
labels = ["D", "F", "H", "J", "M", "O", "U", "V", "W", "X"]
confidence_threshold = 0.5  # Adjust as needed

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping stays within image boundaries
        height, width, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(width, x + w + offset)
        y2 = min(height, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        # Skip if crop is empty
        if imgCrop.size == 0:
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / imgCrop.shape[0]
            wCal = math.ceil(k * imgCrop.shape[1])
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / imgCrop.shape[1]
            hCal = math.ceil(k * imgCrop.shape[0])
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Get prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)

        if prediction[index] >= confidence_threshold:
            letter = labels[index]
            # Draw label only if confident
            cv2.rectangle(imgOutput, (x1, y1 - 50), (x1 + 90, y1), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, letter, (x1 + 10, y1 - 20),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Always draw bounding box
        cv2.rectangle(imgOutput, (x1, y1), (x2, y2), (255, 0, 255), 4)

        # Show intermediate images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
