import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
folder = "data/J"
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        # Ensure valid bounding box dimensions
        if w > 0 and h > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            # Ensure the crop is within bounds
            x1 = max(x - offset, 0)
            y1 = max(y - offset, 0)
            x2 = min(x + w + offset, img.shape[1])
            y2 = min(y + h + offset, img.shape[0])
            
            imgCrop = img[y1:y2, x1:x2]
            if imgCrop.size == 0:  # If crop is empty, skip this iteration
                continue

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
    
    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
    
        # Save multiple images for 'J' (motion)
    if key == ord("j"):
        print("Capturing motion sequence for letter J...")
        for i in range(10):  # Capture 10 frames over 1 second
            success, img = cap.read()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                if w > 0 and h > 0:
                    x1 = max(x - offset, 0)
                    y1 = max(y - offset, 0)
                    x2 = min(x + w + offset, img.shape[1])
                    y2 = min(y + h + offset, img.shape[0])
                    imgCrop = img[y1:y2, x1:x2]
                    if imgCrop.size == 0:
                        continue

                    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    filename = f'{folder}/Image_{time.time()}_{i}.jpg'
                    cv2.imwrite(filename, imgWhite)
                    print(f"Saved motion frame: {filename}")
            cv2.waitKey(100)  # wait 100ms between each frame

cap.release()
cv2.destroyAllWindows()
