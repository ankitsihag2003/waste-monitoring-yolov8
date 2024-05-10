from ultralytics import YOLO
import cv2
import cvzone
import math
# create a webcam object
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

model = YOLO("trained_model.pt")
classNames = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1,y1,w,h))
            #confidence level
            conf = math.ceil((box.conf[0] * 100)) / 100
            #class name
            cls=int(box.cls[0])


            if conf>0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)


    cv2.imshow("image",img)
    cv2.waitKey(1)




