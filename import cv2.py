import cv2
import torch
import numpy as np
import time

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture("2.mp4")
fps_start_time = time.time()
fps_frame_count = 0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))

while True:
    ret, frame = cap.read()
    input_tensor = frame

    with torch.no_grad():
        predictions = model([input_tensor])

    boxes = predictions.pandas().xyxy[0]

    for ind, box in boxes.iterrows():
        if box["name"] == "person":
            x1, y1, x2, y2 = box['xmin'], box['ymin'], box['xmax'], box['ymax']
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)

    fps_frame_count += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_text = f"FPS: {round(fps, 2)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_frame_count = 0
        fps_start_time = time.time()
    
    out.write(frame)
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()