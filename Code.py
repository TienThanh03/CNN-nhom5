import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import filedialog

# Constants
START_POINT = 80
END_POINT = 150
CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7, 8, 9]  # Updated to include truck and motorbike
YOLOV3_CFG = 'yolov3-tiny.cfg.txt'
YOLOV3_WEIGHT = 'yolov3-tiny.weights'
CONFIDENCE_SETTING = 0.45
YOLOV3_WIDTH = 416
YOLOV3_HEIGHT = 416
MAX_DISTANCE = 80

def get_output_layers(net):
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers                 
    except:
        print("Can't get output layers")
        return None

def detections_yolo3(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(get_output_layers(net))
    boxes = []
    class_ids = []                                              
    confidences = []
    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
                print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([int(x), int(y), int(w), int(h)])
    return boxes, class_ids, confidences

def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height):
    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)
        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))

def check_location(box_y, box_height, height):
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT:
        return True
    else:
        return False

def check_start_line(box_y, box_height):
    center_y = int(box_y + box_height / 2.0)
    if center_y > START_POINT:
        return True
    else:
        return False

def counting_vehicle(video_input, video_output, skip_frame=1):
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG, YOLOV3_WEIGHT)

    cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_output, video_format, 25, (width, height))

    list_object = []
    number_frame = 0
    number_vehicle = 0

    while cap.isOpened():
        number_frame += 1
        ret_val, frame = cap.read()
        if frame is None:
            break

        tmp_list_object = list_object
        list_object = []

        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    number_vehicle += 1
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            boxes, class_ids, confidences = detections_yolo3(net, frame, CONFIDENCE_SETTING, YOLOV3_WIDTH,
                                                             YOLOV3_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    check_new_object = True
                    for tracker in list_object:
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        if distance < MAX_DISTANCE:
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        new_tracker = cv2.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }
                        list_object.append(new_object)
                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)

        cv2.putText(frame, "Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 1)
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (255, 0, 0), 2)
        cv2.imshow("Counting", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    root = tk.Tk()
    root.title("Vehicle Counting Application")

    label_video = tk.Label(root, text="Video File:")
    label_video.grid(row=0, column=0, padx=10, pady=10)

    entry_video = tk.Entry(root, width=40)
    entry_video.grid(row=0, column=1, padx=10, pady=10)

    def browse_video():
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
        entry_video.delete(0, tk.END)
        entry_video.insert(0, file_path)

    button_browse = tk.Button(root, text="Browse", command=browse_video)
    button_browse.grid(row=0, column=2, padx=10, pady=10)

    def start_counting():
        video_input = entry_video.get()
        if not video_input:
            print("Please select a video file.")
            return
        counting_vehicle(video_input, 'vehicles.avi')

    button_start = tk.Button(root, text="Start Counting", command=start_counting)
    button_start.grid(row=1, column=0, columnspan=3, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
