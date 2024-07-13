import cv2
import base64
import paho.mqtt.client as mqtt
from functions import get_contour_detections, non_max_suppression, draw_bboxes
import threading


# Background subtractor
sub_type = 'MOG2'  # 'MOG2'
if sub_type == "MOG2":
    backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=True)
    backSub.setShadowThreshold(0.75)
else:
    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)

# MQTT client setup
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
broker_address = "127.0.0.1"
username = "BS"
password = "tfm"
client.username_pw_set(username, password)
client.connect(broker_address, 1883, 60)
print("Connected to MQTT broker")

def on_publish(client, userdata, mid):
    print("Message published")

client.on_publish = on_publish

def publish_image(image_str):
    try:
        client.publish("ObjectDetected", image_str)
        print("Message published")
    except Exception as e:
        print(f"MQTT publish error: {e}")

# Video capture setup
cap = cv2.VideoCapture(r"C:\CB051456.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break
    
    
    # Apply background subtraction
    fgmask = backSub.apply(frame)

    initial_background = backSub.getBackgroundImage()


    _, thresh = cv2.threshold(fgmask, 0, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.medianBlur(thresh, 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    detections = get_contour_detections(motion_mask, 100)
    if detections.shape[0] > 0:
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        nms_bboxes = non_max_suppression(bboxes, scores, 0.1)
        draw_bboxes(frame, nms_bboxes)
        cv2.imshow("Processed Frame", frame)

        for box in nms_bboxes:
            x, y, w, h = map(int, box)
            cropped_frame = frame[y:y+h, x:x+w]
            
            _, buffer = cv2.imencode(".jpg", cropped_frame)
            image_str = base64.b64encode(buffer).decode()
            threading.Thread(target=publish_image, args=(image_str,)).start()

            # try:
            #     client.publish("ObjectDetected", image_str)
            #     print("Message published")
            # except Exception as e:
            #     print(f"MQTT publish error: {e}")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
