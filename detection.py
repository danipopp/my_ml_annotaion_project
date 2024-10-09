import cv2
import numpy as np

# YOLOv3 verwenden (Modell und Konfigurationsdateien müssen vorhanden sein)
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Klassen (Objekttypen)
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Bild laden und verarbeiten
img = cv2.imread("pictures/farmer.jpg")
height, width, channels = img.shape

# Blob erstellen und durch das Netz laufen lassen
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Liste für erkannte Objekte erstellen
boxes = []
confidences = []
class_ids = []

# Durchlaufe alle Vorhersagen
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Prüfe, ob die Confidence größer als 50 % ist
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Speichere die Bounding Box, Confidence und Klassen-ID
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Nicht-Maximum-Suppression anwenden, um überlappende Boxen zu eliminieren
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

# Zeichne die erkannten Objekte
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

        # Zeichne das Rechteck (Bounding Box)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Kleinere Schriftgröße und Dicke für den Text
        font_scale = 5  # Kleinere Schriftgröße
        font_thickness = 4  # Dünnere Schrift

        # Berechne die Textgröße
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)

        # Erstelle ein gefülltes Rechteck oberhalb des Objekts für den Text
        cv2.rectangle(img, (x, y), (x + text_width, y + text_height + baseline), (0, 255, 0), -1)
        
        # Schreibe den Text ins Rechteck
        cv2.putText(img, label, (x, y + text_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

# Zeige das Bild an
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
