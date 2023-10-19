from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)

script_path = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_path, 'tiny.weights')
cfg_path = os.path.join(script_path, 'tiny.cfg')
classes_path = os.path.join(script_path, 'classes.txt')

net = cv2.dnn.readNet(weights_path, cfg_path)
classes = []
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    img = request.files['image']
    img.save(os.path.join(script_path, 'static/input.jpg'))

    image = cv2.imread(os.path.join(script_path, 'static/input.jpg'))
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 1, color, 2)

    cv2.imwrite(os.path.join(script_path, 'static/output.jpg'), image)
    return render_template('result.html', input_image='static/input.jpg', output_image='static/output.jpg')

if __name__ == '__main__':
    app.run(debug=True)