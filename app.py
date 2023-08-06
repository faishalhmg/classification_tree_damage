# app.py

from PIL import Image
import tensorflow as tf
import os
from flask import Flask,Response,request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

class_names = [
    "akar_patah-mati", "batang-akar_patah", "batang_pecah", "brum_akar_atau_batang",
    "cabang_patah_mati", "daun_berubah_warna", "daun_pucuk_tunas_rusak", "gerowong",
    "hilang_pucuk_dominan", "kanker", "konk", "liana", "luka_terbuka", "percabangan_brum_berlebihan",
    "resinosis_gumosis", "sarang_rayap"
]

label_map = {
    1: "akarPatahmati",
    2: "batangakarpatah",
    3: "batangpecah",
    4: "brumakarbatang",
    5: "cabangpatahmati",
    6: "daunberubahwarna",
    7: "daunpucuktunasrusak",
    8: "gerowong",
    9: "hilangpucukdominan",
    10: "kanker",
    11: "konk",
    12: "liana",
    13: "lukaterbuka",
    14: "percabanganbrumberlebihan",
    15: "resinosisgumosis",
    16: "sarangrayap"
}
class_colors = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Cyan
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (128, 0, 0),    # Dark Red
    (0, 128, 0),    # Dark Green
    (0, 0, 128),    # Dark Blue
    (128, 128, 0),  # Dark Cyan
    (0, 128, 128),  # Dark Yellow
    (128, 0, 128),  # Dark Magenta
    (255, 128, 0),  # Orange
    (128, 255, 0),  # Lime
    (0, 128, 255),  # Sky Blue
    (255, 0, 128)   # Pink
]

# Loading the saved_model
PATH_TO_SAVED_MODEL = "export/saved_model"
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


def get_predictions(input_image):
    input_shape = (224, 224)  # Replace with the input shape of your Keras model
    img_array = tf.image.resize(input_image, input_shape)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Load your Keras model here (replace 'your_model.h5' with the path to your model file)
    model = tf.keras.models.load_model('keras_model.h5')

    tflite_model_prediction = model.predict(img_array)
    tflite_model_prediction = tf.argmax(tflite_model_prediction, axis=1).numpy()[0]
    pred_class = class_names[tflite_model_prediction]
    return pred_class

def detect_objects(frame):
    image_np = np.array(frame)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy().tolist() for key, value in detections.items()}
    detections['num_detections'] = num_detections

    image_np_with_detections = image_np.copy()

    # Draw bounding boxes on image
    for box, score, class_id in zip(detections['detection_boxes'], detections['detection_scores'], detections['detection_classes']):
        if score > 0.5:
            height, width, _ = image_np_with_detections.shape
            ymin, xmin, ymax, xmax = box
            left = int(xmin * width)
            top = int(ymin * height)
            right = int(xmax * width)
            bottom = int(ymax * height)
            offset = 10  # Jumlah offset yang ingin Anda gunakan
            class_name = label_map[int(class_id)]

            # Get class color based on class_id
            class_color = class_colors[int(class_id) % len(class_colors)]

            # Draw bounding box
            cv2.rectangle(image_np_with_detections, (left, top + offset), (right, bottom), class_color, 2)

            # Add background box for label
            cv2.rectangle(image_np_with_detections, (left, top + offset - 20), (right, top + offset), class_color, -1)

            # Add class label text
            cv2.putText(image_np_with_detections, f"{class_name} ({round(score * 100, 2)}%)", (left, top + offset - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image_np_with_detections

def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        output_frame = detect_objects(frame)

        ret, buffer = cv2.imencode('.jpg', output_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify(error="No selected file"), 400

    image = Image.open(file)
    image = image.convert("RGB")
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    suggestion = get_predictions(input_image=img_array)
    return jsonify(prediction=suggestion)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    file = request.files['image']
    img_np = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    output_img = detect_objects(img)
    _, buffer = cv2.imencode('.jpg', output_img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(port=80)