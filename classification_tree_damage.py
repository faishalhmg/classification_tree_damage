# Imports
from PIL import Image
import streamlit as st
import tensorflow as tf
import time
import os
class_names = ["kanker","konk","luka_terbuka","resinosis_gumosis","batang_pecah","sarang_rayap","batang_atau_akar_patah","brum_akar_atau_batang","akar_patah_atau_mati","liana","hilang_mati_pucuk_dominan","cabang_patah_mati","percabangan_brum_berlebihan","daun_pucuk_tunas_rusak","daun_berubah_warna","gerowong"]

## Page Title
st.set_page_config(page_title = "Cats vs Dogs Image Classification")
st.title(" Cat vs Dogs Image Classification")
st.markdown("---")

## Sidebar
st.sidebar.header("TF Lite Models")
display = ("Select a Model","Converted FP-16 Quantized Model")
options = list(range(len(display)))
value = st.sidebar.selectbox("Model", options, format_func=lambda x: display[x])
print(value)

if value == 1:
        tflite_interpreter = tf.lite.Interpreter(model_path='convertmodel7504.tflite')

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
  
def get_predictions(input_image):
    output_details = tflite_interpreter.get_output_details()
    set_input_tensor(tflite_interpreter, input_image)
    tflite_interpreter.invoke()
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    pred_class = class_names[tflite_model_prediction]
    return pred_class
    tflite_model_prediction = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_model_prediction = tflite_model_prediction.squeeze().argmax(axis = 0)
    pred_class = class_names[tflite_model_prediction]
    print(tflite_model_prediction)
    print(pred_class)
    return pred_class
  
uploaded_file = st.file_uploader("Upload a Image", type=["jpg","png", 'jpeg'])
if uploaded_file is not None:
    with open(os.path.join("tempDir",uploaded_file.name),"wb") as f:
         f.write(uploaded_file.getbuffer())
    path = os.path.join("tempDir",uploaded_file.name)
    img = tf.keras.preprocessing.image.load_img(path , grayscale=False, color_mode='rgb', target_size=(224,224,3), interpolation='nearest')
    st.image(img)
    print(value)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    
if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)

if st.button("Get Predictions"):
    suggestion = get_predictions(input_image =img_array)
    st.success(suggestion)