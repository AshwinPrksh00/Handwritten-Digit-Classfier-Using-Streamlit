import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

st.set_page_config(page_title= 'Handwritten Digit Classifier',
                   page_icon= "https://cdn.iconscout.com/icon/premium/png-256-thumb/digits-359823.png")

st.title('Handwritten Digit Recognizer')
st.write('This webapp predicts the number that you have drawn on the canvas from 0-9')
def br(i):
    st.markdown('<br>'*i, unsafe_allow_html=True)

# A pie chart to display the prediction

def chart(val):
    names=[0,1,2,3,4,5,6,7,8,9]
    f = (val[0]*100)
    fig = px.pie(names=names, values = f, labels={'names':'Predicted Number ', 'values': 'Prediction rate '})
    fig.update_traces(textposition='inside')
    st.plotly_chart(fig)


br(1)
# Loading the pre-trained MNIST Model
model = {}
model['session'] = tf.compat.v1.Session()
model['model']= load_model('model2.h5')

# Creating a drawing canvas

stroke_width = 20
stroke_color = "#eee"
bg_color = "#000000"
drawing_mode = "freedraw"
realtime_update = True

k1,k2 = st.columns([0.4,1])

  # Create a canvas component
with k2:
    data = st_canvas(
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        update_streamlit=realtime_update,
        height=300,
        width=300,
        drawing_mode=drawing_mode,
        display_toolbar=st.checkbox("Display toolbar", True),
        key="full_app",
       )

   
    if data.image_data is not None:
        st.image(data.image_data)
        img = Image.fromarray(data.image_data.astype("uint8"), mode="RGBA")
        img_red = np.array(img.resize((28,28),Image.ANTIALIAS).convert('L'))/255
        img_red = img_red.reshape(-1,1,28,28)

    if st.button('Predict'):
        br(2)
        val = []
        with model['session'].as_default():
            val = model['model'].predict(img_red)
        st.markdown("***Result***")
        st.markdown(f"<div style='background-color: grey; height: 70px; width: 70px; border-radius: 5px; padding-top: 5px; padding-bottom: 10px; margin-left: 150px; text-align: center;'> <label style='font-weight: bold; color: white; font-size: 35px;'>{np.argmax(val[0])}</label></div>",unsafe_allow_html=True)
        st.markdown("<br><br>***Model Prediction Chart***",unsafe_allow_html=True)
        chart(val)