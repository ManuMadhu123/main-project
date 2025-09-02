import streamlit as st
from PIL import Image
import cv2
import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
from fer import FER
import time
from ptoe import PTES
import tensorflow
import random
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('voice', 'english') 
engine.setProperty('rate',140)

# Function to speak the poem
def speak_poem(poem):
    engine.say(poem)

    engine.runAndWait()

# Function to stop speaking
def stop_speaking():
    time.sleep(5)
    engine.stop()

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
genai.configure(api_key=API_KEY)
## function to load Gemini Pro model and get repsonses
model=genai.GenerativeModel("gemini-pro") 

def get_gemini_response(question,history=[]):
    chat = model.start_chat(history=history)
    response=chat.send_message(question,stream=True)
    return response


emotionobj= PTES('PERC_mendelly.csv')

L,emotions= emotionobj.find_labels()
try:
    # Initialize the FER detector
    detector = FER(mtcnn=True)
except:
    print('face emotion detector failed')
    st.text('loading failed. reload current page')
    st.stop()


def get_emotion_color(emotion):
    emotion_colors = {
        'angry': 'red',
        'disgust': 'green',
        'fear': 'purple',
        'happy': 'yellow',
        'sad': 'blue',
        'surprise': 'orange',
        'neutral': 'gray'
    }
    return emotion_colors.get(emotion, 'black') 

poem = """I have thrown from me the whirling dance of mind
And stand now in the spirit's silence free,
Timeless and deathless beyond creature-kind,
The centre of my own eternity.

I have escaped and the small self is dead;
I am immortal, alone, ineffable;
I have gone out from the universe I made,
And have grown nameless and immeasurable.

My mind is hushed in a wide and endless light,
My heart a solitude of delight and peace,
My sense unsnared by touch and sound and sight,
My body a point in white infinities.

I am the one Being's sole immobile Bliss:
No one I am, I who am all that is."""

# page config
st.set_page_config(
    page_title="PoemEmotion",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded",
)

if "emotion" not in st.session_state:
    st.session_state.emotion = "no emotion detected"
if "flag" not in st.session_state:
    st.session_state.flag= 0
if "poem" not in st.session_state:
    st.session_state.poem= ''

# load images
top_image = Image.open('static/banner_top.jpg')
bottom_image = Image.open('static/banner_bottom.png')
main_image = Image.open('static/main_banner.png')
# title
st.image(main_image,use_container_width='auto')
st.title('Affective Computing and Sentiment Analysis in Poetic Expressions 👨‍🎓')
st.sidebar.image(top_image,use_container_width='auto')
st.sidebar.header('Input 🛠')
## Select camera to feed the model
available_cameras = {'Camera 1': 0, 'Camera 2': 1, 'Camera 3': 2}
cam_id = st.sidebar.selectbox(
    "Select which camera signal to use", list(available_cameras.keys()))
st.sidebar.image(bottom_image,use_container_width='auto')


tab1, tab2 ,tab3= st.tabs(["Emotion Recognition in Poetry", "Facial Emotion-Driven Poetry Curation","Poetic Interpretation"])

with tab1:
    st.header("Poem")
    txt = st.text_area(label='place poem here',height=200,value=poem)

    if st.button('Find Emotion'):
        loaded_model = tensorflow.keras.models.load_model("model.h5")
        out = emotionobj.inference(loaded_model,txt)
        st.markdown(f" Detected EMOTION : <h1 style='color:yellow;'>{out.capitalize()}</h1>", unsafe_allow_html=True)

with tab2:
    text_placeholder = st.empty()
    # checkboxes
    st.info('✨ The Live Feed from Web-Camera will take some time to load up 🎦')
    col1, col2 ,col3 ,col4  = st.columns([1,6,6,1],gap='large')
    with col2:
        live_feed = st.checkbox('Start Web-Camera ✅')
        
    with col3:
        placeholder = st.empty()
    # camera section    
    col1, col2 ,col3 = st.columns([1,5,1])
    with col2:
        frame_placeholder = st.image('static/live-1.png')

    prev =''
    cnt=0
    # Initialize webcam
    cap = cv2.VideoCapture(available_cameras[cam_id])
    while cap.isOpened() and live_feed:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotions
        try:
            emotions = detector.detect_emotions(frame)
        except:
            pass

        # Draw emotions on the frame
        for face in emotions:
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            emotion = max(face['emotions'], key=face['emotions'].get)
            if prev!=emotion:
                prev =emotion
                cnt=0
            else:
                cnt+=1
            if cnt>=5 and emotion != 'neutral':
                print(emotion)
                color = get_emotion_color(emotion)
                if emotion == 'happy':
                    emotion = 'joy' 
                if emotion == 'angry':
                    emotion = 'anger' 
                
                st.session_state.emotion = emotion
                text_placeholder.markdown(f"<h1 style='color: {color};'>{st.session_state.emotion.capitalize()}</h1>", unsafe_allow_html=True)

            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame_placeholder.image( frame)

        # Exit when 'q' is pressed
        if (cv2.waitKey(1) & 0xFF ==ord("q")) or not(live_feed):
            break

    # Release the capture and close all windows
    time.sleep(1)
    cap.release()
    cv2.destroyAllWindows()


    
    st.markdown(f" Detected EMOTION : <h1 style='color:yellow;'>{st.session_state.emotion.capitalize()}</h1>", unsafe_allow_html=True)
    if st.session_state.emotion != "no emotion detected":
        if st.button('display Poems'):
            st.write("Here are some poems for you:")
            print('******************************')
            emotion = st.session_state.emotion
            print(emotion)
            poems = emotionobj.data['poem'][emotionobj.data['class']==emotion].to_list()
            poemtxt = poems[random.randint(0,len(poems)-1)]
            st.write(poemtxt)
            st.session_state.flag =1
            st.session_state.poem = poemtxt
            
    if st.session_state.flag:
        # Buttons to control speaking
        col1, col2 = st.columns(2,)

        with col1:
            if st.button("Display & Speak Poem"):
                st.write("Here is a poem for you:")
                speak_poem(st.session_state.poem)  # Speak the poem

        with col2:
            if st.button("Stop Speaking"):
                stop_speaking()  # Stop speaking immediately

        st.write("Press the buttons to listen or stop the poem.")  
            

with tab3:
    st.header("Poem")
    txt1 = st.text_area(label='place poem here',height=200,value=poem,key='meaning')

    if st.button('Find Meaning'):
        response=get_gemini_response(txt1)

        for chunk in response:
                st.write(chunk.text)
                





st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.markdown("<br><hr><center>Made by  </center><hr>", unsafe_allow_html=True)