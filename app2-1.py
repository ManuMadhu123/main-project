import streamlit_authenticator as stauth
import streamlit as st
from PIL import Image
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from yaml.loader import SafeLoader
from fer import FER
import time
from ptoe import PTES
import tensorflow
import random
from dotenv import load_dotenv
import os
import google.generativeai as genai
import pyttsx3
import comtypes
comtypes.CoInitialize()
from nltk.sentiment.vader import SentimentIntensityAnalyzer


with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if "login_state" not in st.session_state:
    st.session_state["login_state"] = True
def fn():
    st.session_state["login_state"] =  not st.session_state["login_state"]


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
model=genai.GenerativeModel("gemini-1.5-pro-002") 

def get_gemini_response(question,history=[]):
    chat = model.start_chat(history=history)
    response=chat.send_message(question,stream=True)
    return response
def analyze_poem_sentiment(poem):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(poem)
    
    compound_score = sentiment['compound']  # Overall sentiment score
    if compound_score >= 0.5:
        intensity = "Highly Positive"
    elif 0.1 <= compound_score < 0.5:
        intensity = "Moderately Positive"
    elif -0.1 < compound_score < 0.1:
        intensity = "Neutral"
    elif -0.5 <= compound_score <= -0.1:
        intensity = "Moderately Negative"
    else:
        intensity = "Highly Negative"
    
    return {"Sentiment Score": sentiment, "Sentimental Intensity": intensity}

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
# st.set_page_config(
#     page_title="PoemEmotion",
#     page_icon="✨",
#     layout="centered",
#     initial_sidebar_state="expanded",
# )

if "emotion" not in st.session_state:
    st.session_state.emotion = "no emotion detected"
if "flag" not in st.session_state:
    st.session_state.flag= 0
if "poem" not in st.session_state:
    st.session_state.poem= ''

# load images
top_image = Image.open('static/banner_top.jpg')
bottom_image = Image.open('static/banner_bottom.jpg')
main_image = Image.open('static/main_banner.png')

@st.fragment
def displaypoem(selected_poems):
    poemtxt = st.selectbox("Choose a poem:", selected_poems)
    st.write(poemtxt)
    st.session_state.flag =1
    st.session_state.poem = poemtxt

if st.session_state["login_state"] :
    with st.container(key='login'):
        
        authenticator.login(captcha=False)
    if st.session_state['authentication_status'] is None:
        with st.container(key='Signuplink'):    
            # Signup link with Streamlit text and button
            col1,col2,col3 = st.columns([.7,.2,.5],vertical_alignment='center')
            with col1:
                st.markdown('<span class="custom-text">Don\'t have an account?</span>', unsafe_allow_html=True)
            with col2:
                st.button('Sign Up',key='btn',on_click=fn)
    if st.session_state['authentication_status']:
        
        
        
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
        with st.sidebar:
            authenticator.logout()


        tab1, tab2 ,tab3= st.tabs(["Emotion Recognition in Poetry", "Facial Emotion-Driven Poetry Curation","Poetic Interpretation"])

        with tab1:
            st.header("Poem")
            txt = st.text_area(label='place poem here',height=200,value=poem)

            if st.button('Find Emotion'):
                loaded_model = tensorflow.keras.models.load_model("model.h5")
                out = emotionobj.inference(loaded_model,txt)
                st.markdown(f" Detected EMOTION : <h1 style='color:yellow;'>{out.capitalize()}</h1>", unsafe_allow_html=True)
                result = analyze_poem_sentiment(txt)
                st.subheader("Sentiment Scores")
                st.json(result['Sentiment Score'])  # Displaying scores in JSON format
                # Display Sentimental Intensity with a color-coded message
                sentiment_color = {
                    "Highly Positive": "green",
                    "Moderately Positive": "lightgreen",
                    "Neutral": "gray",
                    "Moderately Negative": "lightcoral",
                    "Highly Negative": "red"
                }

                st.markdown(
                    f"<h3 style='color: {sentiment_color[result['Sentimental Intensity']]}'>"
                    f"Sentimental Intensity: {result['Sentimental Intensity']}</h3>",
                    unsafe_allow_html=True
                )



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
                st.session_state.flag =0
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
                    num_poems = min(10, len(poems))  # In case there are fewer than 10 poems
                    selected_poems = random.sample(poems, num_poems)

                    # Dropdown or radio button to choose a poem
                    # poemtxt = st.selectbox("Choose a poem:", selected_poems)
                    displaypoem(selected_poems)
                    # poemtxt = poems[random.randint(0,len(poems)-1)]
                    # st.write(poemtxt)
                    # st.session_state.flag =1
                    # st.session_state.poem = poemtxt
                    
            if st.session_state.flag:
                # Buttons to control speaking
                col1, col2 = st.columns(2,)

                with col1:
                    if st.button("Speak Poem"):
                        st.write("Here is a poem for you:")
                        speak_poem(st.session_state.poem)  # Speak the poem

                # with col2:
                #     if st.button("Stop Speaking"):
                #         stop_speaking()  # Stop speaking immediately

                # st.write("Press the buttons to listen or stop the poem.")  
                    

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
        st.markdown("<br><hr><center>Made by Group 5  </center><hr>", unsafe_allow_html=True)
else:
    with st.container(key='login'):
        try:
            email_of_registered_user, username_of_registered_user, name_of_registered_user = authenticator.register_user(captcha=False,fields= {'Form name':'Register user', 'Username':'Username', 'Password':'Password', 'Register':'Register'})
            if email_of_registered_user:
                st.success('User registered successfully')
                #  Updating the configuration file
                with open('config.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)
        except Exception as e:
            st.error(e)
    with st.container(key='Signuplink'):    
    # Signup link with Streamlit text and button
        col1,col2,col3 = st.columns([.7,.2,.5],vertical_alignment='center')
        with col1:
            st.markdown('<span class="custom-text"> Log in here.</span>', unsafe_allow_html=True)
        with col2:
            st.button('Log in',key='btn',on_click=fn)


