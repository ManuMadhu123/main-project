import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam,SGD
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
import warnings
warnings.simplefilter('ignore')
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report,confusion_matrix
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.utils import set_random_seed
set_random_seed(100)

# create class for poem emotiom classification
class PTES():
  def __init__(self,filename):
    self.filename = filename # dataset filename
    self.classes = ['sad' 'love' 'peace' 'joy' 'courage' 'surprise' 'hate' 'anger' 'fear'] # emotion classes
    df = pd.read_csv(self.filename,encoding='unicode_escape') # read dataset
    indexAge = df[ (df['class'] == 'peace') | (df['class'] == 'love' ) | (df['class'] == 'courage')| (df['class'] == 'hate')].index
    df.drop(indexAge , inplace=True)
    self.data = df
    self.fsize=801 # size of features
    self.bow = CountVectorizer(max_features=380) # initilaize bag of words class
    self.vectorizer = TfidfVectorizer(max_features=380) # initialize tfidf class
    # model method
  def model(self):
    model = keras.Sequential() # create empty sequential model
    model.add(layers.Conv1D(128,1, activation='relu',input_shape=(1,self.fsize)))
    # model.add(layers.Bidirectional(layers.LSTM(5,return_sequences=True)))
    model.add(layers.GRU(50,return_sequences=True))
    model.add(layers.SimpleRNN(20))
    model.add(layers.Dense(64))
    model.add(layers.ELU())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(5, activation='softmax')) # adds a dense output layer with 9 units and softmax activation
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) # compiles the model with sparse categorical crossentropy loss, Adam optimizer with learning rate 0.001, and accuracy metric

    return model
  def remove_stopwords(self,text):
    # The function uses a list comprehension to tokenize the input text using the nltk library and checks if each token is not in the list of stopwords. The resulting list is then joined into a string with spaces between each word and returned by the function.
    stop_words = stopwords.words('english') # creates a list of stopwords in English
    return ' '.join([word for word in nltk.word_tokenize(text) if word not in stop_words])

  # convert to lower case
  def lower(self,text):
    return text.lower()
  
  def lemmatize_words(self,text):
      # group together  variant forms of the same word
      lemmatizer = WordNetLemmatizer()
      return " ".join(lemmatizer.lemmatize(word) for word in text.split())


  def data_preprocessing(self,text):
    text = self.remove_stopwords(text)
    text =self.lower(text)
    text = self.lemmatize_words(text)
    return text

  def train_model(self,model):
    x_train,y_train=self.split_data()
    self.history = model.fit(x_train,y_train,epochs=250, batch_size=16)
    return model

  def find_labels(self):
    lookup = dict(enumerate(self.classes))
    self.enc = LabelEncoder()
    self.enc.fit(self.data['class'])
    label = self.enc.transform(self.data['class'])
    self.target = np.array(label)
    return label,lookup

  # semantic features
  def sentiment(self,text):
    sid = SentimentIntensityAnalyzer()
    sentiment_dict=sid.polarity_scores(text)
    if sentiment_dict['compound'] >= 0.05 : # decide sentiment as positive, negative and neutral
      # print("Positive")
      sm=1
    elif sentiment_dict['compound'] <= - 0.05 :
      sm=2
      # print("Negative")
    else :
      # print("Neutral")
      sm=3
    return sm

  

  def split_data(self):
    # split data
    x_train, x_test, y_train, self.y_test = train_test_split(self.fdata, self.target, test_size=0.2, random_state=36)
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    self.x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    return x_train,y_train

  def inference(self,model,text):
    # test single poem
      text = self.data_preprocessing(text)
      fet = self.features(text)
      fet = np.expand_dims(fet,axis=0)
      out = model.predict(fet)
      out = np.argmax(out[0])
      return self.enc.inverse_transform([out])[0]

  def training_analysis(self):
    # plot training graphs
    plt.plot(self.history.history['loss'], label='train loss')
    plt.legend()
    plt.show()
    plt.savefig('LossVal_loss')

    plt.plot(self.history.history['accuracy'], label='train acc')
    plt.legend()
    plt.show()
    plt.savefig('AccVal_acc')

  def features(self,txt):
    #  extract features from single poem
    fet1 = self.bow.fit_transform([txt]).toarray()
    fet2 = self.vectorizer.fit_transform([txt]).toarray()
    N = 400-fet1.shape[1]
    fet1 = np.hstack([fet1,np.zeros((1,N))])
    N = 400-fet2.shape[1]
    fet2= np.hstack([fet2,np.zeros((1,N))])
    fet3 = np.array([self.sentiment(txt)])
    fet3=np.expand_dims(fet3,axis=1)
    fet = np.concatenate([fet1,fet2,fet3],axis=1)
    # print(fet.shape)
    fet = fet/fet.sum()
    return fet
    
  def extract_feature(self):
    # Bag-Of-Words 
    fet= self.data['poem'].apply(self.features)
    f= np.array(fet.tolist())
    ff = np.squeeze(f,axis=1)
    print(ff.shape)
    self.fdata = ff
 

  def data_processing(self):
    # process whole dataset
    self.data['poem'] = self.data['poem'].apply(self.data_preprocessing)
    


 
 

  def display_data(self,no=8):
    # display data
    print(self.data.head(no))


