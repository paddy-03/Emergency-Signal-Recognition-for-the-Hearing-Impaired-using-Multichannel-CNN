
import os,subprocess
import glob
import time

# External modules
import numpy as np
import pandas as pd
#from tqdm import tqdm

# Librosa for audio
import librosa
import librosa.display

# Plotting modules
import matplotlib.pyplot as plt
import matplotlib.style as ms
import matplotlib.ticker as ticker
import seaborn as sns
sns.set_style("whitegrid")
import matplotlib.gridspec as gridspec

# Scikit-learn modules
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix

# Keras and TensorFlow modules
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
K.set_image_dim_ordering('th')

#import werkzeug
from flask import Flask,render_template, url_for, request, redirect, send_from_directory
from flask_restful import reqparse, abort, Api, Resource
from flask import jsonify 
from werkzeug.utils import secure_filename

K.clear_session()

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp4','mp3'])

app=Flask(__name__)
#api=Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")

def hello():
	return render_template('index.html')

#parser=reqparse.RequestParser()
#parser.add_argument('audio', type=werkzeug.FileStorage, location='files')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def mfcc(y, sr=8000, n_mfcc=12):
    '''
    Finds the MFCC coefficients given sampled audio data
    Arguments:
    y - sampled audio signal
    sr - sampling rate (Hz)
    n_mfcc - Number of MFCC coefficients
    Returns:
    list of MFCC coefficients
    '''
    return librosa.feature.mfcc(y=y,sr=sr, n_mfcc=n_mfcc)

def extract_mfccs(y):
    '''
    Extract MFCC coefficients from short duration audio clips
    Arguments:
    y - sampled audio signal
    Returns:
    list of MFCC coefficients for each sub-sample
    '''
    mfccs_list = []
    ran = len(y)//160
    for i in range(ran-10):
        y_clip = y[160*i:160*(i+1)]
        mfccs_clip = mfcc(y_clip)
        mfccs_clip = np.array(mfccs_clip)
        mfccs_clip = mfccs_clip.flatten()
        mfccs_list.append(mfccs_clip)
    return mfccs_list

def predict_probability(y, scaler):
    mfccs_list = extract_mfccs(y)
    scaler.transform(mfccs_list)
    count = 0
    N = 20 # Window size
    th = 0.5 # Minimum probabilty value for Em presence

    model = load_model('model3.h5')

    prob_list = []
    class_list = []

    for i in range(N):
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)


    if prob > th:
        #print("Em")
        class_list.append(1)
    else:
        #print("Non-em")
        class_list.append(0)

    for i in range(N,len(mfccs_list)):
        prob_list.pop(0)
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        #print(prob)
        if prob > th:
            #print("Em")
            class_list.append(1)
        else:
            #print("Non-em")
            class_list.append(0)

    return class_list


# Test Accuracy
def predict_output(y, scaler):

    mfccs_list = extract_mfccs(y)
    scaler.transform(mfccs_list)
    count = 0
    N = 20
    th = 0.5

    model = load_model('model3.h5')
    model._make_predict_function()

    prob_list = []
    class_list = []
    for i in range(N):
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
    prob = np.mean(prob_list)
    #print(prob)
    if prob > th:
        #print("Em")
        class_list.append(1)
    else:
        #print("Non-em")
        class_list.append(0)

    for i in range(N,len(mfccs_list)):
        prob_list.pop(0)
        p = model.predict(mfccs_list[i].reshape(1,12), batch_size=None, verbose=0)
        p = p.flatten()
        prob_list.append(p)
        prob = np.mean(prob_list)
        #print(prob)
        if prob > th:
            #print("Em")
            class_list.append(1)
        else:
            #print("Non-em")
            class_list.append(0)
    if np.mean(class_list) > 0.5:
        return 1
    else:
        return 0


@app.route("/fun",methods=['GET','POST'])

def fun():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            #return request.url/shivam1
            return "shivam1"

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            #flash('No selected file')
            #return request.url/shivam
            return "shivam2"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))


            #int_message = 1
            #print("Data uploading")
            #print(request.headers)
            #for v in request.values:
            #    print(v)
            #logdata = request.stream.readline()
            #while(logdata):
            #    print "uploading"
            #    print logdata
            #    logdata = request.stream.readline()
            #print("Uploading done")
            #return Response(str(int_message), mimetype='text/plain')
            return redirect(url_for('uploaded_file', filename=filename))
            #return "shivam3"


@app.route('/predict/<filename>',methods=['GET','POST'])

def uploaded_file(filename):

    file = filename
    name = file[:file.rfind(".")]
    #file = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    dir = "/home/cardano/Music/prediction/uploads"
    subprocess.run(["ffmpeg", "-i", dir+"/"+name+".mp3",  dir+"/"+name+".wav"])
   # file = os.path.join(app.config['UPLOAD_FOLDER'], name+".wav")

    y, sr = librosa.load(dir+"/"+name+".wav", sr=8000)

    # Load the scaler obtained from the train data
    scaler_filename = "scaler3.save"
    scaler = joblib.load(scaler_filename)

   # classes = predict_probability(y, scaler)
    output = str(predict_output(y,scaler))
    print(output)
    K.clear_session()
    return (output)


if __name__=='__main__':
    app.run(host="192.168.0.100", port=5020, debug=True)




