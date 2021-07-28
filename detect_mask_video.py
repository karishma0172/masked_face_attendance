# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import MySQLdb
from os import listdir
import face_recognition
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect,url_for,session, Response

from math import sqrt
from sklearn import neighbors
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import face_recognition
from face_recognition import face_locations
from face_recognition.face_recognition_cli import image_files_in_folder

app = Flask(__name__)

app.secret_key = 'secretkey'

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
print(APP_ROOT)


conn = MySQLdb.connect(host="localhost",user="root",password="",db="login_info")
@app.route('/')  
def index():
    return render_template('index.html' ,title="Admin Login")
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response



@app.route('/login',methods=['POST'])
def login():
	user = str(request.form["user"])
	paswd = str(request.form["password"])
	cursor = conn.cursor()
	result = cursor.execute("SELECT * from admin_login where binary username=%s and binary password=%s",[user,paswd])
	if(result is 1):
		return render_template("task.html")
	else:
		return render_template("index.html",title="Admin Login",msg="The username or password is incorrect")


@app.route('/student',methods=['POST'])
def file_upload():
	return render_template("upload.html")

@app.route('/signup_teacher',methods=['POST'])
def signup():
	
	return render_template("signup.html",title="SignUp",msg="successfully marked attendance") 

@app.route('/signup_student',methods=['POST'])
def signup_student():
	user = str(request.form["student_name"])
	email = str(request.form["student_email"])
	roll_id = str(request.form["roll_id"])
	email1 = str(request.form["parent_email"])
	cursor = conn.cursor()
	result = cursor.execute("SELECT * from student_login where binary username=%s",[user])
	print (result)
	if(result == 1):
		return render_template("upload.html",uname=user,msg=" already present")
	cursor.execute("INSERT INTO student_login (username,student_email,parent_email,roll_id) VALUES(%s, %s, %s, %s)",(user,email,email1,roll_id))
	conn.commit()
	return render_template("upload.html",uname=user,msg=" successfully signup")


@app.route("/upload", methods=['POST']) 
def upload():
	target = os.path.join(APP_ROOT,"classroom/")
	if not os.path.isdir(target):
		os.mkdir(target)
	classfolder = str(request.form['class_folder'])
	session['classfolder'] = classfolder
	target1 = os.path.join(target,str(request.form["class_folder"])+"/")
	session['target1']=target1
	print(target1)
	classname = str(request.form['class_folder']+"/")
	session['classname'] = classname
	if not os.path.isdir(target1):
		os.mkdir(target1)
	for file in request.files.getlist("file"):
		print(file)
		filename = file.filename
		destination = "/".join([target1,filename])
		print(destination)
		file.save(destination)
	return call_train()

def call_train():
	
	target1=str(session.get('target1'))
	print (target1) 
	target1 = target1
	print(target1)
	return train(train_dir=target1)

def train(train_dir,  n_neighbors = None, knn_algo = 'ball_tree', verbose=True):
    id_folder=str(session.get('id_folder'))
    X = []
    y = []
    z = 0
    for class_dir in listdir(train_dir):
        if not isdir(join(train_dir, class_dir)):
            continue
        for img_path in image_files_in_folder(join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            faces_bboxes = face_locations(image)
            if len(faces_bboxes) != 1:
                if verbose:
                    print("image {} not fit for training: {}".format(img_path, "didn't find a face" if len(faces_bboxes) < 1 else "found more than one face"))
                    os.remove(img_path)
                    z = z + 1
                continue
            X.append(face_recognition.face_encodings(image, known_face_locations=faces_bboxes)[0])
            y.append(class_dir)
    print(listdir(train_dir))
    train_dir_f = listdir(train_dir)
    for i in range(len(train_dir_f)):
    	if(train_dir_f[i].startswith('.')):
    		os.remove(train_dir+"/"+train_dir_f[i])

    print(listdir(train_dir))
    
    if(listdir(train_dir)==[]):
    	return render_template("upload.html",msg1="training data empty, upload again")
    elif(z >= 1):
    	return render_template("upload.html",msg1="Data trained for "+id_folder+", But one of the image not fit for trainning")
    if n_neighbors is None:
        n_neighbors = int(round(sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically as:", n_neighbors)

    # knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    # knn_clf.fit(X, y)

    # if model_save_path != "":
    #     with open(model_save_path, 'wb') as f:
    #         pickle.dump(knn_clf, f)

    return render_template("upload.html",msg1="Data trained for "+ id_folder)

    
@app.route('/changetask',methods=['POST'])
def changetask():
	return render_template("task.html")


@app.route('/logout',methods=['POST'])
def logout():
	return render_template("index.html",title="Admin Login",msg1="Logged out please login again")


@app.route('/view',methods=['POST'])
def view():

			# read csv file into a DataFrame
	# df = pd.read_csv(r'./Attendance.csv')
	# display DataFrame
	# df
	return render_template("task.html",title="Admin Login",msg1="DONE")





# @app.route('/signup.html', methods=['POST'])
# def task():
	# if request.method == 'POST':


path1 = 'classroom/'
path = 'classroom/be'
images = []
classNames =[]
myList = os.listdir(path)
# print(myList)

for cl in myList:
	curImg = cv2.imread(f'{path}/{cl}')
	images.append(curImg)
	classNames.append(os.path.splitext(cl)[0])

# print(classNames)



def findEncodings(images):
	encodeList = []
	for img in images:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		encode = face_recognition.face_encodings(img)[0]
		encodeList.append(encode)
	return encodeList

def markAttendance(name):
	with open('Attendance.csv', 'r+') as f:
		myDataList = f.readlines()	
		# print(myDataList)
		nameList = []
		print(myDataList)			  
		for line in myDataList:
			entry = line.split(',')
			nameList.append(entry[0])
		if name not in nameList :
			now = datetime.now()
			dtString = now.strftime('%Y-%m-%d')		
			tmString = now.strftime('%H:%M:%S')
			f.writelines(f'\n{name},{dtString}, {tmString}')


def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []


	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector2.model")


encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))   
print("encoding complete")

# initialize the video stream
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture(0)
# return render_template('index.html', prediction_text='Employee Salary should ', frame1= cv2.VideoCapture(0) )



def task():

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame,  img = vs.read()[-6:]
		frame = imutils.resize(img, width=400)
		# frame = cv2.resize(img,(0,0), None, 0.25, 0.25)
		imgS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		facesCurFrame = face_recognition.face_locations(imgS) 
		encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
		
		# detect faces in the frame and determine if they are wearing a
		# face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

		# loop over the detected face locations and their corresponding
		# locations
		for (encodeFace,faceLoc,box, pred) in zip(encodesCurFrame,facesCurFrame,locs, preds):
			# unpack the bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred
			matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
			faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
			# print(faceDis)
			matchIndex = np.argmin(faceDis)

			

			if matches[matchIndex] :
				
				name = classNames[matchIndex].upper()
				print(name)
				y1, x2, y2, x1 = faceLoc
				y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
				# cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
				# cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
				# cv2.putText(img,name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1 , (255,255,255),2)
				# markAttendance(name)
				label = "Mask" if mask > withoutMask else "No Mask"
				
				# markAttendance(name) 
				color = (0, 255, 0) and markAttendance(name) if label == "Mask" else (0, 0, 255)
				label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

				cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				cv2.rectangle(frame, (startX, endY-35), (endX,endY), (0,255,0), cv2.FILLED)
				cv2.putText(frame, name, (startX+6, endY - 6),
				cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

			else:
				name = 'Unknown'
				y1, x2, y2, x1 = faceLoc
				y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
				cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0),2)
				cv2.rectangle(img, (x1, y2-35), (x2,y2), (0,255,0), cv2.FILLED)
				cv2.putText(img,name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1 , (255,255,255),2)
				print(name)
				break
					



			# determine the class label and color we'll use to draw
			# the bounding box and text
			
			# label = "Mask" if mask > withoutMask else "No Mask"
			# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# # include the probability in the label
			# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display the label and bounding box rectangle on the output
			# frame
			# cv2.putText(frame, label, name, (startX, startY - 10),
			# 	cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			# cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		
		# show the output frame
		# cv2.imshow("FrameWebcam", frame)
		ret,buffer = cv2.imencode('.jpg', frame)         #compress n store img to memory buffer
		frame = buffer.tobytes()
		yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

		# return render_template('signup.html', prediction_text='Attendance marked')
		key = cv2.waitKey(1) & 0xFF
		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
	vs.stop()		# do a bit of cleanup
	cv2.destroyAllWindows()
	exit()

@app.route('/video_feed')
def video_feed():
    return Response(task(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
		# app.run(debug=True)
		app.run(host="127.0.0.1",port=5000,debug=True, threaded=True)
#venv\scripts\activate
