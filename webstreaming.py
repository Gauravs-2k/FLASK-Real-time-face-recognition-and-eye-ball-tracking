# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import imutils
import time
import cv2
import numpy as np
import os 
import dlib
from math import hypot
from datetime import datetime

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX
minW = 0.1*400
minH = 0.1*400

id = 1
names = ['','Gaurav']  #key in names, start from the second place, leave first empty

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
log = open("log.txt", "w")



@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def midpoint(p1,p2):
    return int((p1.x + p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points,facial_landmark):
    left_point = (facial_landmark.part(eye_points[0]).x,facial_landmark.part(eye_points[0]).y)
    right_point = (facial_landmark.part(eye_points[3]).x,facial_landmark.part(eye_points[3]).y)
    center_top = midpoint(facial_landmark.part(eye_points[1]),facial_landmark.part(eye_points[2]))
    center_bottom = midpoint(facial_landmark.part(eye_points[5]),facial_landmark.part(eye_points[4]))
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    vert_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    ratio = hor_line_length/vert_line_length
    return ratio

def get_gaze_ratio(frame,eye_points,facial_landmark,gray):
    left_eye_region = np.array([(facial_landmark.part(eye_points[0]).x,facial_landmark.part(eye_points[0]).y),
                                (facial_landmark.part(eye_points[1]).x,facial_landmark.part(eye_points[1]).y),
                                (facial_landmark.part(eye_points[2]).x,facial_landmark.part(eye_points[2]).y),
                                (facial_landmark.part(eye_points[3]).x,facial_landmark.part(eye_points[3]).y),
                                (facial_landmark.part(eye_points[4]).x,facial_landmark.part(eye_points[4]).y),
                                (facial_landmark.part(eye_points[5]).x,facial_landmark.part(eye_points[5]).y)], np.int32)

    height,width,_ = frame.shape
    mask = np.zeros((height,width),np.uint8)
    cv2.polylines(mask,[left_eye_region], True,255,2)
    cv2.fillPoly(mask,[left_eye_region],255)
    eye = cv2.bitwise_and(gray,gray,mask = mask)

    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])

    gray_eye = eye[min_y:max_y,min_x:max_x]
    _,threshold_eye = cv2.threshold(gray_eye,70,255,cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height,0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
        
    right_side_threshold = threshold_eye[0:height,int(width/2):width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white/right_side_white
    return gaze_ratio

def detect_motion(frameCount):
	global vs, outputFrame, lock
	unknownFaceStartTime = ""
	unknownFaceEndTime = ""
	unknownFaceCountTime = 0
	eyesOffscreenStartTime = ""
	eyesOffscreenEndTime = ""
	eyesOffscreenCountTime = 0
	total = 0
	while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
		frame = vs.read()
		frame = imutils.resize(frame, width=400,height=400)
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray,scaleFactor = 1.2,minNeighbors = 5,minSize = (int(minW), int(minH)),)
		eye_faces = detector(gray)
		for(x,y,w,h) in faces:
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
			id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
			if (confidence<120):
				id = names[id]
				confidence = "{0}%".format(round(100-confidence))
			else:
				id = "unknown"
				confidence = "{0}%".format(round(100-confidence))
				unknownFaceCountTime = unknownFaceCountTime+0.5
				now = datetime.now()
				dtString1 = now.strftime("%d/%m/%Y %H:%M:%S")
				if not unknownFaceStartTime:
					unknownFaceStartTime = dtString1
				else:
					unknownFaceEndTime = dtString1
			cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
			cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
		
		for face in eye_faces: #for eye tracking
			landmarks = predictor(gray,face)
			left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41],landmarks)
			right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47],landmarks)
			blinking_ratio = (left_eye_ratio+right_eye_ratio)/2
			gaze_ratio_left_eye = get_gaze_ratio(frame,[36,37,38,39,40,41],landmarks,gray)
			gaze_ratio_right_eye = get_gaze_ratio(frame,[42,43,44,45,46,47],landmarks,gray)
			gaze_ratio_avg = (gaze_ratio_right_eye + gaze_ratio_left_eye)/2
			if 1<gaze_ratio_avg<1.7:
				cv2.putText(frame,"ON-SCREEN",(50,100),font,2,(0,0,255),3)
			else:
				cv2.putText(frame,"OUTSIDE",(50,100),font,2,(0,0,255),3)
				eyesOffscreenCountTime = eyesOffscreenCountTime+0.5
				now = datetime.now()
				dtString2 = now.strftime("%d/%m/%Y %H:%M:%S")
				if not eyesOffscreenStartTime:
					eyesOffscreenStartTime = dtString2
				else:
					eyesOffscreenEndTime = dtString2
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# firstUnknownFaceIntervalString = "Unknown face spotted first at: "+unknownFaceStartTime+" - "+unknownFaceEndTime+"\n"
# totalUnknownFaceIntervalString = "Total amount of time unknown face(s) were spotted: "+str(unknownFaceCountTime)+" secs.\n"
# firstEyesOffscreenIntervalString = "Eyes went offscreen first at: "+unknownFaceStartTime+" - "+unknownFaceEndTime+"\n"
# totalEyesOffscreenIntervalString = "Total amount of time eyes were offscreen: "+str(unknownFaceCountTime)+" secs.\n"
# log.write(firstUnknownFaceIntervalString)
# log.write(totalUnknownFaceIntervalString)
# log.write(firstEyesOffscreenIntervalString)
# log.write(totalEyesOffscreenIntervalString)
# print("\nExiting Program and cleaning up stuff")
log.close()
vs.stop()