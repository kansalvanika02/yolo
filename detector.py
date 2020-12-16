# USAGE
# python track_object.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#	--model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --video input/race.mp4 \
#	--label person --output output/race_output.avi

# import the necessary packages
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import cv2



# video = 0 # for camera
video ='input1/people.mp4'
output = 'output.mp4'
confidence = 0.9
find = 'person'

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
# colors  = np.random.uniform(0, 255, size = (len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
prototxt = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, dlib correlation tracker, output video
# writer, and predicted class label
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(video)
writer = None
label = ""

# start the frames per second throughput estimator
fps = FPS().start()

# loop over frames from the video file stream
while True:
	# grab the next frame from the video file
	(grabbed, frame) = vs.read()

	# check to see if we have reached the end of the video file
	if frame is None:
		break
	else:
		print('Processing Image')
	
	# resize the frame for faster processing and then convert the
	frame = imutils.resize(frame, width=1200)
	h, w, c = frame.shape #height width channel

	# if we are supposed to be writing a video to disk, initialize
	# the writer
	if output is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output, fourcc, 100, (w, h), True)

	# grab the frame dimensions and convert the frame to a blob
	# pass the blob through the network and obtain the detections
	# and predictions
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
	net.setInput(blob)
	detections = net.forward()

	# ```
	# conf = inference_results[0, 0, i, 2]   # extract the confidence (i.e., probability) 
	# idx = int(inference_results[0, 0, i, 1])   # extract the index of the class label
	# boxPoints = inference_results[0, 0, i, 3:7]
	# ```
	
	# ensure at least one detection is made
	# if len(detections)<1:
	# 	continue
	ctr=0
	for i in range(len(detections[0,0,:,0])):
		# grab the probability associated with the object along
		# with its class label
		try:
			conf = detections[0, 0, i, 2]
			if detections[0, 0, i, 1]<0:
				detections[0, 0, i, 1]=0

			label = CLASSES[int(detections[0, 0, i, 1])]
			label_text = CLASSES[int(detections[0, 0, i, 1])]+' '+str(conf)
		except Exception as e:
			print(e)
			continue
		# filter out weak detections by requiring a minimum
		# confidence
		if conf > confidence :
			# compute the (x, y)-coordinates of the bounding box
			# for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")


			# draw the bounding box and text for the object
			if label==find:
				ctr+=1
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
			else:
				cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 255), 2)

			cv2.putText(frame, label_text, (startX, startY - 15),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1) # BGR

	cv2.putText(frame, f"Person Tracked: {ctr}", (100,80),
		cv2.FONT_HERSHEY_SIMPLEX, 1.45, (55, 40, 255), 3) # BGR



	# check to see if we should write the frame to disk
	if writer is not None:
		writer.write(frame)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# check to see if we need to release the video writer pointer
if writer is not None:
	writer.release()

# do a bit of cleanup
cv2.destroyAllWindows()
vs.release()
