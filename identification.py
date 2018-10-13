import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import scipy.misc
import time

import cv2
import glob as gb

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from collections import Counter
length = 20
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
output_movie = cv2.VideoWriter('output2_4.mp4',fourcc,length,(800,600))
frame_number = 0

# if tf.__version__ != '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# cap = cv2.VideoCapture("tcamsrc serial=19814669 ! video/x-bayer,width=1600, height=1200 ! bayer2rgb ! videoconvert ! appsink")

# video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# get model of faster rcnn
PATH_TO_CKPT = '/home/hsnl-iot/models/research/object_detection/model_dir/2018_0924/frozen_inference_graph.pb'
# get label of data
PATH_TO_LABELS = os.path.join('data', 'coffee_label_map.pbtxt')
# get number of class of data (good/bad coffee)
NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		while True:
			time.sleep(0.1)
			# ret: True/False, image_np: frame
            # Get origin image from Camera
			ret, image_np = cap.read()
			print(ret)
			#	cv2.imshow(image_np)

			# OpenCV in object detec
			img = image_np
			img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			ret,thresh = cv2.threshold(img_gray,64,255,cv2.THRESH_BINARY)
			image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			i = 0
			for c in contours:
				x,y,w,h = cv2.boundingRect(c)
				if (w >30 or h>30) and (w<400 or h<300):

					text = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
					rect = cv2.minAreaRect(c)
					box = cv2.boxPoints(rect)
					box = np.int0(box)
					i+=1

					#if img[y:y+h,x:x+w].shape != [0,0]
					cut = img[y:y+h,x:x+w,:]# get the region of image
					image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
					detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
					detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
					detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
					num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                         # 181004
					image_np_expanded = np.expand_dims(cut, axis=0)

					(boxes,scores,classes,num) = sess.run(
						[detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
					bean = ''
					#print(detection_classes)
					#print(type(detection_classes))
                    # Counting the score after running the faster rcnn model
					c = Counter()
					c = Counter(np.squeeze(classes).astype(np.int32))
					if(c.most_common()[0][0]==2):
						bean = 'bad'
					else:bean = 'good'
					average = ''
					average = str(round((np.squeeze(scores)[0]+np.squeeze(scores)[1])/2*100))+'%'
					#print('average: '+average)
					#print('bean: '+bean)

                    # Draw the word of class and average number on the picture 
					font = cv2.FONT_HERSHEY_SIMPLEX

					cv2.putText(img[y-35:y+h,x:x+w,:],bean,(0,10),font,0.5,(255,255,255),2,cv2.LINE_AA)
					#cv2.putText(img[y-20:y+h,x:x+w,:],average,(0,13),font,0.5,(255,255,255),2,cv2.LINE_AA)
					 # Visualization of the results of a detection
					'''
					vis_util.visualize_boxes_and_labels_on_image_array(
                               			img,
                                		np.squeeze(boxes),
                                		np.squeeze(classes).astype(np.int32),
                                		np.squeeze(scores),
                                		category_index,
                                		use_normalized_coordinates=True,
                                		line_thickness=4)
					'''
					frame = img
					frame_number += 1
					#output_movie.write(frame)
					cv2.imshow('object detection', img)

					if cv2.waitKey(25) & 0xFF == ord('q'):
						cv2.destroyAllWindows()
						break

			'''
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			# 181004
			image_np_expanded = np.expand_dims(img, axis=0)

			(boxes, scores, classes, num) = sess.run(
				[detection_boxes, detection_scores, detection_classes, num_detections],
				feed_dict={image_tensor: image_np_expanded})

			# Visualization of the results of a detection
			vis_util.visualize_boxes_and_labels_on_image_array(
				img,
				np.squeeze(boxes),
				np.squeeze(classes).astype(np.int32),
				np.squeeze(scores),
				category_index,
				use_normalized_coordinates=True,
				line_thickness=4)
			cv2.imshow('object detection', img)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			'''
