# import the necessary packages
import numpy as np
import pickle
from numpy.lib.function_base import rot90
import torch
import cv2

# package for TTA
import test_time_augmentation_utils as tta_utils
#import odach as oda

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open("coco_classes.pickle", "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

def inference(model,
			  image,
			  batch_i,
			  DEVICE,
			  augment = False,
			  arg_confidence=0.5,
			  tta = None,
			  visualize=True):

	# convert the image from BGR to RGB channel ordering and change the
	# image from channels last to channels first ordering
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
	orig = np.array(image.copy())
	width = image.shape[1]
	height = image.shape[0]
	image = image.transpose((2, 0, 1))

	# add the batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the image to a floating point tensor
	image = np.expand_dims(image, axis=0)
	image = image / 255.0
	image = torch.FloatTensor(image)

	# send the input to the device and pass the it through the network to
	# get the detections and predictions
	image = image.to(DEVICE)

	if not augment:
		# No TTA
		detections = model(image)[0]
	else:
		
		if tta == 0 :
			tta = []
		elif tta == 1:
			tta = [tta_utils.Multiply(0.9)]
		elif tta == 2:
			tta = [tta_utils.HorizontalFlip()]
		elif tta == 3:
			tta = [tta_utils.HorizontalFlip(), tta_utils.Multiply(0.9), tta_utils.Multiply(1.1)]
		elif tta == 4:
			tta = [tta_utils.HorizontalFlip(), tta_utils.VerticalFlip(), tta_utils.Rotate90Left(), tta_utils.Multiply(0.9), tta_utils.Multiply(1.1)]

		# With TTA
		tta_model = tta_utils.TTAWrapper(model, tta)
		boxes, scores, labels = tta_model(image)

		for i in range(0, len(boxes)):
			new_boxes = boxes[i].copy()
			new_boxes[0] *= width
			new_boxes[2] *= width
			new_boxes[1] *= height
			new_boxes[3] *= height
			boxes[i] = new_boxes

		detections = {"boxes":boxes,"scores":scores,"labels":labels}

	if visualize :
		# loop over the detections
		for i in range(0, len(detections["boxes"])):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections["scores"][i]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > arg_confidence:
				# extract the index of the class label from the detections,
				# then compute the (x, y)-coordinates of the bounding box
				# for the object
				idx = int(detections["labels"][i])
				if not augment:
					box = detections["boxes"][i].detach().cpu().numpy()
				else :
					box = detections["boxes"][i]
					
				(startX, startY, endX, endY) = box.astype("int")
				# display the prediction to our terminal
				label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
				print("[INFO] {}".format(label))

				# draw the bounding box and label on the image
				cv2.rectangle(orig, (startX, startY), (endX, endY),
					COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(orig, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

			# show the output image
			cv2.imwrite("img_{}.jpg".format(batch_i),orig)  # save
		exit()
	
	return detections

def xywh2xyxy(x):
	# Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
	y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
	y[0] = x[0]  # top left x
	y[1] = x[1]  # top left y
	y[2] = x[0] + x[2]  # bottom right x
	y[3] = x[1] + x[3]  # bottom right y
	return y