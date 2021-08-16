# USAGE
# python detect_image.py 

import warnings
from numpy.lib.polynomial import roots
warnings.filterwarnings('ignore',category=FutureWarning)

# import the necessary packages
from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import os
from tqdm import tqdm
from dataloader_utils import CocoDetection
from inference_utils import inference, xywh2xyxy
from mean_avg_precision_utils import mean_average_precision as map

def main():
    
	if not os.path.exists("../coco/images/val2017/") and not os.path.exists("../coco/annotations/instances_val2017.json"):
		print("Dataset not present. You need to run ./cocodataset.sh script before going further")
		exit() 
	else :
		# set the device we will be using to run the model
		DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		root = "../coco/images/val2017/"
		annotations = "../coco/annotations/instances_val2017.json"

		# initialize a dictionary containing model name and it's corresponding 
		# torchvision function call
		MODELS = {"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,}

		# load the list of categories in the COCO dataset and then generate a
		# set of bounding box colors for each class
		CLASSES = pickle.loads(open("coco_classes.pickle", "rb").read())
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

		# load the model and set it to evaluation mode
		model = MODELS["frcnn-resnet"](pretrained=True, progress=True,num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
		model.eval()

		dataloader = CocoDetection(root=root, annFile=annotations)
		if args["iou"] is not None :
			map_scores = []
		else :
			map_scores_05 = []
			map_scores_07 = []
			map_scores_09 = []
		
		for batch_i, (img, targets) in enumerate(tqdm(dataloader)):
			detections = inference(model,img,batch_i,DEVICE,args["augment"],args["confidence"],args["option"],visualize=False)

			if not args["augment"]:
				boxes = detections["boxes"].cpu().detach().numpy()
				scores = detections["scores"].cpu().detach().numpy()
				labels = detections["labels"].cpu().detach().numpy()
			else :
				boxes = detections["boxes"]
				scores = detections["scores"]
				labels = detections["labels"]

			preds = np.zeros(boxes.shape[0],dtype=[('idx', 'i4'), ('labels', 'i4'),('scores','f4'),('x1', 'f4'),('y1', 'f4'),('x2', 'f4'),('y2', 'f4')])
			preds["idx"] = batch_i
			preds["labels"] = labels
			preds["scores"] = scores
			preds["x1"] = boxes[:,0]
			preds["y1"] = boxes[:,1]
			preds["x2"] = boxes[:,2]
			preds["y2"] = boxes[:,3]
			preds = preds[preds["scores"] > args["confidence"]]
			preds = preds.tolist()
			preds = [list(ele) for ele in preds]

     		#[train_idx, class_prediction, prob_score, x1, y1, x2, y2]
			for p in preds:
				# corners need to have min,max
				if p[3]>p[5]:
					p[3], p[5] = p[5], p[3]
				if p[4]>p[6]:
					p[4], p[6] = p[6], p[4]

			gts = []
			for t in targets:
				bbox = t["bbox"]
				bbox = xywh2xyxy(bbox)
				# corners need to have min,max
				if bbox[0]>bbox[2]:
					bbox[0], bbox[2] = bbox[2], bbox[0]
				if bbox[1]>bbox[3]:
					bbox[1], bbox[3] = bbox[3], bbox[1]
				gts.append([batch_i,t["category_id"],1.0,bbox[0],bbox[1],bbox[2],bbox[3]])

			if len(gts) > 0 :
				if args["iou"] is not None :
					score = map(preds,gts,iou_threshold=args["iou"],box_format="corners",num_classes=len(CLASSES))
					score = score.cpu().detach().numpy()
					map_scores.append(score)
				else :
					for i in [0.5,0.7,0.9]:
						score = map(preds,gts,iou_threshold=i,box_format="corners",num_classes=len(CLASSES))
						score = score.cpu().detach().numpy()
						if i == 0.5:
							map_scores_05.append(score)
						elif i == 0.7:
							map_scores_07.append(score)
						elif i == 0.9:
							map_scores_09.append(score)

		if args["iou"] is not None :
			map_scores = np.mean(np.array(map_scores))
			np.save("mAP_{}_option_{}".format(str(args["iou"]),str(args["option"])),map_scores)
		else : 
			map_scores_05 = np.mean(np.array(map_scores_05))
			map_scores_07 = np.mean(np.array(map_scores_07))
			map_scores_09 = np.mean(np.array(map_scores_09))
			np.save("mAP_{}_option_{}".format("0.5",str(args["option"])),map_scores_05)
			np.save("mAP_{}_option_{}".format("0.7",str(args["option"])),map_scores_07)
			np.save("mAP_{}_option_{}".format("0.9",str(args["option"])),map_scores_09)

if __name__ == "__main__":
    
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-a", "--augment", action='store_true', help='augmented inference')
	ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
	ap.add_argument("-o", "--option", type=int, default=0,help="select the type of tta")
	ap.add_argument("-i", "--iou", type=float, default=None,help="Intersection Over Union threshold")
	args = vars(ap.parse_args())

	main()