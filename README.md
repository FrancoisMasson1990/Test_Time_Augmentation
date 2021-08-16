# Test-time augmentation (TTA) for object detection

## Requirement

Before running the project, several python packages need to be installed. The following project used Python 3.8.* version.

```
python3.8 -m pip install -r requirements.txt
```

## MS-COCO detection dataset

In order to get the MS-COCO detection (val 2017) dataset and its corresponding annotations, the following script should be run :

```
./cocodataset.sh
```

This will download under your main root, a coco folder with all the images and labels.


## Benchmarking

Run the following code to obtain the benchmark for the pre-trained model without TTA methods with an IOU of [0.5,0.7,0.9] and with 0.5 of confidence.

```
python3.8 test_time_augmentation.py
```

You can fine-tune the code by passing extra argument 

```
python3.8 test_time_augmentation.py --iou 0.7 --confidence 0.6
```

## TTA-Test

Run the following code to obtain the benchmark for the pre-trained model with TTA methods with an IOU of [0.5,0.7,0.9] and with 0.5 of confidence.

```
python3.8 test_time_augmentation.py -a -o 1
```

Argument o refers to a list of options for TTA defined as :

- option 0 : No TTA
- option 1 : Scaling of 0.9
- option 2 : Horizontal Flip
- option 3 : [Horizontal Flip, Scaling of 0.9, Scaling of 1.1]
- option 4 : [Horizontal Flip, Scaling of 0.9, Scaling of 1.1, Vertical Flip, Left Rotation]

## Presentation & Results

See the document Lunit_Project.pdf for a more detailled description of the project.

