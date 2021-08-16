# Concatenate results from numpy file into bar graph

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from natsort import natsorted

## Benchmark graph
options = ["option_0"]

# create data
df = pd.DataFrame()
iou = ["0.5","0.7","0.9"]
df["IOU"] = iou
for opt in options:
    files = natsorted(glob.glob("./results/*{}*npy".format(str(opt))))
    results = []    
    for f in files:
        results.append(100*np.load(f))
    results = np.array(results).flatten().tolist()
    df[opt] = results

# plot grouped bar chart
plot = df.plot(x='IOU',kind='bar',stacked=False,title='Test-Time Augmentation performance')
plot.set_ylabel("mAP (%")
plot.get_figure().savefig('output_total.png')


## TTA results
options = ["option_0","option_1","option_2","option_3","option_4"]
# create data
df = pd.DataFrame()
iou = "0.5"
df["IOU"] = [iou]
for opt in options:
    files = natsorted(glob.glob("./results/*{}*npy".format(str(opt))))
    results = []    
    for f in files:
        if iou in f:
            results.append(100*np.load(f))
    results = np.array(results).flatten().tolist()
    df[opt] = results

plot = df.plot(x='IOU',kind='bar',stacked=False,title='Test-Time Augmentation performance')
plot.set_ylabel("mAP (%")
plot.set_ylim([50,70])
plot.get_figure().savefig('output_0.5.png')