#run the training
from Main import main
import os
import json
import numpy as np

loss,metrics=main()



if not os.path.exists("/kaggle/working/results"):
    os.mkdir("/kaggle/working/results")

loss_file_path = os.path.join("/kaggle/working/results", "loss.npy")
metrics_file_path = os.path.join("/kaggle/working/results", "metrics.json")

np.save(loss_file_path,loss)
    
with open(metrics_file_path, 'w') as f: 
    json.dump(metrics, f)