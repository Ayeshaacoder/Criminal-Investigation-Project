import numpy as np
from pathlib import Path
from CropImage import crop
from GetFeatures import Features
from PIL import Image
import cv2

def match(path):
    featuresDB = []
    img_pathsDB = []
    for feature_path in Path("./static/Features").glob("*.npy"):
        featuresDB.append(np.load(feature_path))
        img_pathsDB.append(Path("./static/Profile") / (feature_path.stem + ".jpg"))  
    featuresDB = np.array(featuresDB)
    img=cv2.imread(path)
    img=crop(img)
    save_path = 'static/CombinedFace/'
    save_path = save_path + "query.jpg" 
    cv2.imwrite(save_path, img)
    query=Features(save_path) 
    dists = np.linalg.norm(np.subtract(featuresDB,query), axis=1)
    ids = np.argsort(dists)[:6]  # Top 3 results
    # print(ids)
    scores=[]
    check=[]
    for id in ids:
        value=img_pathsDB[id].stem.split('_')[1]
        if value  in check:
            continue
        else:
            scores.append([dists[id], img_pathsDB[id]])
            check.append(value)
            # print(scores)
    scores=scores[:3]
        
    # scores = [(dists[id], img_pathsDB[id]) for id in ids]
    return scores