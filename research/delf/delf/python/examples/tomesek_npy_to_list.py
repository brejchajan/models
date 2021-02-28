import numpy as np
import os
from tqdm import tqdm

d = np.load('/mnt/matylda1/itomesek/PUBLIC/Switzerland_evaluation/Alps_photos_to_segments_compact_test.npy', allow_pickle='True').item()
#root_path = '/mnt/matylda1/Locate/data/photoparam_raw/netvlad/Alps/database_depth'
root_path = '/mnt/data1/data/delg/Alps_query'

with open('switzerland_query_noscale.txt', 'w') as file:
    for img_path in tqdm(d['qImageFns']):
        depth_path = os.path.join(root_path, img_path.replace("_segments.png", "_depth.exr"))
        file.write(depth_path + "\n")
