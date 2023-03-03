#!/usr/bin/env python3

import rospkg
import numpy as np
import os
from aarapsi_intro_pack import aarapsi_NetVLAD_feature_extract
from aarapsi_intro_pack import aarapsi_HybridNet_feature_extract
from PIL import Image
import time

#################### Open test image ###################################
img = Image.open(rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/scripts/testing/frame_id_000000.png')

#################### NetVLAD test example ##############################
config_path = rospkg.RosPack().get_path(rospkg.get_package_name(os.path.abspath(__file__))) + '/src/aarapsi_intro_pack/Patch_NetVLAD/patchnetvlad/configs/netvlad_extract.ini'
model, config = aarapsi_NetVLAD_feature_extract.load_model(config_path, False)
t1 = time.time()
qry_ftrs = aarapsi_NetVLAD_feature_extract.feature_query_extract(img, model, False, config)
t2 = time.time()
print(t2-t1)
print(qry_ftrs.shape)

#################### HybridNet test example ############################
model, transformer = aarapsi_HybridNet_feature_extract.load_HybridNet()
t1 = time.time()
qry_ftrs = aarapsi_HybridNet_feature_extract.HybridNet_query_extract(model, transformer, np.asarray(img))
t2 = time.time()
print(t2-t1)
print(qry_ftrs.shape)