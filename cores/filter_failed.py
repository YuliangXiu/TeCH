import os
from glob import glob
import numpy as np

subjects = np.loadtxt("all_subjects.txt", dtype=str, delimiter=" ")[:, 0]
subjects = [f"./results/tech/{outfit}/" for outfit in subjects]

failed_econ = []
failed_blip = []
failed_sd = []
failed_sds = []

def lst_to_file (lst, filename):
    
    with open(filename, "w") as f:
        for item in lst:
            item = item.replace("./results/tech/","")
            subject, outfit = item.split('/')[-3:-1]
            f.write(f"{item[:-1]} {subject} {outfit}\n")

for subject in subjects:
    if not os.path.exists(os.path.join(subject, "obj/07_C_smpl.obj")):
        failed_econ.append(subject)
    else:
        if not os.path.exists(os.path.join(subject, "prompt.txt")):
            failed_blip.append(subject)
            
    if not os.path.exists(os.path.join(subject, "sd_model/model_index.json")):
        failed_sd.append(subject)
        
    # if not os.path.exists(os.path.join(subject, "geometry/visualize/geometry_ep0100_norm.mp4")):
    if not os.path.exists(os.path.join(subject, "obj/07_C_texture.obj")):
    # if not os.path.exists(os.path.join(subject, "texture/visualize/texture_ep0070_rgb.mp4")):
        failed_sds.append(subject)

print("Failed econ: ", len(failed_econ))
print("Failed blip: ", len(failed_blip))
print("Failed SD: ", len(failed_sd))
print("Failed SDS: ", len(failed_sds))


lst_to_file(failed_econ, "failed_econ.txt")
lst_to_file(failed_blip, "failed_blip.txt")
lst_to_file(failed_sd, "failed_sd.txt")
lst_to_file(failed_sds, "failed_sds.txt")
        


