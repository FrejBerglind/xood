"""
As discussed here https://discuss.pytorch.org/t/filenotfounderror-couldnt-find-any-class-folder/138578/3 
the validation images need to be set in subfolders inorder to work with torchvision.datasets.ImageFolder.
1. The main files are: 
    # ILSVRC2012_img_val.rar ( download the validation set) 
    # ILSVRC2012_devkit_t12.tar.
    # ILSVRC (comes with 155GB package)
    
2. Download them from Imagenet 2012 dataset server
3. Extract all files.
4. ILSVRC2012_validation_ground_truth.txt  is the ground truth that comes with dev kit. 
5. The class names will be selected from annotations files (
    from  "ILSVRC"/"Annotations"/"CLS-LOC"/ "val" directory) ,
   and subfolders will be created based on the class names
6. ILSVRC2012_img_val  contains validation images, which need to put inside seperate  subfolders 
7. All subfolders will be put inside a parent folder named "val_subfolders".
8. For every other run you should delete the "val_subfolders" folder first created by previous run.

"""

import pandas as pd
import shutil
from pathlib import Path

path = Path.cwd()  # root imagenet data folder
img_path = path / "ILSVRC2012_img_val"  # ILSVRC2012_img_val folder path
val_label_path = path / "ILSVRC2012_devkit_t12" / "data"  # ILSVRC2012_devkit_t12/data folder path
annotations_path = path / "ILSVRC" / "Annotations" / "CLS-LOC" / "val"  # /ILSVRC/Annotations/CLS-LOC/val/"
csv_file_saving_path = path

# =============================================================================
#  extracting class name from annotation folder
# =============================================================================

files_in_folder = Path.iterdir(annotations_path)
img_nm_list = []
clss_list = []
data2 = pd.DataFrame(columns=['img_id', 'class_nm'])

for file in (annotations_path.glob("*.xml")):
    df = pd.read_xml(file)
    img_nm = df['filename'].iloc[1]
    clss = df['name'].iloc[5]
    img_nm_list.append(img_nm)
    clss_list.append(clss)

df1 = pd.DataFrame(img_nm_list, columns=['img_id'])
df2 = pd.DataFrame(clss_list, columns=['class_nm'])
df = pd.concat([df1, df2], axis=1, join='inner')
df = df.sort_values(by='img_id')
csv_path = path / "validation_img_name_and_class2.csv"
df.to_csv(csv_path, header=False)  # ILSVRC2012_val_00000001 , n01751748
df3 = df["class_nm"]
class_nm_list = df3.values.tolist()

# =============================================================================
#  creating a dictionary - "val_subfolders" for image names and their corresponding classes
# =============================================================================
image_name_list = []

for child in sorted(img_path.iterdir()):
    # print((child.parts[-1]))
    image_name_list.append((child.parts[-1]))

print(image_name_list[0:5])

val_subf_img_dir = path / "val_subfolders"
if not val_subf_img_dir.exists():
    val_subf_img_dir.mkdir()

# Open and read val label 
fp = open((val_label_path / 'ILSVRC2012_validation_ground_truth.txt'), 'r')
label_list = fp.readlines()
print("label_list[0:5]", label_list[0:5])
print("clss_list[0:5]", clss_list[0:5])

# Create dictionary to store img filename and corresponding
val_img_dict = {}
for idx, label in enumerate(class_nm_list):  # CHANGED clss_list
    key = image_name_list[idx]
    val_img_dict[key] = label
fp.close()

# Display first 10 entries of resulting val_img_dict dictionary
print("# val_img_dict (first five pairs)")
print({k: val_img_dict[k] for k in list(val_img_dict)[:5]})

# =============================================================================
#   creating 1000 subfolders inside "val_subfolders"based on class
# =============================================================================
# Create subfolders (if not present) for validation images based on label,
#    and move images into the respective folder    
for img, folder in val_img_dict.items():
    subfolder = val_subf_img_dir / folder
    if not subfolder.exists():
        subfolder.mkdir()
    if ((img_path / img)).exists():  # if the image remains in ILSVRC2012_img_val
        source = img_path / img
        destination = (subfolder / img)
        shutil.copy(source, destination)
