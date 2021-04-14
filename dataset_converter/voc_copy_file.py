import os
import shutil

def copy(voc_file, img_dir, target_dir):
    with open(voc_file, "r") as f:
        for line in f.readlines():
            split = line.split()
            file_path = split[0]

            img_path = os.path.join(img_dir, file_path)
            shutil.copy2(img_path, target_dir)



if __name__ == '__main__':

    copy(
        voc_file='/Users/rav/Desktop/netvision_voc_output.txt',
        img_dir='/Users/rav/Desktop/netvision_label/',
        target_dir='/Users/rav/Desktop/netvision_label/labeled_img')
