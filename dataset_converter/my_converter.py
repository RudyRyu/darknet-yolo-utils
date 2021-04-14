import os

def convert_mine_to_voc(label_dir_path, voc_txt):

    f_out = open(voc_txt, 'w+')

    for label_file in os.listdir(label_dir_path):
        name, ext = os.path.splitext(label_file)
        if ext != '.txt':
            continue

        with open(os.path.join(label_dir_path, label_file), "r") as f:
            f_out.write(name + '.JPEG')
            for line in f.readlines()[1:]:
                coord = ','.join(line.split()) + ',0'
                f_out.write(' ' + coord)

        f_out.write('\n')

    f_out.close()

convert_mine_to_voc(
    label_dir_path='/Users/rav/Desktop/BBox-Label-Tool/Labels/001',
    voc_txt='voc.txt')
