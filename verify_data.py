import os

def new_train_valid_txt(train_path, valid_path):
    train = open(train_path, 'r')
    train_lines = train.readlines()

    new_train = open('new_train.txt', 'w')
    for path in train_lines:
        if os.path.exists(path.strip()):
            new_train.write(path)

    new_valid = open('new_valid.txt', 'w')
    valid = open(valid_path, 'r')
    valid_lines = valid.readlines()
    for path in valid_lines:
        if os.path.exists(path.strip()):
            new_valid.write(path)


if __name__=='__main__':
    new_train_valid_txt(
        train_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/train.txt',
        valid_path='/Users/rudy/Desktop/Development/Virtualenv/darknet-yolo-utils/digit_data_label/valid.txt'
    )