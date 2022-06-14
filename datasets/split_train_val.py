import os
import shutil

if __name__ == '__main__':
    data_path = "/home/Face_recognition/v1_pretraining_resnet18/data/105_classes_pins_dataset/"
    train_path = "/home/Face_recognition/v1_pretraining_resnet18/data/train/"
    val_path = "/home/Face_recognition/v1_pretraining_resnet18/data/val/"
    data_list = os.listdir(data_path)

    for d in data_list:

        person_path = data_path + d
        img_list = os.listdir(person_path)
        val_size = len(img_list) // 4
        train_size = len(img_list) - val_size

        train_path_person = train_path + d
        train_path_person = train_path_person.replace(' ', '')
        os.makedirs(train_path_person, exist_ok=True)
        val_path_person = val_path + d
        val_path_person = val_path_person.replace(' ', '')
        os.makedirs(val_path_person, exist_ok=True)

        for train_im in img_list[:train_size]:
            im_path = person_path + '/' + train_im
            train_im_path = train_path_person + '/' + train_im
            train_im_path = train_im_path.replace(' ', '')
            shutil.copy(im_path, train_im_path)

        for val_im in img_list[train_size:]:
            im_path = person_path + '/' + val_im
            val_im_path = val_path_person + '/' + val_im
            val_im_path = val_im_path.replace(' ', '')
            shutil.copy(im_path, val_im_path)

    print("done.")
