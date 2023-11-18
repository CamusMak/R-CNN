
import os
import shutil
import random


def split(source_image_root,source_annoation_root,train_root,validation_root,test_root,random_seed=None,train_size=0.7,validation_size=0.2,test_size=0.1):

    files = os.listdir(source_image_root)
    files = [file.split(".")[0] for file in files]

    random.seed(random_seed)

    random.shuffle(files)

    len_files = len(files)
    train_index = int(len_files*train_size)
    test_index = len_files - int(len_files*test_size)

    # train_files = files[:train_index]
    # validation_files = files[train_index:test_index]
    # test_files = files[test_index:]

    for i in range(len_files):


        print("*"*30,round((i+1)/len_files,2),"%","*"*30)

        file_name = files[i]
             
        # train
        if i < train_index:

            image_file_source = source_image_root + "/" + file_name + ".png"
            annotation_file_source = source_annoation_root + "/" + file_name + ".xml"

            image_file_dest = train_root + "/images/" + file_name + ".png"
            annotation_file_dest = train_root + "/annotations/" + file_name + ".xml"




        # validation
        elif (i > train_index) and (i < test_index):

            image_file_source = source_image_root + "/" + file_name + ".png"
            annotation_file_source = source_annoation_root + "/" + file_name + ".xml"

            image_file_dest = validation_root + "/images/" + file_name + ".png"
            annotation_file_dest = validation_root + "/annotations/" + file_name + ".xml"


        
        # test
        else:
            image_file_source = source_image_root + "/" + file_name + ".png"
            annotation_file_source = source_annoation_root + "/" + file_name + ".xml"

            image_file_dest = test_root + "/images/" + file_name + ".png"
            annotation_file_dest = test_root + "/annotations/" + file_name + ".xml"
        

        shutil.copy(image_file_source,image_file_dest)
        shutil.copy(annotation_file_source,annotation_file_dest)




    
    


    



    
