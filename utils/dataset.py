
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
import os
import xml.etree.ElementTree as ET
from PIL import Image



annotation_root = "../data/annotations"

def xml_to_dict(file):

    file_path = annotation_root + "/" + file
    tree = ET.parse(file_path)
    root = tree.getroot()


    xmin = int(root.find("./object/bndbox/xmin").text)
    xmax = int(root.find("./object/bndbox/xmax").text)
    ymin = int(root.find("./object/bndbox/ymin").text)
    ymax = int(root.find("./object/bndbox/ymax").text)

    return {"file_name":root.find("./filename").text,
                "label":root.find("./object/name").text,
                "widht": root.find("./size/width").text,
                "height" : root.find("./size/height").text,
                "depth" : root.find("./size/depth").text,
                "xmin" : xmin,
                "xmax" : xmax,
                "ymin" : ymin,
                "ymax" : ymax,
                "target" : [xmin,xmax,ymin,ymax]}





class CatDogDataset(Dataset):

    def __init__(self,root,transformes=None,label_dict = {"dog":1,"cat":2}):

        self.root = root
        self.transformes = transformes 
        self.files = [file.split(".")[0] for file in os.listdir(self.root+"/images")]

        self.label_dict = label_dict


    def __getitem__(self, index):

        file_name = self.files[index]
        # file_name = file.split(".")[0]
        xml_file = file_name + ".xml"
        image_file = file_name + ".png"

        ann = xml_to_dict(xml_file)

        target = {"boxes":torch.as_tensor([
                  ann['xmin'],
                  ann["ymin"],
                  ann["xmax"],
                  ann["ymax"]])}
        
        target['labels'] = torch.as_tensor([self.label_dict[ann['label']]])
        target['image_id'] = index

        # target = torch.as_tensor([[
        #           ann['xmin'],
        #           ann["ymin"],
        #           ann["xmax"],
        #           ann["ymax"]]]),torch.as_tensor([self.label_dict[ann['label']]])
        
        image = Image.open(self.root+"/images/"+image_file).convert("RGB")


        transform = transforms.Compose([transforms.PILToTensor()])
        image_tensor = transform(image)
        

        print(image_tensor.size())
        # quit()

        return image_tensor,target
   

    def __len__(self):
        
        return len(self.files)



# cat = CatDogDataset(root="../data/")
# print(cat[0])




if __name__ == "__main__":
    
    print("Inside main!!")



    cat = CatDogDataset(root="../data/")

    print(cat[0])

    print(cat.root)