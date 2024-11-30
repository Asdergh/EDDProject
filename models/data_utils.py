import numpy as np 
import matplotlib.pyplot as plt
import json as js
import os
import torch as th
import cv2

from torch.utils.data import (
    Dataset,
    DataLoader
)
from torchvision.io import read_image
from torchvision.transforms.v2 import (
    Resize,
    Normalize,
    Compose
)







class ImageDetectionSet(Dataset):

    def __init__(
            self,
            data_dir: str,
            data_split: str,
            transform: th.nn.Module = None
    ) -> None:
        

        super().__init__()
        self.tranform = transform
        self.data_path = os.path.join(data_dir, data_split)
        self.change_annotations_coco_()

        annots_file = os.path.join(self.data_path, "_annotations.json")
        with open(annots_file, "r") as js_file:
            annots = js.load(js_file)
        
        self.images_js = annots["images"]
        self.annots_js = annots["annotations"]
    

    def change_annotations_coco_(self):

        annots_path = os.path.join(self.data_path, "_annotations.coco.json")
        with open(annots_path, "r") as js_file:

            data = js.load(js_file)
            images = {sample["id"]: sample for sample in data["images"]}
            annots = {sample["image_id"]: sample for sample in data["annotations"]}

        data["images"] = images
        data["annotations"] = annots
        annots_path = os.path.join(self.data_path, "_annotations.json")

        with open(annots_path, "w") as js_file:
            js.dump(data, js_file)

    def __len__(self):
        return len(self.images_js)

    def __getitem__(self, idx):

        image = read_image(os.path.join(self.data_path, self.images_js[f"{idx}"]["file_name"])).to(th.float32)
        if not self.tranform is None:
            image = self.tranform(image)

        image_label = self.annots_js[f"{idx}"]["category_id"]
        image_bbox = np.array(self.annots_js[f"{idx}"]["bbox"])

        return (
            image,
            image_label,
            image_bbox
        ) 







if __name__ == "__main__":

    data_dir = "C:\\Users\\1\\Downloads\\lateralidade.v16i.coco"
    transforms = Compose([
        #Resize(size=(128, 128)),
        Normalize(mean=[0.128, 0.027, 0.341],
                  std=[0.128, 0.27, 0.341])
    ])

    trainset = ImageDetectionSet(
        data_dir=data_dir,
        data_split="train",
        # transform=transforms
    )
    testset = ImageDetectionSet(
        data_dir=data_dir,
        data_split="test",
        #transform=transforms
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=trainset,
        batch_size=32,
        shuffle=True
    )
    for sample in train_loader:
        break
    

   
    plt.style.use("dark_background")
    _, axis = plt.subplots()

    image = sample[0].T / 255.0
    print(image.size())
    # print(image.T.size())
    # axis.imshow(image)
    # plt.show()
        