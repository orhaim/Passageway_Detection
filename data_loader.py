import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import ast


class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        tags_dict = {}
        with open(os.path.join(root, "annotations.txt"), "r") as ann_f:
            lines = ann_f.readlines()
        for line_ in lines:
            line = line_.replace(' ', '')
            imName = line.split(':')[0]
            anns_ = line[line.index(':') + 1:].replace('\n', '')
            anns = ast.literal_eval(anns_)
            if (not isinstance(anns, tuple)):
                anns = [anns]
            tags_dict[imName] = anns
        self.tags_dict = tags_dict

        # HACK to ignore imgs without annotationsTrain
        for img_name in self.imgs:
            if tags_dict.get(img_name, None) is None:
                print(f"{img_name} not found in annotations.txt. IGNORE IT")
                self.imgs.remove(img_name)


    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        # TODO: DELETE HACKKKKKK
        try:
            anns = self.tags_dict[self.imgs[idx]]
        except KeyError as e:
            print(e)
            anns = []
        # "([xmin1, ymin1, width1, height1,color1], [xmin1, ymin1, width1, height1,color1])"
        # get bounding box coordinates for each mask
        num_anns = len(anns)  # num of boxes
        boxes = []
        labels = []
        for ann in anns:
            xmin = ann[0]
            xmax = xmin + ann[2]
            ymin = ann[1]
            ymax = ymin + ann[3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann[4])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_anns,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
