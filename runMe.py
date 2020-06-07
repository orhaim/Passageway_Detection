import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2  # 6 class (buses) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def create_line(pic_name, boxes, labels):
    line_str = f"{pic_name}:"  # DSCF1013.JPG:[1217,1690,489,201,1],[1774,1619,475,224,2]
    for box_idx in range(len(boxes)):
        box = boxes[box_idx]
        x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        label = labels[box_idx]
        if box_idx == 0:
            box_str = f"[{x0},{y0},{x1 - x0},{y1 - y0},{label}]"
        else:
            box_str = f",[{x0},{y0},{x1 - x0},{y1 - y0},{label}]"

        line_str += box_str
    line_str += '\n'
    return line_str


def run(myAnnFileName, test_dir):
    imgs = list(sorted(os.listdir(test_dir)))

    model = get_model()
    model.load_state_dict(torch.load("model_state_dict.pt"))  # load weights
    model.eval()  # put the model in evaluation mode

    with open(myAnnFileName, "w") as annFileEstimations:
        for idx in range(len(imgs)):
            # pick one image from the buses folder
            img_name = imgs[idx]
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = transforms.Compose([transforms.ToTensor()])(
                img)  # TODO Check out I did a good transform (it's similiar to get_transform)

            with torch.no_grad():
                prediction = model([torch.as_tensor(np.array(img))])[
                    0]  # TODO: how to predict with higher threshold? (we find too many objects..)

            boxes = prediction.get("boxes").cpu().numpy()  # boxes is tensor([[x0, y0, x1, y1], ..])
            labels = prediction.get("labels").cpu().numpy()  # labels is tensor([label, label, ..])
            line = create_line(img_name, boxes, labels)
            print(line)
            annFileEstimations.write(line)


if __name__ == "__main__":
    myAnnFileName = "annotations_with_loaded_model.txt"
    test_dir = os.path.join("doors", "naor", "test", "images")
    run(myAnnFileName, test_dir)
    print(f"created the file {myAnnFileName}")
