import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision_lib.engine import train_one_epoch, evaluate
from torchvision_lib import utils, transforms as T
from data_loader import ObjectDataset


def get_model():
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 2  # 6 class (buses) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def choose_device(limit_to_cpu=False):
    if limit_to_cpu:
        return torch.device('cpu')
    else:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(device, num_epochs, batch_size, add_text):
    # use our dataset and defined transformations
    dataset = ObjectDataset('doors/naor/train', get_transform(train=True))
    dataset_test = ObjectDataset('doors/naor/test', get_transform(train=False))


    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # model on the right device
    model = get_model()
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for num_ephocs epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every num_epochs iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        # if epoch % 10 == 0:
        # save each epoch
        save_model(model, epoch, batch_size, add_text)
    print("That's it trained!")

    return model, dataset_test


def save_model(model, num_epochs, batch_size, add_text):
    import datetime
    from shutil import copyfile
    import os

    models_dir = "saved_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    unique_filename = f"model__num_epochs-{num_epochs}__batch_size-{batch_size}__add_text={add_text}__{timestamp}.pt"
    unique_filepath = os.path.join(models_dir, unique_filename)
    standard_filepath = "model_state_dict.pt"

    torch.save(model.state_dict(), unique_filepath)
    copyfile(unique_filepath, standard_filepath)

    print("updated model_state_dict.pt")
    print(f" also saved to {unique_filepath}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--add_text')

    args = parser.parse_args()

    device = choose_device()

    # parse cli arguments or take default values
    num_epochs = int(args.num_epochs)
    if num_epochs is None:
        print("USAGE python trainer.py --num_epochs <num>")
        exit(0)
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = 2

    if args.add_text:
        add_text = args.add_text
    else:
        add_text = ""


    model, dataset_test = train(device=device, num_epochs=num_epochs, batch_size=batch_size, add_text=add_text)

    save_model(model, num_epochs, batch_size, add_text)
    print("DONE training!")
