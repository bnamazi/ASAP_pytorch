import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from dataloaders import ASAPDataset
from data_augmentation import transform_train, transform_val

# from torch.utils.tensorboard import SummaryWriter
from models import ConvLSTM
from losses import LabelSmoothingCrossEntropy, EMD
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import imshow, show_img, online_mean_and_sd, get_lr

# from skimage import io, transform
import numpy as np
from tqdm import tqdm
import time
import os
import copy
import argparse
from sklearn.metrics import f1_score
import wandb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_dir", type=str, default="./data", help="path to training directory "
)
parser.add_argument(
    "--test_dir", type=str, default="./data", help="path to test directory"
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="batch size (default: 32)"
)
parser.add_argument(
    "--num_epochs", type=int, default=100, help="number of epochs (default: 50)"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--positional_encoding", type=bool, default=True, help="encoding the frame number"
)
parser.add_argument(
    "--emd_loss",
    type=bool,
    default=False,
    help="earth movers distance as the main loss function",
)
parser.add_argument(
    "--emd_reg",
    type=bool,
    default=True,
    help="earth movers distance as a second loss function",
)
parser.add_argument("--LR", type=float, default=0.00005, help="initial learning rate")
parser.add_argument(
    "--decay_rate", type=float, default=0.7, help="rate of learning rate decay"
)
parser.add_argument(
    "--decay_step",
    type=int,
    default=5,
    help="number of steps before learning rate decay",
)
parser.add_argument(
    "--saved_path",
    type=str,
    default="./runs/Resnet50_rcnn_ASAP4.pth",
    help="path to the saved model",
)
parser.add_argument(
    "--summary_path",
    type=str,
    default="./runs/Resnet50_rcnn_ASAP4",
    help="path to save tensorboard summaries",
)
parser.add_argument(
    "--loss_ratio",
    type=float,
    default=0.1,
    help="the weight of EMD loss when added to CE",
)


args = parser.parse_args()


mode = args.mode
training_strategy = "joint"

device = torch.device(args.device)

num_classes = [0, 0, 0]
num_workers = 64
num_validation_samples = 9000
# PATH = './runs/'
# clip_size = 16

model_saved_path = args.saved_path
model_summary_path = args.summary_path


def run():
    wandb.init(project="ASAP", entity="bnamazi")

    config = wandb.config

    config.learning_rate = args.LR
    config.model = "CNN-Resnet50"
    config.batch_size = args.batch_size
    config.positional_encoding = args.positional_encoding
    config.emd_reg = args.emd_reg
    # torch.multiprocessing.freeze_support()
    # writer = SummaryWriter(log_dir=model_summary_path)

    num_epochs_stop = 15
    epochs_no_improve = 0
    early_stop = False

    train_dataset = ASAPDataset(
        data_dir=args.train_dir,
        transform=transform_train,
        positional_encoding=args.positional_encoding,
        train=True,
    )
    test_dataset = ASAPDataset(
        data_dir=args.test_dir,
        transform=transform_val,
        positional_encoding=args.positional_encoding,
        train=False,
    )
    validation_dataset = torch.utils.data.Subset(
        test_dataset, range(1, num_validation_samples)
    )

    # print(len(train_dataset))
    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            validation_dataset, batch_size=16, shuffle=True, num_workers=num_workers
        ),
        "test": DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=num_workers
        ),
    }

    model = ConvLSTM(num_classes=19, mode=mode)
    # print(model)
    model = model.to(device)

    # swa_model = AveragedModel(model)

    if args.emd_loss:
        criterion = EMD()
    else:
        criterion = LabelSmoothingCrossEntropy()
    criterion_emd = EMD()
    criterion_ML = nn.MultiLabelSoftMarginLoss
    optimizer = optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    scheduler = lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.decay_rate
    )  # CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.001)

    swa_start = 10
    # swa_scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.001, cycle_momentum=False) #SWALR(optimizer, swa_lr=0.0007)

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # train and validate
    for epoch in range(args.num_epochs):

        # Each epoch has a training and validation phase
        for split in ["train", "val"]:
            if split == "train":
                # wandb.watch(model, log_freq=10, log=all)
                model.train()  # Set model to training mode
                dataset_length = len(train_dataset)
            else:
                model.eval()  # Set model to evaluate mod
                dataset_length = len(validation_dataset)

            running_loss = 0.0
            running_corrects = 0, 0, 0, 0
            y_true = []
            y_pred = []

            # Iterate over data.
            with tqdm(total=dataset_length) as epoch_pbar:
                epoch_pbar.set_description(f"Epoch {epoch}/{args.num_epochs - 1}")
                for batch, (image, target, frame_num, path) in enumerate(
                    dataloaders[split]
                ):
                    # grid = utils.make_grid(image)
                    # grid = show_img(grid)
                    # writer.add_image('images', grid, 0)
                    frame_num = frame_num.to(device)
                    inputs = image.to(device)

                    labels = target.to(device)
                    labels = labels.reshape(-1)

                    # writer.add_graph(model(inputs, frame_num=frame_num), inputs)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    model.lstm.reset_hidden_state()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(split == "train"):

                        outputs = model(inputs, frame_num)
                        # print(outputs.shape, labels.shape)
                        # if epoch > swa_start and split == 'val':
                        #     swa_outputs = swa_model(inputs, frame_num)[0]
                        #     _, preds = torch.max(swa_outputs, 1)
                        # else:
                        _, preds = torch.max(outputs, -1)
                        loss = criterion(outputs, labels.long())
                        if args.emd_reg:
                            loss += args.loss_ratio * criterion_emd(
                                outputs, labels.long()
                            )

                        # backward + optimize only if in training phase
                        if split == "train":

                            loss.backward(retain_graph=True)
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    running_accuracy = (running_corrects).double() / (
                        args.batch_size * (epoch + batch)
                    )

                    y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
                    y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))

                    desc = (
                        f"Epoch {epoch}/{args.num_epochs - 1} - loss {loss.item():.4f}"
                    )
                    epoch_pbar.set_description(desc)
                    epoch_pbar.update(inputs.shape[0])

                    if split == "train":
                        if batch % 10 == 9:
                            wandb.log({f"{split} loss": loss})
                            wandb.log({f"{split} accuracy": running_accuracy})

            if split == "train":
                # if epoch > swa_start:
                #     swa_model.update_parameters(model)
                #     swa_scheduler.step()
                # else:
                scheduler.step()

            epoch_loss = running_loss / dataset_length
            epoch_acc = running_corrects.double() / (dataset_length)
            epoch_f1 = f1_score(y_pred=y_pred, y_true=y_true, average="macro")
            epoch_f1_micro = f1_score(y_pred=y_pred, y_true=y_true, average="micro")

            print(
                "{} Loss: {:.4f} Acc: {:.4f} F1: {:.4f}".format(
                    split, epoch_loss, epoch_acc, epoch_f1
                )
            )

            if split == "val":

                wandb.log({f"{split} f1_macro": epoch_f1})
                wandb.log({f"{split} f1_micro": epoch_f1_micro})
                wandb.log({f"{split} accuracy": epoch_acc})

            # deep copy the model
            if split == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    epochs_no_improve = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), model_saved_path)
                else:
                    epochs_no_improve += 1

            # Early stopping
            if epoch > swa_start and epochs_no_improve == num_epochs_stop:
                print("Early stopping!")
                early_stop = True
                break

        if early_stop:
            break
    # torch.optim.swa_utils.update_bn(dataloaders['train'], swa_model(frame_num=frame_num))

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), model_saved_path)

    # test the model
    # y_pred = []
    # y_true = []
    # with torch.no_grad():
    #     for batch, (image, target, target1, target2) in enumerate(dataloaders['test']):
    #         inputs = image.to(device)
    #         labels = target2.to(device)
    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #
    #         y_true = np.concatenate((labels.cpu().data.numpy(), y_true))
    #         y_pred = np.concatenate((preds.cpu().data.numpy(), y_pred))
    #
    # print(f1_score(y_pred=y_pred, y_true=y_true, average='macro'))


if __name__ == "__main__":
    run()
