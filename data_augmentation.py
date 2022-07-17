from torchvision import transforms, utils
#import transforms_video as VT
# from torchvideotransforms.video_transforms import Compose, Resize, RandomCrop, RandomRotation, ColorJitter, Normalize, RandomHorizontalFlip, RandomResize, CenterCrop
# from torchvideotransforms.volume_transforms import ClipToTensor
#import torchvideo.transforms as VT
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


transform_train = transforms.Compose([
    transforms.Resize((256,256), interpolation=2),
    transforms.RandomApply(
    [
    transforms.RandomOrder((
    transforms.RandomResizedCrop(224, scale=(0.65, 1.20), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.1,0.1), shear=0.05, resample=False, fillcolor=0),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomGrayscale(p=0.05),
    transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomHorizontalFlip(p=0.2),

    ))
     ], 0.85),
    transforms.Resize((224,224), interpolation=2),
    transforms.ToTensor(),
    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
    ])

transform_val = transforms.Compose([#transforms.Resize((256,256), interpolation=2),
                                    transforms.Resize((224,224), interpolation=2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
                                    ])

# transform_train_video = VT.Compose([
#     transforms.RandomApply(
#      [
#     # transforms.RandomOrder((
#     VT.RandomResizedCropVideo(112, scale=(0.65, 1.20), ratio=(0.75, 1.3333333333333333), interpolation=2),
#     #transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=None, shear=0.05, resample=False, fillcolor=0),
#     #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
#     #VT.RandomGrayscale(p=0.05),
#     #transforms.RandomVerticalFlip(p=0.5),
#     VT.RandomHorizontalFlipVideo(p=0.5),
#
#     # ))
#      ], 0.85),
#     VT.ResizeVideo((112,112), interpolation=2),
#     VT.CollectFrames(),
#     VT.PILVideoToTensor(),
#     VT.NormalizeVideo((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
#     #transforms.RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0, inplace=False)
#     ])
#
# transform_val_video = VT.Compose([VT.ResizeVideo((112,112), interpolation=2),
#                                     VT.PILVideoToTensor(),
#                                     VT.NormalizeVideo((0.4270, 0.2752, 0.2710), (0.2403, 0.2212, 0.2203)),
#                                     ])
# video_transform_train = Compose([
#     Resize(300),
#     RandomCrop(224),
#     #RandomResize( ratio=(0.75, 1.3333333333333333)),
#     RandomRotation(0),
#     RandomHorizontalFlip(),
#     #ColorJitter(0.25, 0.25, 0.25, 0.25),
#     Resize(224),
#     ClipToTensor(),
#     Normalize(mean=[0.4270, 0.2752, 0.2710], std=[0.2403, 0.2212, 0.2203]),
# ]
#                            )
#
# video_transform_val = Compose([
#     Resize(300),
#     CenterCrop(224),
#     ClipToTensor(),
#     Normalize(mean=[0.4270, 0.2752, 0.2710], std=[0.2403, 0.2212, 0.2203]),
# ]
#                            )