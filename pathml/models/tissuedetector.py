import torch
import torch.nn as nn
from torchvision import transforms, models



def tissueDetector(modelStateDictPath='../pathml/pathml/models/deep-tissue-detector_densenet_state-dict.pt', architecture='densenet'):

    if architecture == 'inceptionv3':
        patch_size = 299
    else:
        patch_size = 224

    data_transforms = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # For now required PyTorch housekeeping
    if architecture == "resnet18":
        model_ft = models.resnet18(pretrained=False)
        model_ft.fc = nn.Linear(512, 3)
    elif architecture == "inceptionv3":
        model_ft = models.inception_v3(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(768, 3)
        model_ft.fc = nn.Linear(num_ftrs, 3)
    elif architecture == "vgg16":
        model_ft = models.vgg16(pretrained=False)
        model_ft.classifier[6] = nn.Linear(4096, 3)
    elif architecture == "vgg16_bn":
        model_ft = models.vgg16_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
    elif architecture == 'vgg19':
        model_ft = models.vgg19(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
    elif architecture == 'vgg19_bn':
        model_ft = models.vgg19_bn(pretrained=False)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, 3)
    elif architecture == "densenet":
        model_ft = models.densenet121(pretrained=False)
        model_ft.classifier = nn.Linear(1024, 3)
    elif architecture == "alexnet":
        model_ft = models.alexnet(pretrained=False)
        model_ft.classifier[6] = nn.Linear(4096, 3)
    elif architecture == "squeezenet":
        model_ft = models.squeezenet1_1(pretrained=False)
        model_ft.classifier[1] = nn.Conv2d(
            512, 3, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = 3
    else:
        raise ValueError('architecture not currently supported; supported architectures include resnet18, inceptionv3, vgg16, vgg16_bn, vgg19, vgg19_bn, densenet, alexnet, and squeezenet.')

    # It might struggle finding this dict here. Use full path to pathml
    model_ft.load_state_dict(torch.load(modelStateDictPath, map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model_ft = torch.nn.DataParallel(model_ft)
    model_ft.to(device).eval()

    return device, model_ft, data_transforms
