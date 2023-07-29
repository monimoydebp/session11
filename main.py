from torchsummary import summary
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
import cv2
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from data_transformations.data_transform_cifar10_resnet import get_train_transform, get_test_transform
from data_loaders.cifar10_data_loader import get_train_loader, get_test_loader, get_classes
from models.resnet18 import ResNet18
from utils.train_test_utils import train,test
from utils.accuracy_utils import get_test_accuracy,get_accuracy_per_class
from utils.plot_metrics_utils import plot_loss_accuracy
from utils.misclassified_image_utils import  display_misclassfied_ciphar10_images
from utils.gradcam_utils import process_grad

def train_test_loader():
    transform_train = get_train_transform()
    transform_test = get_test_transform()
    trainloader = get_train_loader(256, transform_train)
    testloader = get_test_loader(256, transform_test)
    classes = get_classes()
    return trainloader, testloader, classes

def get_model_parameters() :   
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model =  ResNet18().to(device)
    return model, device

def perform_training(model, device, trainloader, testloader,  train_losses, test_losses,train_acc,test_acc, PATH, epochs=40, lr=0.01, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=5, verbose=True)
	
    torch.save(model.state_dict(), PATH)
    best_test_accuracy = 0.0

    EPOCHS = epochs
    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        train(model, device, trainloader, optimizer, epoch, train_losses,train_acc )
        test(model, device, testloader, test_losses, test_acc)
        t_acc = test_acc[-1]
        if t_acc > best_test_accuracy:
            print("Test Accuracy: " + str(t_acc) + " has increased. Saving the model")
            best_test_accuracy = t_acc
            torch.save(model.state_dict(), PATH)
        scheduler.step(t_acc)
    return train_losses, test_losses,train_acc,test_acc

def incorrect_image(model, device, test_loader, incorrect_image_list, predicted_label_list, correct_label_list):
    for (i, [data, target]) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).squeeze(1)         
        idxs_mask = (pred !=  target).view(-1)
        img_nm = data[idxs_mask].cpu().numpy()
        img_nm = img_nm.reshape(img_nm.shape[0], 3, 32, 32)
        if img_nm.shape[0] > 0:
            img_list = [img_nm[i] for i in range(img_nm.shape[0])]
            incorrect_image_list.extend(img_list)
            predicted_label_list.extend(pred[idxs_mask].detach().cpu().numpy())
            correct_label_list.extend(target[idxs_mask].detach().cpu().numpy())
    return incorrect_image_list, predicted_label_list, correct_label_list