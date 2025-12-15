#!/usr/bin/env python
# coding: utf-8

# In[65]:


import torch
import torch.nn as nn
from torchvision import transforms,datasets
import os
import matplotlib.pyplot as plt
import zipfile


# In[66]:


import torch
print(torch.__version__)          # Shows PyTorch version and CUDA version if available
print(torch.version.cuda)         # Shows CUDA version PyTorch was compiled with
print(torch.cuda.is_available())  # True if GPU with CUDA is usable


# In[67]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# In[68]:


with zipfile.ZipFile("archive.zip", "r") as zip_ref:
    zip_ref.extractall("./dataset")


# In[69]:


for dir_path,dir_name,file_name in os.walk("dataset"):
    print(f"dir: {dir_path.split('\\')[-1]} -> {len(file_name)}")


# ### data dir

# In[70]:


train_dir = "dataset/train"
test_dir = "dataset/test"


# In[71]:


from PIL import Image
import random
import pathlib
img_path = list(pathlib.Path(train_dir).glob("*/*.jpg"))
random_img_path = random.sample(img_path,k=1)[0]
print(random_img_path.parent.stem)
random_img = Image.open(random_img_path)
random_img


# In[72]:


train_img_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_img_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# In[73]:


def plt_random_img():
    img_path = list(pathlib.Path(train_dir).glob("*/*.jpg"))
    random_img_path = random.sample(img_path,k=1)[0]
    random_img = Image.open(random_img_path).convert('RGB')
    img_label = random_img_path.parent.stem
    transformed_img = train_img_transformer(random_img)

    fig,ax = plt.subplots(nrows=1, ncols=2)

    ax[0].imshow(transformed_img.permute(1,2,0),cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(random_img)
    ax[1].axis('off')
    plt.show()
plt_random_img()


# # image dataset

# In[74]:


train_dataset = datasets.ImageFolder(train_dir,transform=train_img_transformer)
test_dataset = datasets.ImageFolder(test_dir,transform=test_img_transformer)


# In[75]:


class_names = train_dataset.classes
class_with_idx = train_dataset.class_to_idx


# In[76]:


from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True,num_workers=8,pin_memory=True)
test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=8,pin_memory=True)


# In[77]:


len(train_dataloader),len(test_dataloader)


# ### alexNet

# In[78]:


# class AlexNet(nn.Module):
#     def __init__(self):
#         super(AlexNet, self).__init__()


# In[79]:


def accuracy_func(y_true, y_pred):
    correct = (y_pred == y_true).sum().item()
    accuracy = correct / y_true.size(0)
    return accuracy * 100


# In[80]:


from torchvision.models import alexnet
from tqdm.auto import tqdm
def training_step(model,traing_dataloader,loss_fn,optimizer,accuracy_fn):
    batch_loss = 0
    batch_accuracy = 0
    model.train()
    for batch,(x,y) in enumerate(traing_dataloader):
        y_logits = model(x.to(device))
        loss = loss_fn(y_logits,y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_prob = torch.softmax(y_logits,dim=1)
        y_pred = torch.argmax(y_prob,dim=1)
        accuracy = accuracy_fn(y.to(device),y_pred)
        batch_accuracy += accuracy
        batch_loss += loss.item()

    batch_accuracy = batch_accuracy / len(traing_dataloader)
    batch_loss = batch_loss / len(traing_dataloader)
    print(f"Batch Train loss: {batch_loss:.2f} -- Batch Accuracy: {batch_accuracy:.2f}")
    return batch_loss,batch_accuracy


# In[81]:


def testing_step(model,test_dataloader,loss_fn,accuracy_fn):
    batch_test_loss = 0
    batch_test_accuracy = 0
    model.eval()
    with torch.inference_mode():
        for x,y in test_dataloader:
            y_logits = model(x.to(device))
            batch_test_loss += loss_fn(y_logits,y.to(device)).item()

            y_prob = torch.softmax(y_logits,dim=1)
            y_pred = torch.argmax(y_prob,dim=1)
            batch_test_accuracy += accuracy_fn(y.to(device),y_pred)

        batch_test_loss = batch_test_loss / len(test_dataloader)
        batch_test_accuracy = batch_test_accuracy / len(test_dataloader)
        print(f"batch test loss {batch_test_loss:.2f} -- batch test accuracy: {batch_test_accuracy:.2f}")
        return batch_test_loss,batch_test_accuracy


# In[82]:


check_point = torch.load('models/alex_model_v2_with_freeze_convo.pt',weights_only=True)
alex_model = alexnet(pretrained=False)
alex_model.classifier[6] = nn.Linear(4096, len(class_names))
for param in alex_model.features.parameters():
    param.requires_grad = True

alex_model.load_state_dict(check_point["model_state"])

alex_model.to(device)

loss_function = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(alex_model.classifier[6].parameters(),lr=0.0005)
optim.load_state_dict(check_point["optimizer_state"])


# In[87]:


model_result = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "epoch": 0
}


# In[ ]:


EPOCHS = 10

for epoch in tqdm(range(EPOCHS)):

    train_loss,train_acc = training_step(alex_model,train_dataloader,loss_function,optim,accuracy_func)
    test_loss,test_acc = testing_step(alex_model,test_dataloader,loss_function,accuracy_func)

    model_result["test_loss"].append(test_loss)
    model_result["train_loss"].append(train_loss)
    model_result["train_accuracy"].append(train_acc)
    model_result["test_accuracy"].append(test_acc)
    model_result["epoch"] += 1


# In[90]:


image_path = list(pathlib.Path(train_dir).glob("*/*.jpg"))
random_img_path = random.sample(image_path,4)
img_set = 0

plt.figure(figsize=(10,10))
for image_path in random_img_path:
    img_set +=1
    random_img = Image.open(image_path).convert("RGB")
    transfomed_random_img = test_img_transformer(random_img)
    alex_model.eval()
    with torch.inference_mode():
        y_logits = alex_model(transfomed_random_img.unsqueeze(0).to(device))

        y_prob = torch.softmax(y_logits,dim=1)
        y_pred = torch.argmax(y_prob,dim=1)

    # plt.subplots(nrows=2, ncols=2)
    plt.subplot(2,2,img_set)
    plt.imshow(random_img)

    if class_names[y_pred] == image_path.parent.stem:
        plt.title(f"{class_names[y_pred]} | {image_path.parent.stem}",color="green")
    else:
        plt.title(f"{class_names[y_pred]} | {image_path.parent.stem}",color="red")

    plt.axis('off')
plt.show()


# In[91]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(model_result["train_loss"],label="Train Loss")
plt.plot(model_result["test_loss"],label="Test Loss")
plt.legend()

plt.subplot(2,2,2)
plt.plot(model_result["test_accuracy"],label = "test accuracy")
plt.plot(model_result["train_accuracy"],label = "train accuracy")
plt.legend()
plt.show()


# In[58]:


torch.save({
    "model_state":alex_model.state_dict(),
    "optimizer_state":optim.state_dict(),
    "epoch":model_result["epoch"]
},"models/alex_model_v3_without_freeze_convo.pt")


# In[63]:




