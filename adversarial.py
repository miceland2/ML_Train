import torch
import torch.nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from torchvision import transforms
import numpy as np
import requests, io
import matplotlib.pyplot as plt
from torch.autograd import Variable

def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
model.eval();

input_image = Image.open('peppers.jpeg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

image_tensor = preprocess(input_image)
image_tensor = image_tensor.unsqueeze(0)


img_variable = Variable(image_tensor, requires_grad=True)
output = model(img_variable)

label_idx = torch.max(output.data, 1)[1][0]
print(label_idx.item())

labels_link = "https://savan77.github.io/blog/labels.json"    
labels_json = requests.get(labels_link).json()
labels = {int(idx):label for idx, label in labels_json.items()}
x_pred = labels[label_idx.item()]
print(x_pred)

output_probs = F.softmax(output, dim=1)
x_pred_prob =  torch.max(output_probs.data, 1)[0][0] * 100
print(x_pred_prob.item())

y_fool = 345
y_target = Variable(torch.LongTensor([y_fool]), requires_grad=False)
print(y_target)

epsilon = 0.25
num_steps = 5
alpha = 0.025

img_variable.data = image_tensor

for i in range(num_steps):
    zero_gradients(img_variable)
    output = model(img_variable)
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, y_target)
    loss_cal.backward()
    x_grad = alpha * torch.sign(img_variable.grad.data)
    adv_temp = img_variable.data - x_grad
    total_grad = adv_temp - image_tensor
    total_grad = torch.clamp(total_grad, -epsilon, epsilon)
    x_adv = image_tensor + total_grad
    img_variable.data = x_adv

#img_variable = transforms.functional.hflip(img_variable)
#img_variable = transforms.functional.vflip(img_variable)
#transform = transforms.Grayscale(3)
#img_variable = transform(img_variable)
#transform = transforms.CenterCrop(185)
#img_variable = transform(img_variable)
#img_variable = transforms.functional.rotate(img_variable, 330)
#transform = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomCrop(32, padding=4)]), p=0.5)
#transform = transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=(19, 23), sigma=(20, 25))]))
#transform = transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=(-180,180))]))
#transform = transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(hue=0.3)]), p=0.5)
#transform = transforms.RandomApply(torch.nn.ModuleList([transforms.Grayscale(3)]))
#transform = transforms.RandomApply([gauss_noise_tensor])
#transform = transforms.RandomErasing()
#img_variable = preprocess(img_variable)


output_adv = model(img_variable)
x_adv_pred = labels[torch.max(output_adv.data, 1)[1][0].item()]
output_adv_probs = F.softmax(output_adv, dim=1)
x_adv_pred_prob = (torch.max(output_adv_probs.data, 1)[0][0]) * 100
#print()
#print(x_adv_pred)
#print(adv_pred_prob.item())

def visualize(x, x_adv, x_grad, epsilon, clean_pred, adv_pred, clean_prob, adv_prob):
    
    x = x.squeeze(0)     #remove batch dimension # B X C H X W ==> C X H X W
    x = x.mul(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1)).add(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1)).detach().numpy()#reverse of normalization op- "unnormalize"
    x = np.transpose( x , (1,2,0))   # C X H X W  ==>   H X W X C
    x = np.clip(x, 0, 1)
    
    x_adv = x_adv.squeeze(0)
    x_adv = x_adv.mul(torch.FloatTensor([0.229, 0.224, 0.225]).view(3,1,1)).add(torch.FloatTensor([0.485, 0.456, 0.406]).view(3,1,1)).detach().numpy()#reverse of normalization op
    x_adv = np.transpose( x_adv , (1,2,0))   # C X H X W  ==>   H X W X C
    x_adv = np.clip(x_adv, 0, 1)
    
    x_grad = x_grad.squeeze(0).numpy()
    x_grad = np.transpose(x_grad, (1,2,0))
    x_grad = np.clip(x_grad, 0, 1)
    
    figure, ax = plt.subplots(1,3, figsize=(18,8))
    ax[0].imshow(x)
    ax[0].set_title('Clean Example', fontsize=20)
    
    
    ax[1].imshow(x_grad)
    ax[1].set_title('Perturbation', fontsize=20)
    ax[1].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    
    ax[2].imshow(x_adv)
    ax[2].set_title('Adversarial Example', fontsize=20)
    
    ax[0].axis('off')
    ax[2].axis('off')

    ax[0].text(1.1,0.5, "+{}*".format(round(epsilon,3)), size=15, ha="center", 
             transform=ax[0].transAxes)
    
    ax[0].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(clean_pred, clean_prob), size=15, ha="center", 
         transform=ax[0].transAxes)
    
    ax[1].text(1.1,0.5, " = ", size=15, ha="center", transform=ax[1].transAxes)

    ax[2].text(0.5,-0.13, "Prediction: {}\n Probability: {}".format(adv_pred, adv_prob), size=15, ha="center", 
         transform=ax[2].transAxes)
    

    plt.show()
    
visualize(image_tensor, img_variable, x_grad, 0.02, x_pred, x_adv_pred, x_pred_prob, x_adv_pred_prob)
