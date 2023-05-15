# -*- coding: utf-8 -*-

import os
import sys
import datetime
import pickle as pkl
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MNISTFiltered

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

np.random.seed(0)

channels = 1
feats = 28
C = 2

m = int(sys.argv[1]) #4, 8, 12, 16
alpha = float(sys.argv[2]) #0, 0.01, 0.1
sig = 0.1
batch_size = 'all'#sys.argv[3] #32 #'all'
low_data = str(sys.argv[3]) == 'True'
if 'all' not in str(batch_size):
    batch_size = int(batch_size)
label_noise = True
scaling = float(sys.argv[4]) #0.5, 1, 2, 3

if low_data:
    lr = 0.00005
    reduction_factor = 0.9*scaling*(feats*feats)/12000
else:
    lr = 0.0001
    reduction_factor = scaling*C*(m)*(feats*feats-1)/12000
if label_noise:
    thisFilename = 'binary_mnist_label_noise_low_data=' + str(low_data) + '_m=' + str(m) + '_a=' + str(alpha) + '_sc=' + str(scaling) # This is the general name of all related files
else:
    thisFilename = 'binary_mnist_low_data=' + str(low_data) + '_m=' + str(m) + '_a=' + str(alpha) + '_sc=' + str(scaling) # This is the general name of all related files
saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) 

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])

if low_data == True:
    n_epochs = 100
else:
    n_epochs = 100
    
val_ratio = 0.1
trainset = MNISTFiltered(root='./data', labels=[0,1], train=True,
                                        download=True, transform=transform)
train_size = len(trainset)
train_perm = torch.randperm(train_size)
valset = torch.utils.data.Subset(trainset,
                                   train_perm[0:int(val_ratio*reduction_factor*train_size)])
trainset = torch.utils.data.Subset(trainset,
                                   train_perm[int(val_ratio*reduction_factor*train_size):int(reduction_factor*train_size)])
val_size = len(valset)
train_size = len(trainset)
if batch_size == 'all':
    batch_size = train_size
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)
val_interval = 1
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

testset = MNISTFiltered(root='./data', labels=[0,1], train=False,
                                       download=True, transform=transform)
test_size = len(testset)
testset = torch.utils.data.Subset(testset,
                                   torch.randperm(test_size)[0:int(reduction_factor*test_size)])
test_size = len(testset)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('0','1')#,'2','3','4','5','6','7','8','9')

# Save info

hyperparameter_dict = {'nb_activations': str(m), 
                       'label_noise_flag': str(label_noise),
                       'alpha': str(alpha),
                       'sigma': str(sig),
                       'lr': str(lr),
                       'train_size': str(train_size),
                       'batch_size': str(batch_size),
                       'val_size': str(val_size),
                       'val_interval': str(val_interval),
                       'test_size': str(test_size),
                       'data dimension': str(channels*feats*feats),
                       'nb_params_per_activ': str(channels*feats*feats*C)
                       }

with open(os.path.join(saveDir,"hyperparameters.txt"), 'w') as f: 
    for key, value in hyperparameter_dict.items(): 
        f.write('%s:%s\n' % (key, value))

########################################################################
# Let us show some of the training images, for fun.

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


########################################################################
# 2. Define a Convolutional Neural Network
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Copy the neural network from the Neural Networks section before and modify it to
# take 3-channel images (instead of 1-channel images as it was defined).

import torch.nn as nn
import torch.nn.functional as F

def nonlinear(x, alpha):
    return torch.pow(x,3) + alpha*x

class Net(nn.Module):
    def __init__(self, m, alpha):
        super().__init__()
        self.m = m
        self.alpha = alpha
        fc = []
        for i in range(m):
            this_layer = nn.Linear(channels*feats*feats, C, bias=False, device=device)
            nn.init.ones_(this_layer.weight)
            fc.append(this_layer)
        self.fc = nn.ParameterList(fc)

    def forward(self, x):
        x = torch.reshape(x,shape=(x.shape[0],-1))
        for i, activ in enumerate(self.fc):
            y = activ(x)
            y = nonlinear(y,self.alpha)
            if i == 0:
                agg = y
            else:
                agg = agg + y
        return agg


net = Net(m, alpha)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr)#, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.
loss_vec = []
weights_list = []
test_accs = []
x_axis = []
acc_old = 0
save_labels = torch.empty(0, device=device)
save_x = torch.empty(0, device=device)

step_count = 0
x_axis = [step_count]
for epoch in range(n_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0)):
        if epoch == 0 and i == 0:
            weights = []
            for weight in list(net.parameters()):
                weights.append(weight.detach().clone())
            weights_list.append(weights)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        updated_labels = F.one_hot(labels, num_classes=C).float() 
        noisy_labels = updated_labels.clone()
        if label_noise:
            noisy_labels += torch.normal(0,
                                         sig*torch.ones(updated_labels.shape,device=device))
        if epoch == 0:
            save_x = torch.cat((save_x, inputs), dim=0)
            save_labels = torch.cat((save_labels, updated_labels), dim=0)
        loss = criterion(outputs, noisy_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % val_interval == val_interval-1:    # print every 100 mini-batches
            step_count = step_count + val_interval 
            x_axis.append(step_count)
            weights = []
            for weight in list(net.parameters()):
                weights.append(weight.detach().clone())
            weights_list.append(weights)
            loss_vec.append(running_loss/val_interval)
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / val_interval:.3f}')
            running_loss = 0.0
            total = 0
            correct = 0
            with torch.no_grad():
                for data in valloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_acc = correct/total
                print(f'[{epoch + 1}, {i + 1:5d}] accu: {val_acc:.3f}')
                if val_acc > acc_old:
                    save_net = net
                    acc_old = val_acc
                for data in testloader:
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    # calculate outputs by running images through the network
                    outputs = net(images)
                    # the class with the highest energy is what we choose as prediction
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                test_accs.append(100*correct/total)
                
    scheduler.step()
print('Finished Training')
weights = []
for weight in list(net.parameters()):
    weights.append(weight.detach().clone())
weights_list.append(weights)
x_axis.append(x_axis[-1]+val_interval)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Training Loss (MSE)')
ax1.plot(x_axis[1:-1], loss_vec, color=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Test accuracy (%)')  # we already handled the x-label with ax1
ax2.plot(x_axis[1:-1], test_accs, color=color)

fig.tight_layout()

#plt.show()

fig.savefig(os.path.join(saveDir,'train_test.png'))
fig.savefig(os.path.join(saveDir,'train_test.pdf'))

########################################################################
# Let's quickly save our trained model:

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

########################################################################
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_
# for more details on saving PyTorch models.
#
# 5. Test the network on the test data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
#imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

########################################################################
# Next, let's load back in our saved model (note: saving and re-loading the model
# wasn't necessary here, we only did it to illustrate how to do so):

net = Net(m, alpha)
net.load_state_dict(torch.load(PATH))

########################################################################
# Okay, now let us see what the neural network thinks these examples above are:

outputs = outputs.to(device)
images = images.to(device)
outputs = net(images)

########################################################################
# The outputs are energies for the 10 classes.
# The higher the energy for a class, the more the network
# thinks that the image is of the particular class.
# So, let's get the index of the highest energy:
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

########################################################################
# The results seem pretty good.
#
# Let us look at how the network performs on the whole dataset.

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')

########################################################################
# That looks way better than chance, which is 10% accuracy (randomly picking
# a class out of 10 classes).
# Seems like the network learnt something.
#
# Hmmm, what are the classes that performed well, and the classes that did
# not perform well:

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

save_dict = {'train_loss': loss_vec, 'test_acc': test_accs}
pkl.dump(save_dict,open(os.path.join(saveDir,'train_test.p'),'wb'))
save_dict = {'weights_list': weights_list, 'save_labels': save_labels, 'save_x': save_x}
pkl.dump(save_dict,open(os.path.join(saveDir,'saved_data.p'),'wb'))

# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%
del dataiter
# %%%%%%INVISIBLE_CODE_BLOCK%%%%%%

y = save_labels
x = save_x

y = y/m
if alpha == 0:
    y = torch.pow(y,1/3)
x = x.reshape(x.shape[0],-1)

y = y.cpu().numpy()
if alpha !=0 :
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i,j] = np.roots(np.array([1,0,alpha,-y[i,j]]))[0]

x = x.cpu().numpy()

regressor = LinearRegression(fit_intercept=False)

regressor.fit(x, y)

coefficients = regressor.coef_

y_reg = np.dot(x,np.transpose(coefficients))

figSize = (200/3*int(m/2),15)
plt.rcParams.update({'font.size': 8})

fig, axs = plt.subplots(2,int(m/2),sharex=True, sharey=True)

save_y_gnn = np.zeros((len(weights_list),m,x.shape[0],C)) # training steps, m, nb of samples, nb of classes
save_rrmse = []

for i in range(m):
    rrmse = []
    for j, weights in enumerate(weights_list):
        weight = weights[i].cpu().numpy()
        y_gnn = np.dot(x,np.transpose(weight))
        rrmse.append(100*np.sqrt(np.mean(np.linalg.norm(y_reg-y_gnn,axis=-1)**2)/
                                 np.sum(np.linalg.norm(y_reg,axis=-1)**2)))   
        save_y_gnn[j,i] = y_gnn
    rrmse = np.array(rrmse)
    save_rrmse.append(rrmse)
    print(rrmse[-1])
    axs[int(i % 2),int(i % int(m/2))].plot(x_axis, rrmse)
    
save_dict = {'rrmse': save_rrmse}
pkl.dump(save_dict,open(os.path.join(saveDir,'rrmse.p'),'wb'))

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
plt.xlabel("Training Steps")
plt.ylabel("RRMSE (%)")
    
fig.savefig(os.path.join(saveDir,'rrmse.png'))
fig.savefig(os.path.join(saveDir,'rrmse.pdf'))

# Rank

fig_rank, ax_rank = plt.subplots(1,1)

rank = []
eigs = np.zeros((m, save_y_gnn.shape[0]))
save_y_gnn = save_y_gnn[:,:,save_labels[:,0].cpu().numpy()==1,0]

for i in range(save_y_gnn.shape[0]):
    rank.append(np.linalg.matrix_rank(np.reshape(save_y_gnn[i],(m,-1)),tol=0.01))
    #_, L, _ = np.linalg.svd(np.reshape(save_y_gnn[i],(m,-1)))
    #eigs[:,i] = L[0:m]
    aux_tensor = torch.tensor(np.reshape(save_y_gnn[i],(m,-1)))
    aux_tensor = aux_tensor.to_sparse()
    U, L,_ = torch.svd_lowrank(aux_tensor,q=m)
    eigs[:,i] = L.cpu().numpy()
    U = U.cpu().numpy()
    
save_dict = {'eigs': eigs}
pkl.dump(save_dict,open(os.path.join(saveDir,'eigs.p'),'wb'))

for i in range(m):
    ax_rank.plot(x_axis, eigs[i]/eigs[0], label='lam'+str(i+1))
#ax_rank.legend()
plt.xlabel("Training Steps")
plt.ylabel("Singular Values")
    
fig_rank.savefig(os.path.join(saveDir,'rank.png'))
fig_rank.savefig(os.path.join(saveDir,'rank.pdf'))