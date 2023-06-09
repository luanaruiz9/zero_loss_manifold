# -*- coding: utf-8 -*-

import os
import sys
import datetime
import pickle as pkl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import SyntheticData

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


##############################################################################
## Neural network architecture


def nonlinear(x, alpha):
    return torch.pow(x,3) + alpha*x

class Net(nn.Module):
    def __init__(self, m, alpha, ortho=False):
        super().__init__()
        self.m = m
        self.alpha = alpha
        fc = []
        for i in range(m):
            this_layer = nn.Linear(feats, 1, bias=False, device=device)
            if ortho:
                nn.init.zeros_(this_layer.weight)
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


##############################################################################
## Hyperparameters of the simulation

np.random.seed(0)
n_realizations = 1 # number of runs

feats = 128 # data dimension
m_teacher = 5 # nb activations of the teacher network
m = 100 #int(sys.argv[1]) #4, 8, 12, 16 # nb activations student network
alpha = 0.1#float(sys.argv[2]) #0, 0.01, 0.1 # alpha param. of nonlinearity

batch_size = 'all'#sys.argv[3] #32 #'all'
low_data = True#str(sys.argv[1]) == 'True' # if True, overparametrized; o.w., 
                                            # underparametrized
if 'all' not in str(batch_size):
    batch_size = int(batch_size)
    
label_noise = True # whether to use SGD or label noise SGD
sig = 0.05 # variance of label noise

scaling = 0.7#float(sys.argv[4]) #0.5, 1, 2, 3 # scaling factor for the 
                                                # number of samples. 
                                                # If low_data=True, it multiplies 
                                                # the current number of samples; 
                                                # o.w., it divides it.

# Setting learning rate and reduction factor for the number of samples in either case
if low_data:
    lr = 0.000001
    reduction_factor = 0.9*scaling*(feats)/10000
else:
    lr = 0.000001
    reduction_factor = (1/scaling)*(m)*(feats-1)/10000
    
    
##############################################################################
## File handling
 
if label_noise:
    thisFilename = 'synthetic_label_noise_low_data=' + str(low_data) + '_m=' + str(m) + '_a=' + str(alpha) + '_sc=' + str(scaling) # This is the general name of all related files
else:
    thisFilename = 'synthetic_low_data=' + str(low_data) + '_m=' + str(m) + '_a=' + str(alpha) + '_sc=' + str(scaling) # This is the general name of all related files
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


##############################################################################
## Training and test settings

# Nb epochs in either case
if low_data == True:
    n_epochs = 20000
else:
    n_epochs = 50000
    
val_ratio = 0.1 # ratio of training samples to use for validation
old_train_size = 10000
old_test_size = 2000
old_trainset = SyntheticData(old_train_size, feats, mu=0, sigma=0.5)
old_testset = SyntheticData(old_test_size, feats, mu=0, sigma=0.5)

# Rescale train and test size according to reduction factor
train_size = int(reduction_factor*old_train_size)
test_size = int(reduction_factor*len(old_testset))

# Define and apply teacher network to generate labels
net_teacher = Net(m_teacher, alpha, ortho=False)

trainloader = torch.utils.data.DataLoader(old_trainset, batch_size=old_train_size,
                                          shuffle=False, num_workers=0)
dataiter = iter(trainloader)
x, _ = next(dataiter)
x = x.to(device)
with torch.no_grad():
    y = net_teacher(x)
    old_trainset.change_labels(torch.tensor(y))
    
testloader = torch.utils.data.DataLoader(old_testset, batch_size=old_test_size,
                                          shuffle=False, num_workers=0)
dataiter = iter(testloader)
x, _ = next(dataiter)
x = x.to(device)
with torch.no_grad():
    y = net_teacher(x)
    old_testset.change_labels(torch.tensor(y))
    
##############################################################################
## Saving hyperparameters and training details

hyperparameter_dict = {'nb_activations': str(m), 
                       'label_noise_flag': str(label_noise),
                       'alpha': str(alpha),
                       'sigma': str(sig),
                       'lr': str(lr),
                       'train_size': str(train_size),
                       'batch_size': str(batch_size),
                       'test_size': str(test_size),
                       'data dimension': str(feats),
                       'nb_params_per_activ': str(feats)
                       }

with open(os.path.join(saveDir,"hyperparameters.txt"), 'w') as f: 
    for key, value in hyperparameter_dict.items(): 
        f.write('%s:%s\n' % (key, value))
        
##############################################################################
## Training    

all_loss_vecs = [] # save training losses for all realizations
all_test_accs = [] # save test losses for all realizations
all_eigs = [] # save singular values for all realizations
all_rank = [] # save rank for all realizations
all_trace = [] # save trace for all realizations

for r in range(n_realizations):

    # Random realization of the data    
    train_perm = torch.randperm(old_train_size)[0:train_size]
    valset = torch.utils.data.Subset(old_trainset,
                                       train_perm[0:int(val_ratio*train_size)])
    trainset = torch.utils.data.Subset(old_trainset,
                                       train_perm[int(val_ratio*train_size):train_size])
    train_size = len(trainset)
    val_size = len(valset)
    
    if batch_size == 'all':
        batch_size = train_size
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    val_interval = 2
    valloader = torch.utils.data.DataLoader(valset, batch_size=val_size,
                                             shuffle=False, num_workers=0)
    testset = torch.utils.data.Subset(old_testset,
                                       torch.randperm(old_test_size)[0:test_size])
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_size,
                                             shuffle=False, num_workers=0)
    
    ########################################################################
    # 2. Define a Convolutional Neural Network
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Copy the neural network from the Neural Networks section before and modify it to
    # take 3-channel images (instead of 1-channel images as it was defined).
    
    net = Net(m, alpha, ortho=False)
    
    ########################################################################
    # 3. Define a Loss function and optimizer
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Let's use a Classification Cross-Entropy loss and SGD with momentum.
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr)#, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)
    
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
    if r == 0:
        x_axis = []
    acc_old = 100000000000
    save_labels = torch.empty(0, device=device)
    save_x = torch.empty(0, device=device)
    
    if r == 0:
        step_count = 0
        x_axis = [step_count]
    running_loss = 0
    for epoch in range(n_epochs):  # loop over the dataset multiple times
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
            updated_labels = labels#F.one_hot(labels, num_classes=C).float() 
            noisy_labels = updated_labels.clone()
            if label_noise:
                noisy_labels += torch.normal(0,sig*torch.ones(updated_labels.shape,
                                                              device=device))
            if epoch == 0:
                save_x = torch.cat((save_x, inputs), dim=0)
                save_labels = torch.cat((save_labels, updated_labels), dim=0)
            loss = criterion(outputs, noisy_labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if epoch % val_interval == val_interval-1 or (epoch == 0 and i == 0):    # print every 100 mini-batches
                if r == 0:
                    step_count = step_count + val_interval 
                    x_axis.append(step_count)
                weights = []
                for weight in list(net.parameters()):
                    weights.append(weight.detach().clone())
                weights_list.append(weights)
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / val_interval:.3f}')
                if epoch == 0 and i == 0:
                    loss_vec.append(running_loss)
                else:
                    loss_vec.append(running_loss/val_interval)
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
                        predicted = outputs
                    val_acc = F.mse_loss(labels,predicted)
                    print(f'[{epoch + 1}, {i + 1:5d}] accu: {val_acc:.3f}')
                    if val_acc < acc_old:
                        save_net = net
                        acc_old = val_acc
                    for data in testloader:
                        images, labels = data
                        images = images.to(device)
                        labels = labels.to(device)
                        # calculate outputs by running images through the network
                        outputs = net(images)
                        # the class with the highest energy is what we choose as prediction
                        predicted = outputs
                        test_acc = F.mse_loss(labels,predicted)
                    test_accs.append(test_acc.cpu().numpy())
                    
        scheduler.step()
    print('Finished Training')
    
    all_loss_vecs.append(loss_vec)
    all_test_accs.append(test_accs)
    
    weights = []
    for weight in list(net.parameters()):
        weights.append(weight.detach().clone())
    weights_list.append(weights)
    
    
    ##############################################################################
    ## Plot training and test loss for this realization
    
    fig, ax1 = plt.subplots()
    
    color = 'tab:red'
    ax1.set_xlabel('Training Steps')
    ax1.set_xscale('log')
    ax1.set_ylabel('Loss (MSE)')
    loss_vec = loss_vec
    ax1.plot(x_axis[1:], loss_vec, color=color, label='training')
    color = 'tab:blue'
    ax1.plot(x_axis[1:], test_accs, color=color, label='test')
    plt.legend()
    fig.tight_layout()
    #plt.show()
    
    fig.savefig(os.path.join(saveDir,'train_test_' + str(r) + '.png'))
    fig.savefig(os.path.join(saveDir,'train_test_' + str(r) + '.pdf'))
    
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
    
    ########################################################################
    # Next, let's load back in our saved model (note: saving and re-loading the model
    # wasn't necessary here, we only did it to illustrate how to do so):
    
    net = Net(m, alpha, ortho=False)
    net.load_state_dict(torch.load(PATH))
    
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
            predicted = outputs
            test_acc = F.mse_loss(labels,predicted)
    
    print(f'Loss of the network on the test images: {test_acc}')
    
    # Save training and test loss vectors, as well as the networks weights for all epochs
    save_dict = {'train_loss': loss_vec, 'test_acc': test_accs}
    pkl.dump(save_dict,open(os.path.join(saveDir,'train_test_' + str(r) + '.p'),'wb'))
    save_dict = {'weights_list': weights_list, 'save_labels': save_labels, 'save_x': save_x}
    pkl.dump(save_dict,open(os.path.join(saveDir,'saved_data_' + 'r' + '.p'),'wb'))
    
    ##############################################################################
    ## Regression
    
    y = save_labels
    x = save_x
    
    """
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
    """
    
    ##############################################################################
    ## Plot properties of the Hessian and rrmse between NN and regression weights
    ## for this realization
    
    save_y_gnn = np.zeros((len(weights_list),m,x.shape[0])) # training steps, m, nb of samples, nb of classes
    #save_rrmse = []
    
    for i in range(m):
        #rrmse = []
        for j, weights in enumerate(weights_list):
            weight = weights[i].cpu().numpy()
            y_gnn = np.dot(x.cpu().numpy(),np.transpose(weight))
            #rrmse.append(100*np.sqrt(np.mean(np.linalg.norm(y_reg-y_gnn,axis=-1)**2)/
            #                         np.sum(np.linalg.norm(y_reg,axis=-1)**2)))   
            save_y_gnn[j,i] = y_gnn.squeeze()
        #rrmse = np.array(rrmse)
        #save_rrmse.append(rrmse)
        #print(rrmse[-1])
        #axs[int(i % 2),int(i % int(m/2))].plot(x_axis, rrmse)
        
    #save_dict = {'rrmse': save_rrmse}
    #pkl.dump(save_dict,open(os.path.join(saveDir,'rrmse_' + str(r) + '.p'),'wb'))
    
    # SVs, rank, trace
    
    fig_rank, ax_rank = plt.subplots(1,3, figsize=(45,15), sharex=True)
    
    eigs = np.zeros((m, save_y_gnn.shape[0]))
    rank = np.zeros(save_y_gnn.shape[0])
    trace = np.zeros(save_y_gnn.shape[0])
    
    for i in range(save_y_gnn.shape[0]):
        #aux_tensor = torch.tensor(np.reshape(save_y_gnn[i],(m,-1)))
        #aux_tensor = aux_tensor.to_sparse()
        #_, L,_ = torch.svd_lowrank(aux_tensor,q=15)
        #eigs[0:L.shape[0],i] = L.cpu().numpy()
        aux = np.reshape(save_y_gnn[i],(m,-1))
        _, L, _ = np.linalg.svd(aux)
        eigs[0:L.shape[0],i] = L
        rank[i] = np.linalg.matrix_rank(aux)
        trace[i] = np.sum(L)
        
    all_eigs.append(eigs)
    all_rank.append(rank)
    all_trace.append(trace)
        
    save_dict = {'eigs': eigs, 'rank': rank, 'trace': trace}
    pkl.dump(save_dict,open(os.path.join(saveDir,'eigs_' + str(r) + '.p'),'wb'))
    
    plt.xscale('log')
    for i in range(0,m):
        ax_rank[0].plot(x_axis, eigs[i,0:-1]/eigs[0,0:-1], label='lam'+str(i+1))
    ax_rank[0].set_title("SVs")
    ax_rank[1].plot(x_axis, rank[0:-1])
    ax_rank[1].set_title("Rank")
    ax_rank[2].plot(x_axis, trace[0:-1])
    ax_rank[2].set_title("Trace")
    #ax_rank.legend()
        
    fig_rank.savefig(os.path.join(saveDir,'rank_' + str(r) + '.png'))
    fig_rank.savefig(os.path.join(saveDir,'rank_' + str(r) + '.pdf'))


##############################################################################
## Plot mean and standard deviations over all realizations (outdated)

"""

all_loss_vecs = np.array(all_loss_vecs)
all_test_accs = np.array(all_test_accs)

mean_loss = np.mean(all_loss_vecs, axis=0)
std_loss =  0.5*np.std(all_loss_vecs, axis=0)

mean_acc = np.mean(all_test_accs, axis=0)
std_acc =  0.5*np.std(all_test_accs, axis=0)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Loss (MSE)')
ax1.set_xscale('log')
ax1.fill_between(x_axis[1:], mean_loss-std_loss, 
                 mean_loss+std_loss,
                 color=color, alpha=0.1)
ax1.plot(x_axis[1:], mean_loss, color=color, label='training')

#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
#ax2.set_ylabel('Test Loss (MSE)')  # we already handled the x-label with ax1
ax1.fill_between(x_axis[1:], mean_acc-std_acc,
                 mean_acc + std_acc,
                 color=color, alpha=0.1)
ax1.plot(x_axis[1:], mean_acc, color=color, label='test')
plt.legend()
fig.tight_layout()

#plt.show()

fig.savefig(os.path.join(saveDir,'mean_train_test.png'))
fig.savefig(os.path.join(saveDir,'mean_train_test.pdf'))

# SVs

all_eigs = np.vstack(all_eigs)
all_eigs = np.reshape(all_eigs,(n_realizations,m,-1))
all_eigs = all_eigs/np.tile(np.expand_dims(all_eigs[:,0,:],1),(1,m,1))
mean_eigs = np.mean(all_eigs,axis=0)
std_eigs = 0.5*np.std(all_eigs,axis=0)

fig_rank, ax_rank = plt.subplots(1,1)

cmap = plt.cm.get_cmap('Spectral')
color_code = np.linspace(0,1,m) 

for i in range(m):
    ax_rank.fill_between(x_axis, mean_eigs[i,0:-1]-std_eigs[i,0:-1], mean_eigs[i,0:-1]+std_eigs[i,0:-1],
                         color=cmap(color_code[i]),alpha=0.1)
    ax_rank.plot(x_axis, mean_eigs[i,0:-1], label='lam'+str(i+1),color=cmap(color_code[i]))
#ax_rank.legend()
ax_rank.set_xscale('log')
plt.xlabel("Training Steps")
plt.ylabel("Singular Values")
    
fig_rank.savefig(os.path.join(saveDir,'mean_rank.png'))
fig_rank.savefig(os.path.join(saveDir,'mean_rank.pdf'))

"""