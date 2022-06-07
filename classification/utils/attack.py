#attack.py
import time
import pdb
import torch
import torch.nn as nn
import numpy as np
import foolbox as fb
from functools import partial
from torchvision import datasets, transforms

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #pdb.set_trace()
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.fc = nn.Linear(1024, 300)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(300, 300)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        out = x.reshape(x.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out



def load_mnist():
   
    transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('data', train=True, download=True,
                       transform=transform)
    test_set = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False)

    X_train, y_train, X_test, y_test = [], [], [], []
    for batch_idx, (img, label) in enumerate(train_loader):
        X_train.append(img)
        y_train.append(label)
        #print(batch_idx, img.shape, label.shape)
    for batch_idx, (img, label) in enumerate(test_loader):
        X_test.append(img)
        y_test.append(label)

    X_train = torch.cat(X_train).cuda()    
    y_train = torch.cat(y_train).cuda()
    X_test = torch.cat(X_test).cuda()
    y_test = torch.cat(y_test).cuda()    

    return X_train, y_train, X_test, y_test, train_loader, test_loader

def train(model, train_loader, name='lenet'):
    model.train()

    criterion = nn.CrossEntropyLoss()
    if name.lower() == 'lenet':
        optimizer = torch.optim.SGD(
                        model.parameters(), 
                        momentum=0.9, 
                        lr=0.01, 
                        weight_decay = 0.0001
                    )
    elif name.lower() == 'mlp':
        optimizer = torch.optim.Adam(
                        model.parameters(), 
                        lr=0.001, 
                        weight_decay = 0.0001
                    )
    
    epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    for epoch in range(epochs):
        for batch_idx, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()        
            optimizer.zero_grad()
        scheduler.step()

        print('epoch: {}, loss: {:.2f}'.format(epoch, loss.item()))
    
    return model

def test(model, test_loader):
    model.eval()
    correct, nelement = 0, 0
    for batch_idx, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        pred = model(X)    
        #pdb.set_trace()
        correct += (torch.argmax(pred, axis=1) == y).sum().item()
        nelement += pred.shape[0]

    acc = correct/nelement
    print('Acc: {:.2f}'.format(acc*100))

# X: feature (numpy) [0,255], sigma=1 means total change of 0 to 5
def add_noise(X, sigma=0, eps=0, noise_type=None): #is_uniform=False):

    # add noise
    X = torch.from_numpy(X).float()
    if noise_type == 'const':
        eps_prime = 255 * eps
        X_noisy = X + eps_prime
    elif noise_type == 'unif':
        eps_prime = 255 * eps
        X_noisy = X + torch.rand_like(X) * 2 * eps_prime - eps_prime
        #pass
    elif noise_type == 'normal':
        noise = torch.randn_like(X)
        max_val = noise.abs().max()
        noise = noise/max_val * (255 * eps)
        #pdb.set_trace()
        print(noise.abs().max())
        X_noisy = X + noise
    elif noise_type == 'sign':
        eps_prime = 255 * eps
        noise = eps_prime * (2 * torch.bernoulli(torch.ones_like(X)*0.5) - 1)        
        print(noise.abs().max())
        X_noisy = X + noise
    else:
        raise NotImplementedError #X_noisy = X + torch.randn_like(X) * sigma * 5 
    X_noisy = torch.clamp(X_noisy, min=0, max=255).int()

    # check the norm of (X - X_noisy), for each sample
    print((X_noisy - X).max())
    #noise_norm = torch.norm(X_noisy - X, dim=1)
    #print('noise_norm: ', noise_norm)

    X_noisy = X_noisy.cpu().numpy()

    return X_noisy



# X: feature (numpy), y: label (numpy)
def get_adv_ex(data='mnist', pretrained=False, eps=0.3, source='', target=''):#test_mlp_attack=0):

    if data.lower() != 'mnist':
        raise NotImplementedError

    # load model & data
    if source.lower() == 'lenet':
        model = LeNet5().cuda()
    elif source.lower() == 'mlp':
        model = MLP().cuda()
    else:
        raise NotImplementedError
    
    X_train, y_train, X_test, y_test, train_loader, test_loader = load_mnist()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # train a model & test (on clean data)
    if pretrained:
        PATH = f'{source}_mnist.pth'
        model.load_state_dict(torch.load(PATH))
    else:
        model = train(model, train_loader)
    #test(model, train_loader)
    test(model, test_loader)

    # generate adversarial sample 
    # make adv_test_loader using (X_adv_test, y_test)
    X_adv_test = attack_and_test(X_test, y_test, model, eps)
    adv_test_dataset = torch.utils.data.TensorDataset(X_adv_test, y_test)
    adv_test_loader = torch.utils.data.DataLoader(adv_test_dataset, batch_size=128, shuffle=False)
    test(model, adv_test_loader)

    # normalize & convert to numpy
    X_adv_test = (X_adv_test * 255).round().cpu().numpy().astype(int)    
    X_train = (X_train * 255).round().cpu().numpy().astype(int)
    print(X_train.max(), X_train.min())
    print(X_adv_test.max(), X_adv_test.min())
    y_train = y_train.cpu().numpy()
    y_test = y_test.cpu().numpy()

    # save
    eps_str = f'_{eps}'.replace('.', '_')
    source_str = f'_{source}'
    # with open(f'mnist_X_train.npy', 'wb') as f:
    #     np.save(f, X_train)
    # with open(f'mnist_y_train.npy', 'wb') as f:
    #     np.save(f, y_train)
    # with open(f'mnist_adv_X_test{source_str}{eps_str}.npy', 'wb') as f:
    #     np.save(f, X_adv_test)
    # with open(f'mnist_adv_y_test{source_str}{eps_str}.npy', 'wb') as f:
    #     np.save(f, y_test)


    if target != source:

        if target.lower() == 'lenet':
            target_model = LeNet5().cuda()
        elif target.lower() == 'mlp':
            target_model = MLP().cuda()
        else:
            raise NotImplementedError

        if pretrained:
            PATH = f'{target}_mnist.pth'
            target_model.load_state_dict(torch.load(PATH))
        else:    # train model // test on clean data & adversarial data
            model = train(target_model, train_loader, name=target.lower())
        test(target_model, test_loader)
        test(target_model, adv_test_loader)
        #exit()

    return X_train, y_train, X_adv_test, y_test





def attack_and_test(X, y, model, eps): #, Xmin, Xmax, eps):

    #X, y = X[:10], y[:10]

    attack_radius_fraction = eps # 0.3
    Xmin, Xmax = X.min(), X.max()
    eps = torch.tensor([(Xmax - Xmin) * attack_radius_fraction])
    print('Min: {}, Max: {}, Attack radius: {}'.format(Xmin, Xmax, eps))

     
    # load model
    model.eval()  
    bounds = (Xmin, Xmax)
    fmodel = fb.PyTorchModel(model, bounds=bounds)

    # check the clean accuracy
    cln_acc = fb.utils.accuracy(fmodel, X, y)
    print('clean accuracy: ', cln_acc)

    #pdb.set_trace()

    # attack model
    # cls_samples = {}
    # num_labels = 10
    # for clss in range(num_labels):
    #     idx = clean_pred != clss
    #     cls_samples[clss] = X[idx][0]  # Just pick the first example

    # for current_X, current_Y in zip(X,y):
    #     starting_points = []
    #     for y in current_Y:
    #         starting_points.append(cls_samples[int(y)])
    #     starting_points = torch.stack(starting_points, dim=0).to(device)
    #     advs, _, success = attack_fn(fmodel, current_X, current_Y, starting_points=starting_points, epsilons=epsilon_list)

    start_time = time.time()

    #attack = fb.attacks.BoundaryAttack()#(steps=25000) #(init_attack = fb.attacks.SaltAndPepperNoiseAttack) #LinfDeepFoolAttack #L2CarliniWagnerAttack
    attack = fb.attacks.LinfProjectedGradientDescentAttack()
    raw, clipped, is_adv = attack(fmodel, X, y, epsilons=eps)#, starting_points=X)
    rob_acc = 1 - is_adv.float().mean(axis=-1)
    print('robust accuracy: ', rob_acc)
    
    fin_time = time.time()
    print('running time: {:.6f}'.format(fin_time - start_time))


    return clipped[0]



# def transfer_attack(X, y, X_test, y_test, model='mlp'):
#     # input: X, y, X_test, y_test: array
#     # X, X_test: [0, 1, ..., 255]

#     if model.lower() != 'mlp':
#         raise NotImplementedError
    
#     # convert pixels in X to normalized tensor
#     # convert y to tensor
#     X, X_test = torch.from_numpy(X)/255, torch.from_numpy(X_test)/255
#     y, y_test = torch.from_numpy(y), torch.from_numpy(y_test)

#     # make dataloader
#     train_dataset = torch.utils.data.TensorDataset(X, y)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
#     test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

#     # load the model (MLP)


#     # train the model 

#     # test the model        

