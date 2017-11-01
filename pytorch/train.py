import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch.utils as utils
from pytorch.data import ImageDirectoryDataset, MasterData
from pytorch.model import Net

BASE_FOLDER = 'data'
IMG_PATH = 'data/images'



def train(batch_size=64, epochs=1):

    transformations = transforms.Compose([ transforms.ToTensor()])
    ds = MasterData()

    X_train, X_test, y_train, y_test = ds.train_val_split(val_size=0.2)
    ds_train = ImageDirectoryDataset(X_train, y_train, transform=transformations)
    ds_val = ImageDirectoryDataset(X_test, y_test, transform=transformations)

    pos_sample = np.sum(y_train)
    neg_sample = len(y_train) - pos_sample
    class_sample_count = [pos_sample*1.0, neg_sample*1.0]
    print(class_sample_count)
    weights = 1 / torch.DoubleTensor(np.array(class_sample_count))
    print(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batch_size)

    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              shuffle=None,
                              num_workers=1,
                              sampler=sampler
                              # pin_memory=True # CUDA only
                              )
    val_loader = DataLoader(ds_val,
                            batch_size=batch_size,
                            num_workers=1,
                            sampler=sampler
                            # pin_memory=True # CUDA only
                            )
    if torch.cuda.is_available():
        model = Net().cuda()
    else:
        model = Net()

    run_id = utils.generate_run_id()


    for epoch in range(epochs):
        # running_loss = AverageMeter()
        # running_accuracy = AverageMeter()
        # val_loss_meter = AverageMeter()
        # val_acc_meter = AverageMeter()
        # pbar = tqdm_notebook(train_loader, total=len(train_loader))

        train_model(model=model, epoch=epoch, train_loader=train_loader)
        torch.save(model.state_dict(), 'generated/model/{}_{}.pth.tar'.format(run_id, epoch))


def train_model(model, epoch, train_loader):
    model.train()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(async=True), target.cuda(async=True) # On GPU
        print("Starting batch_idx: {} and target : {}".format(batch_idx,target))
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data[0]))


if __name__ == '__main__':
    train()
