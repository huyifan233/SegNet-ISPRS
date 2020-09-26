import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from fl_model import SegNet
from isprs_dataset_win import ISPRS_dataset



WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
BATCH_SIZE = 10
LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target,weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size


def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=5):
    train_ids = ['4_10', '4_11', '5_10', '5_11']
    train_set = ISPRS_dataset(train_ids, cache=CACHE)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    weights = weights.cuda()
    net.cuda()
    criterion = nn.NLLLoss(weight=weights)
    iter_ = 0

    for e in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step()
        net.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data.cuda()), Variable(target.cuda())
            optimizer.zero_grad()
            output = net(data)
            loss = CrossEntropy2d(output, target, weight=None)
            loss.backward()
            optimizer.step()

            #             losses[iter_] = loss.data[0]
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100):iter_])

            if iter_ % 100 == 0:
                # clear_output()
                rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {}'.format(
                    e, epochs, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.item(), accuracy(pred, gt)))
                # plt.plot(mean_losses[:iter_]) and plt.show()
                # fig = plt.figure()
                # fig.add_subplot(131)
                # plt.imshow(rgb)
                # plt.title('RGB')
                # fig.add_subplot(132)
                # plt.imshow(convert_to_color(gt))
                # plt.title('Ground truth')
                # fig.add_subplot(133)
                # plt.title('Prediction')
                # plt.imshow(convert_to_color(pred))
                # plt.show()
            iter_ += 1

            del (data, target, loss)

        # if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            # acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
            # torch.save(net.state_dict(), './segnet256_epoch{}_{}'.format(e, acc))
    torch.save(net.state_dict(), './segnet_final')


def main():
    base_lr = 0.01
    net = SegNet()
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        if '_D' in key:
            # Decoder weights are trained at the nominal learning rate
            params += [{'params': [value], 'lr': base_lr}]
        else:
            # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
            params += [{'params': [value], 'lr': base_lr / 2}]

    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
    # We define the scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

    train(net, optimizer, 50, scheduler)

if __name__ == "__main__":
    main()