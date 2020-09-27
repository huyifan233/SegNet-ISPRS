
import itertools
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
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
from skimage import io



WINDOW_SIZE = (256, 256) # Patch size
STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels (e.g. RGB)
BATCH_SIZE = 32
LABELS = ["roads", "buildings", "low veg.", "trees", "cars", "clutter"] # Label names
N_CLASSES = len(LABELS)
WEIGHTS = torch.ones(N_CLASSES)
CACHE = True
palette = {0 : (255, 255, 255), # Impervious surfaces (white)
           1 : (0, 0, 255),     # Buildings (blue)
           2 : (0, 255, 255),   # Low vegetation (cyan)
           3 : (0, 255, 0),     # Trees (green)
           4 : (255, 255, 0),   # Cars (yellow)
           5 : (255, 0, 0),     # Clutter (red)
           6 : (0, 0, 0)}       # Undefined (black)

MAIN_FOLDER = 'D:\\Potsdam\\'
DATA_FOLDER = MAIN_FOLDER + '3_Ortho_IRRG\\3_Ortho_IRRG\\top_potsdam_{}_IRRG.tif'
LABEL_FOLDER = MAIN_FOLDER + '5_Labels_for_participants\\5_Labels_for_participants\\top_potsdam_{}_label.tif'
ERODED_FOLDER = MAIN_FOLDER + '5_Labels_for_participants_no_Boundary\\5_Labels_for_participants_no_Boundary\\top_potsdam_{}_label_noBoundary.tif'


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


def convert_from_color(arr_3d, palette={v: k for k, v in palette.items()}):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]


def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        range(len(label_values)))

    # print("Confusion matrix :")
    # print(cm)
    #
    # print("---")

    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    # print("{} pixels processed".format(total))
    # print("Total accuracy : {}%".format(accuracy))
    #
    # print("---")
    return accuracy
    # Compute F1 score
    # F1Score = np.zeros(len(label_values))
    # for i in range(len(label_values)):
    #     try:
    #         F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
    #     except:
    #         # Ignore exception if there is no element in class i for test set
    #         pass
    # print("F1Score :")
    # for l_id, score in enumerate(F1Score):
    #     print("{}: {}".format(label_values[l_id], score))
    #
    # print("---")

def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # Use the network on the test set
    test_images = (1 / 255 * np.asarray(io.imread(DATA_FOLDER.format(id)), dtype='float32') for id in test_ids)
    test_labels = (np.asarray(io.imread(LABEL_FOLDER.format(id)), dtype='uint8') for id in test_ids)
    eroded_labels = (convert_from_color(io.imread(ERODED_FOLDER.format(id))) for id in test_ids)
    all_preds = []
    all_gts = []

    # Switch the network to inference mode
    net.eval()

    for img, gt, gt_e in zip(test_images, test_labels, eroded_labels):
        pred = np.zeros(img.shape[:2] + (N_CLASSES,))

        # total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
        for i, coords in enumerate(
                grouper(batch_size, sliding_window(img, step=stride, window_size=window_size))):
            # Display in progress results
            # if i > 0 and total > 10 and i % int(10 * total / 100) == 0:
            #     _pred = np.argmax(pred, axis=-1)
                # fig = plt.figure(figsize=(10, 5))
                # fig.add_subplot(1, 3, 1)
                # plt.imshow(np.asarray(255 * img, dtype='uint8'))
                # fig.add_subplot(1, 3, 2)
                # plt.imshow(convert_to_color(_pred))
                # fig.add_subplot(1, 3, 3)
                # plt.imshow(gt)
                # clear_output()
                # plt.show()

            # Build the tensor
            image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

            # Do the inference
            outs = net(image_patches)
            outs = outs.data.cpu().numpy()

            # Fill in the results array
            for out, (x, y, w, h) in zip(outs, coords):
                out = out.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += out
            del (outs)

        pred = np.argmax(pred, axis=-1)

        # Display the result
        # clear_output()
        # fig = plt.figure()
        # fig.add_subplot(1, 3, 1)
        # plt.imshow(np.asarray(255 * img, dtype='uint8'))
        # fig.add_subplot(1, 3, 2)
        # plt.imshow(convert_to_color(pred))
        # fig.add_subplot(1, 3, 3)
        # plt.imshow(gt)
        # plt.show()

        all_preds.append(pred)
        all_gts.append(gt_e)

        # clear_output()
        # # Compute some metrics
        # metrics(pred.ravel(), gt_e.ravel())
    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                           np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy

def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    train_ids = ['2_10', '2_11', '3_10', '3_11', '6_10', '6_11', '4_10', '4_11', '5_10', '5_11', '7_10', '7_11']
    test_ids = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']
    train_set = ISPRS_dataset(train_ids, cache=CACHE)
    test_set = ISPRS_dataset(test_ids, cache=CACHE,test=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)
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
                # rgb = np.asarray(255 * np.transpose(data.data.cpu().numpy()[0], (1, 2, 0)), dtype='uint8')
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

        if e % save_epoch == 0:
            # We validate with the largest possible stride for faster computing
            net.eval()
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = Variable(data.cuda()), Variable(target.cuda())
                output = net(data)
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                test_acc = accuracy(pred, gt)
        #     # acc = test(net, test_ids, all=False, stride=min(WINDOW_SIZE))
                print("test accuracy: {}".format(test_acc))
        #     torch.save(net.state_dict(), './segnet256_epoch{}_{}'.format(e, acc))
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