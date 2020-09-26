import torch
import numpy as np
from torchvision import datasets, transforms
from isprs_dataset import ISPRS_dataset
from gfl.core.client import FLClient
from gfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from gfl.core.trainer_controller import TrainerController

CLIENT_ID = 1
CACHE = True

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return torch.nn.functional.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0),input.size(1), -1)
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))
        target = target.view(-1)
        return torch.nn.functional.cross_entropy(output, target,weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

def accuracy(output, target):
    pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
    gt = target.data.cpu().numpy()[0]
    return 100 * float(np.count_nonzero(pred == gt)) / gt.size



if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    train_ids = ['4_10', '4_11', '5_10', '5_11', '7_10', '7_11']
    # test_ids = ['5', '21', '15', '30']

    train_set = ISPRS_dataset(train_ids, cache=CACHE)



    client = FLClient()
    gfl_models = client.get_remote_gfl_models()

    for gfl_model in gfl_models:
        optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
        train_strategy = TrainStrategy(optimizer=optimizer, scheduler=scheduler, batch_size=10,
                                       loss_function=CrossEntropy2d, accuracy_function=accuracy)
        gfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=gfl_models, data=train_set, client_id=CLIENT_ID,
                      curve=True, concurrent_num=3).start()