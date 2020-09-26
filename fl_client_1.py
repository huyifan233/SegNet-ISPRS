import torch
from torchvision import datasets, transforms
from isprs_dataset import ISPRS_dataset
from gfl.core.client import FLClient
from gfl.core.strategy import WorkModeStrategy, TrainStrategy, LossStrategy
from gfl.core.trainer_controller import TrainerController

CLIENT_ID = 1
CACHE = True

if __name__ == "__main__":
    # CLIENT_ID = int(sys.argv[1])

    train_ids = ['4_10', '4_11', '5_10', '5_11']
    # test_ids = ['5', '21', '15', '30']

    train_set = ISPRS_dataset(train_ids, cache=CACHE)



    client = FLClient()
    gfl_models = client.get_remote_gfl_models()

    for gfl_model in gfl_models:
        optimizer = torch.optim.SGD(gfl_model.get_model().parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)
        train_strategy = TrainStrategy(optimizer=optimizer, scheduler=scheduler, batch_size=32,
                                       loss_function=LossStrategy.NLL_LOSS)
        gfl_model.set_train_strategy(train_strategy)

    TrainerController(work_mode=WorkModeStrategy.WORKMODE_STANDALONE, models=gfl_models, data=train_set, client_id=CLIENT_ID,
                      curve=True, concurrent_num=3).start()