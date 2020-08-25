import torch
from isprs_dataset import ISPRS_dataset



def split_dataset():
    train_ids = ['1', '3', '23', '26', '7', '11', '13', '28', '17', '32', '34', '37']
    test_ids = ['5', '21', '15', '30']

    train_set = ISPRS_dataset(train_ids, cache=True)
    test_set = ISPRS_dataset(test_ids, cache=True)

    train_set_0, train_set_1, train_set_2 = torch.utils.data.random_split(train_set, [0.3, 0.3, 0.4])

    torch.save(train_set_0, "data/train_sets/train_set_0")
    torch.save(train_set_1, "data/train_sets/train_set_1")
    torch.save(train_set_2, "data/train_sets/train_set_2")
    torch.save(test_set, "data/test_sets/test_set")



if __name__ == "__main__":
    split_dataset()