import torch
from isprs_dataset import ISPRS_dataset



def split_dataset():

    train_ids_0 = ['2_10', '2_11', '3_10', '3_11']
    train_ids_1 = ['4_10', '4_11', '5_10', '5_11']
    train_ids_2 = ['6_10', '6_11', '7_10', '7_11']
    test_ids = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']

    train_set_0 = ISPRS_dataset(train_ids_0, cache=True)
    train_set_1 = ISPRS_dataset(train_ids_1, cache=True)
    train_set_2 = ISPRS_dataset(train_ids_2, cache=True)
    test_set = ISPRS_dataset(test_ids, cache=True)

    # print(train_set.data_files)

    # train_set_0, train_set_1, train_set_2 = torch.utils.data.random_split(train_set, [0.3, 0.3, 0.4])

    torch.save(train_set_0, "data/train_sets/train_set_0")
    torch.save(train_set_1, "data/train_sets/train_set_1")
    torch.save(train_set_2, "data/train_sets/train_set_2")
    torch.save(test_set, "data/test_sets/test_set")



if __name__ == "__main__":
    split_dataset()