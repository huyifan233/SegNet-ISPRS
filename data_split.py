import torch
from isprs_dataset import ISPRS_dataset



def split_dataset():

    train_ids_0 = ['2_10', '2_11', '2_12', '2_13', '3_10', '3_11', '3_12', '3_13']
    train_ids_1 = ['4_10', '4_11', '4_12', '4_13', '5_10', '5_11', '5_12', '5_13']
    train_ids_2 = ['6_10', '6_11', '6_12', '6_13', '7_10', '7_11', '7_12', '7_13']
    test_ids = ['2_14', '3_14', '4_14', '5_14', '6_14', '7_14']

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