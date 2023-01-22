import numpy as np


def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)


def split_data(all_data, test_data_ratio: float):
    shuffle_in_unison(all_data[0], all_data[1])
    nr_of_elements_test = int(len(all_data[0]) * test_data_ratio)
    test_data = all_data[0][:nr_of_elements_test], all_data[1][:nr_of_elements_test]
    train_data = all_data[0][nr_of_elements_test:], all_data[1][nr_of_elements_test:]
    return train_data, test_data
