import numpy as np
import matplotlib.pyplot as plt


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


def comparison_plots(acc_NN_RL, acc_RL_NN, acc_RL, acc_NN):
    final_plot_data = []
    final_plot_labels = []
    if acc_NN_RL is not None:
        plt.plot(acc_NN_RL)
        final_plot_data.append(acc_NN_RL)
        final_plot_labels.append('NN_to_RL accuracy')
    if acc_RL_NN is not None:
        plt.plot(acc_RL_NN)
        final_plot_data.append(acc_RL_NN)
        final_plot_labels.append('RL_to_NN accuracy')
    if acc_RL is not None:
        plt.plot(acc_RL)
        final_plot_data.append(acc_RL)
        final_plot_labels.append('RL accuracy')
    if acc_NN is not None:
        plt.plot(acc_NN)
        final_plot_data.append(acc_NN)
        final_plot_labels.append('NN accuracy')
    plt.legend(final_plot_labels, loc='lower right')
    plt.title('Accuracy comparison line plot')
    plt.show()

    plt.boxplot(final_plot_data, labels=final_plot_labels)
    plt.title('Accuracy comparison box plot')
    plt.show()
