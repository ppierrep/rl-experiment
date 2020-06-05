from IPython.display import clear_output
import matplotlib.pyplot as plt


def plot(rewards, losses, action_takens):
    clear_output(True)
    plt.figure(figsize=(20,5))

    plt.subplot(131)
    plt.title('rewards')
    plt.plot(rewards)

    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)

    plt.subplot(133)
    unique, counts = np.unique(action_takens, return_counts=True)
    plt.bar(unique, counts/np.sum(counts))
    plt.title("Action distribution")

    plt.show()