import re
import os
import os.path as osp
import matplotlib.pyplot as plt


def plot1():
    """
    plot most recent training curve.
    """
    logs = os.listdir('logs')
    logs.sort()
    file = logs[-1]
    file_path = osp.join('logs', file)
    loss_list = []
    step_list = []
    accu_list = []
    with open(file_path) as f:
        for line in f.readlines():
            if 'step' in line:
                line = line.split('step')[1]
                numbers = re.findall(r"\d+\.?\d*", line)
                numbers = list(map(lambda x: float(x), numbers))
                if 'accuracy' in line:
                    step_list.append(numbers[0])
                    loss_list.append(numbers[1])
                    accu_list.append(numbers[2])
    plt.plot(step_list, loss_list, label='loss')
    plt.plot(step_list, accu_list, label='accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot1()
