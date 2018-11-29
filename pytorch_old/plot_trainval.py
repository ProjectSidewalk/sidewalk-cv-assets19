import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

def accuracy_plot(P_list, train_acc, test_acc):
    val_1_acc = test_acc[:25] 
    val_2_acc = test_acc[25:]
    train_1_acc = train_acc[:25]
    train_2_acc = train_acc[25:]

    plt.plot(P_list, val_1_acc, 'r-', label='val accuracy run 1')
    #plt.plot(P_list, val_1_acc, 'ro')
    plt.plot(P_list, val_2_acc, 'm-', label='val accuracy run 2')
    #plt.plot(P_list, val_2_acc, 'mo')

    plt.plot(P_list, train_1_acc, 'b-', label='train accuracy run 1')
    #plt.plot(P_list, train_1_acc, 'bo')
    plt.plot(P_list, train_2_acc, 'c-', label='train accuracy run 2')
    #plt.plot(P_list, train_2_acc, 'co')

    plt.xlim(0,24)
    plt.ylim(0.7,1)
    plt.xlabel('Iteration over data (Epoch)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def loss_plot(P_list, train_loss, test_loss):
    val_1_loss = test_loss[:25]
    val_2_loss = test_loss[25:]
    train_1_loss = train_loss[:25]
    train_2_loss = train_loss[25:]

    plt.plot(P_list, val_1_loss, 'r-', label='val loss run 1')
    #plt.plot(P_list, val_1_loss, 'ro')
    plt.plot(P_list, val_2_loss, 'm-', label='val loss run 2')
    #plt.plot(P_list, val_2_loss, 'mo')

    plt.plot(P_list, train_1_loss, 'b-', label='train loss run 1')
    #plt.plot(P_list, train_1_loss, 'bo')
    plt.plot(P_list, train_2_loss, 'c-', label='train loss run 2')
    #plt.plot(P_list, train_2_loss, 'co')

    plt.xlim(0,24)
    plt.ylim(0,1)
    plt.xlabel('Iteration over data (Epoch)')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def read_data(filename):
    train_acc = []
    test_acc = []
    
    train_loss = []
    test_loss = [] 

    with open(filename, 'r') as f:
        data = f.readlines()
    for line in data:
        l = line.split()
        if (len(l) > 0) and (l[0] == 'train'):
            train_loss.append(float(l[2]))
            train_acc.append(float(l[4]))
        if (len(l) > 0) and (l[0] == 'val'):
            test_loss.append(float(l[2]))
            test_acc.append(float(l[4]))
    return train_loss, train_acc, test_loss, test_acc


P_list = range(25)

train_loss, train_acc, val_loss, val_acc = read_data('pytorch_output.txt')
accuracy_plot(P_list, train_acc, val_acc)
loss_plot(P_list, train_loss, val_loss)

