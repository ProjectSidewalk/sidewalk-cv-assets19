import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

######## Make fake data ########

N_true = 10
N_pred = 15
R_test = 100
#fake_precision = 0.5
fake_recall = 0.6
Y_range = (-1000, 1000)
X_range = (0, 3000)
N_classes = 4

fake_pred_dict = {} 
fake_truth_dict = {}

for i in xrange(N_true):
    x = np.random.randint(X_range[0], X_range[1])
    y = np.random.randint(Y_range[0], Y_range[1])
    coords = str(x) + "," + str(y)
    label = np.random.randint(0,N_classes)
    fake_truth_dict[coords] = label
    if np.random.random() <= fake_recall:
        fake_pred_dict[coords] = label
    else:
        x = np.random.randint(X_range[0], X_range[1])
        y = np.random.randint(Y_range[0], Y_range[1])
        coords = str(x) + "," + str(y)
        label = np.random.randint(0,N_classes)
        fake_pred_dict[coords] = label

num_true = len(fake_truth_dict)
print "Generated", num_true, "true"

for j in xrange(N_pred - N_true): 
    x = np.random.randint(X_range[0], X_range[1])
    y = np.random.randint(Y_range[0], Y_range[1])
    coords = str(x) + "," + str(y)
    label = np.random.randint(0,N_classes)
    fake_pred_dict[coords] = label

num_pred = len(fake_pred_dict)
print "Generated", num_pred, "predicted"

######## Process points ########

def process(points_dict):
    coords = map(lambda s: tuple(s.split(",")), points_dict.keys())
    l = np.array(points_dict.values())
    coords_int = map(lambda x: (int(x[0]), int(x[1])), coords)
    #x_tup, y_tup = zip(*coords_int)
    x_arr, y_arr = zip(*coords_int)
    #x_arr = np.array(x_tup)
    #y_arr = np.array(y_tup)
    return x_arr, y_arr, l

######## precision recall fn ########

def precision_recall(pred_dict, truth_dict, R):
    true_x, true_y, true_labels = process(truth_dict)
    pred_x, pred_y, pred_labels = process(pred_dict)
    #print true_x
    #print true_y
    #print true_labels

    output = np.zeros((N_classes, 3))
    # format: 
    # each row is a class w/ data [# correct, # predicted, # true]
    for g in xrange(len(true_x)):
        output[true_labels[g],2] += 1

    for c in xrange(N_classes): 
        dists = np.full((len(pred_x), len(true_x)), np.infty)
        for p in xrange(len(pred_x)):
            if pred_labels[p] == c:
                # num predicted per class
                output[c,1] += 1
                for t in xrange(len(true_x)):
                    if true_labels[t] == c:
                        dists[p,t] = np.linalg.norm(np.array([true_x[t],true_y[t]]) - np.array([pred_x[p],pred_y[p]]))
            correct = np.any(dists[p] < R) 
            if correct:
                output[c,0] += 1
    #print output
    return output

######## main #########


true_x, true_y, true_labels = process(fake_truth_dict)
pred_x, pred_y, pred_labels = process(fake_pred_dict)

precision_recall(fake_pred_dict, fake_truth_dict, R_test)
