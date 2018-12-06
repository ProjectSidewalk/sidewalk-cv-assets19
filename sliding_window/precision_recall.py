import numpy as np
import math
from collections import defaultdict

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

def epr(pred_dict, truth_dict, R, N_classes=4):
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
        print "class {}".format(c) 
        dists = np.full((len(pred_x), len(true_x)), np.infty)
        for p in xrange(len(pred_x)):
            if pred_labels[p] == c:
                print "found prediction matching class"
                # num predicted per class
                output[c,1] += 1
                for t in xrange(len(true_x)):
                    if true_labels[t] == c:
                        dists[p,t] = np.linalg.norm(np.array([true_x[t],true_y[t]]) - np.array([pred_x[p],pred_y[p]]))
            correct = np.any(dists[p] < R) 
            if correct:
                output[c,0] += 1
        print dists
    #print output
    return output

class Point(object):
    """docstring for Point"""
    def __init__(self, x,y):
        super(Point, self).__init__()
        self.x = x
        self.y = y

    def dist(self, other):
        assert isinstance(other, Point)
        
        xd = self.x - other.x
        yd = self.y - other.y
        return math.sqrt( xd**2 + yd**2 )

    def __str__(self):
        return "{},{}".format(self.x, self.y)
        

def gpr(pred_dict, truth_dict, R, N_classes=4):
    true_x, true_y, true_labels = process(truth_dict)
    pred_x, pred_y, pred_labels = process(pred_dict)

    output = np.zeros((N_classes, 3))

    # count true
    for g in xrange(len(true_x)):
        output[true_labels[g],2] += 1

    # count predicted
    for g in xrange(len(pred_x)):
        output[pred_labels[g],1] += 1

    # load predictions into dict by label
    predictions_by_label = defaultdict(set)
    for i, label in enumerate(pred_labels):
        pt = Point(pred_x[i], pred_y[i])
        predictions_by_label[label].add(pt)

    # count correct
    # iterate over true points
    # for each, see if there's a nearby
    # prediction that matches the label
    for i, label in enumerate(true_labels):
        pt = Point(true_x[i], true_y[i])

        print "Looking for {} near {}".format(label, pt)

        for pred in predictions_by_label[label]:
            print "\tTrying {}\t{:07.1} away".format(pred, pt.dist(pred))
            if pt.dist(pred) <= R:
                print "Found a correct"
                output[label,0] += 1
                break

    return output


precision_recall = gpr