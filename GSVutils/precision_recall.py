import numpy as np
import math
from collections import defaultdict


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

    @classmethod
    def from_str(cls, s):
        x,y = map(int, s.split(','))
        return cls(x,y)
        

def precision_recall(pred_dict, truth_dict, R, N_classes=4):
    """ returns a N_classes x 3 matrix where the rows correspond to classes
        and the cols correspond to the counts of [correct, predicted, actual]
    """
    output = np.zeros((N_classes, 3))

    # count true
    for label in truth_dict.values():
        output[label,2] += 1

    # count predicted
    for label in pred_dict.values():
        output[label,1] += 1

    # load predictions into dict by label
    predictions_by_label = defaultdict(set)
    for coords, label in pred_dict.iteritems():
        pt = Point.from_str(coords)
        predictions_by_label[label].add(pt)

    # count correct
    # iterate over true points
    # for each, see if there's a nearby
    # prediction that matches the label
    for coords, label in truth_dict.iteritems():
        pt = Point.from_str(coords)

        #print "Looking for {} near {}".format(label, pt)

        for pred in predictions_by_label[label]:
            #print "\tTrying {}\t{:07.1} away".format(pred, pt.dist(pred))
            if pt.dist(pred) <= R:
                #print "Found a correct"
                output[label,0] += 1
                break

    return output

def partition_based_on_correctness(pred_dict, truth_dict, R):
    ''' given two dictionaries of points, one true, and one predicted,
        this function partitions the predicted points into "correct"
        and "incorrect" sets for each class.
    '''
    gt_by_label = defaultdict(set)
    for coords, label in truth_dict.iteritems():
        pt = Point.from_str(coords)
        gt_by_label[label].add(pt)

    cor_preds_by_label = defaultdict(set)
    inc_preds_by_label = defaultdict(set)
    for coords, label in pred_dict.iteritems():
        pt = Point.from_str(coords)

        for truth in gt_by_label[label]:
            if truth.dist(pt) < R:
                cor_preds_by_label[label].add(pt)
                continue
        if pt not in cor_preds_by_label[label]:
            inc_preds_by_label[label].add(pt)

    correct   = {}
    incorrect = {}
    for label in cor_preds_by_label:
        for pt in cor_preds_by_label[label]:
            correct[str(pt)] = label

    for label in inc_preds_by_label:
        for pt in inc_preds_by_label[label]:
            incorrect[str(pt)] = label

    return correct, incorrect
