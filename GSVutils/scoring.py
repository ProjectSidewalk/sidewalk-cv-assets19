import numpy as np
import math
from collections import defaultdict

from GSVutils.clustering import non_max_sup

label_from_int = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')
pytorch_label_from_int = ('Missing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")


class Point(object):
    """docstring for Point"""
    def __init__(self, x,y):
        super(Point, self).__init__()
        self.x = x
        self.y = y
        self.r = None

    def dist(self, other):
        assert isinstance(other, Point)
        
        xd = self.x - other.x
        yd = self.y - other.y
        return math.sqrt( xd**2 + yd**2 )

    def near(self, other):
        assert isinstance(other, Point)

        return self.dist(other) <= self.r

    def __str__(self):
        return "{},{}".format(self.x, self.y)

    @classmethod
    def from_str(cls, s):
        x,y = map(float, s.split(','))
        return cls(x,y)

def score(pred_dict, truth_dict, sizes_dict, r=.9, clust_r=150):
    ''' takes in three dicts:
        a dict of raw predictions in [missing ramp, null, obstruction, ramp, sfc_problem] encoding,
        where each prediction is a vector of length five. These are converted using non_max_sup

        a dict of truths using [ramp, no ramp, obstruction, sfc prob] encoding

        and a dict of sizes (with keys matching truths) for the size of each true feature

        takes in an r radius for correctness. This float is a fraction of the crop size,
        eg for a feature to be marked correct, it must be within r * crop_size (length of a single edge)

        clust_r is the clustering radius passed to the non-max suppression function for clustering crops

        outputs dicts of features (using the same encoding) of correct, incorrect features drawn from pred_dict.
    '''

    # here predictions are still a dict of arrays
    pred_dict = non_max_sup(pred_dict, radius=clust_r, clip_val=None, ignore_ind=1)
    # now predictions are ints

    # in here we need to map from pytorch class numbering to
    # [ramp, no ramp, obstruction, sfc_prob] zero indexed
    # ground truth is stored in DB using above encoding but 1-indexed
    # this is compensted when loaded using get_ground_truth
    for coord in pred_dict:
        pytorch_label = pred_dict[coord]
        label = label_from_int.index( pytorch_label_from_int[ pytorch_label ] )
        pred_dict[coord] = label


    gt_by_label = defaultdict(set)
    for coords, label in truth_dict.iteritems():
        pt   = Point.from_str(coords)
        try:
            pt.r = float(sizes_dict[coords]) * r
        except:
            # if for some reason we can't get sizes, let's fallback on a hard value
            pt.r = 300.0
        gt_by_label[label].add(pt)

    cor_preds_by_label = defaultdict(set)
    inc_preds_by_label = defaultdict(set)
    for coords, label in pred_dict.iteritems():
        pt = Point.from_str(coords)

        for truth in gt_by_label[label]:
            if truth.near(pt):
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

    print "Returning {} correct and {} incorrect.".format(len(correct), len(incorrect))
    return correct, incorrect
