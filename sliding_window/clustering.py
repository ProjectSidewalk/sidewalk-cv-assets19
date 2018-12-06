#!/usr/bin/env python
# coding: utf-8
from collections import defaultdict
import collections
import math
import csv


class Point(object):
    def __init__(self, x, y, preds):
        super(Point, self).__init__()
        
        self.x = x
        self.y = y
        self.preds = preds
        return
        
    def dist(self, other):
        assert isinstance(other, Point)
        
        xd = self.x - other.x
        yd = self.y - other.y
        return math.sqrt( xd**2 + yd**2 )
    
    def __str__(self):
        return "({},{} label:{} {})".format(self.x, self.y, self.label(), self.score())
    
    @classmethod
    def from_str(cls, s, preds):
        x,y = map(int, s.split(','))
        return cls(x,y, preds)
    
    def label(self):
        return self.preds.index( max(self.preds) )
    
    def score(self):
        return( max(self.preds) )

    def to_pred(self):
        coord = "{},{}".format(self.x, self.y)
        return coord, self.label()


class PointSet(collections.Set):
    ''' set of points of the same label value '''
    def __init__(self, points=()):
        super(PointSet, self).__init__()
        
        self._set = set()
        
        self.label = None
        
        for point in points:  
            self.add(point)
        return
    
    def add(self, other):
        assert isinstance(other, Point)
        
        if self.label is None:
            self.label = other.label()
        else:
            assert other.label() == self.label
        
        self._set.add(other)
    
    def overlaps(self, other, radius):
        ''' returns True if other is within radius of *any* 
            point in this set, *and* other has same label as
            this set '''
        assert isinstance(other, Point)
        
        if other.label() != self.label:
            return False
        
        for point in self._set:
            if point.dist(other) <= radius:
                return True
        return False
    
    def any_overlaps(self, other_set, radius):
        assert isinstance(other_set, PointSet)
        
        if other_set.label != self.label:
            return False
        
        for pt in other_set:
            if self.overlaps(pt, radius):
                return True
            
        return False
    
    def merge(self, other_set):
        assert isinstance(other_set, PointSet)
        assert other_set.label == self.label
        
        for pt in other_set:
            self.add(pt)

    def get_strongest(self):
        ''' returns the item in the set with the highest score '''
        return max(self, key=lambda x: x.score())
        
    def __len__(self):
        return len(self._set)
    
    def __iter__(self):
        return self._set.__iter__()
    
    def __contains__(self, other):
        return self._set.__contains__(other)


def non_max_sup(predictions, radius=1.1, clip_val=None, ignore_last=False):
    unclustered = set()

    print "have {} untrimmmed".format(len(predictions))
    
    # load unclipped and non-nullcrop predictions into set
    for coords in predictions:
        pt = Point.from_str( coords, predictions[coords])
        
        # clip
        clip = (clip_val is not None) and (pt.score() < clip_val)
            
        # ignore if last label is strongest (eg nullcrop)
        ignore = ignore_last and (pt.label() == len(pt.preds)-1)
        
        if not clip and not ignore:
            unclustered.add( Point.from_str(coords, predictions[coords]) )

    print "trimmed to {}".format(len(unclustered))
    
    clustered = []
    while len(unclustered) > 0:
        make_new_clust = True
        
        pt = unclustered.pop()
                
        for cluster in clustered:
            if cluster.overlaps(pt, radius):
                cluster.add(pt)
                make_new_clust = False
                # we can stop for this point
                break
        
        # if we get here, we need to make a
        # new cluster containing this pt
        if make_new_clust:
            s = PointSet()
            s.add(pt)
            clustered.append( s )
            
    # now we need to consolidate clusters
    # loop over clusters, and if any can be
    # merged, do so. return when converged
    merged_this_loop = 1
    while merged_this_loop > 0:
        merged_this_loop = 0
        for this_clust in clustered:
            for other_clust in clustered:                
                if this_clust is not other_clust \
                and this_clust.any_overlaps(other_clust, radius):
                    this_clust.merge(other_clust)
                    clustered.remove(other_clust)
                    merged_this_loop += 1

    # now we need to put clusters into a dict
    clustered_dict = {}
    for clust in clustered:
        pt = clust.get_strongest()
        coord, label = pt.to_pred()
        clustered_dict[coord] = label

    return clustered_dict
