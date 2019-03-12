import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import GSVutils.utils
from GSVutils.point import Point as Point


class Feat(object):
    def __init__(self, row):
        self.pano_id = row[0]
        self.sv_image_x = float(row[1])
        self.sv_image_y = float(row[2])
        self.label_type = int(row[3])
        self.photographer_heading = float(row[4]) if row[4] is not None else None
        self.heading = float(row[5]) if row[5] is not None and len(row[5]) > 1 else None
        
    def to_row(self):
        row =[]
        row.append(self.pano_id)
        row.append(self.sv_image_x)
        row.append(self.sv_image_y)
        row.append(self.label_type)
        row.append(self.photographer_heading)
        row.append(self.heading)
        return row
    
    def point(self):
        return Point( self.sv_image_x, self.sv_image_y )
    
    def __str__(self):
        label = GSVutils.utils.label_from_int[self.label_type-1]
        return '{} at {}'.format(label, self.point() )
    
    @classmethod
    def header_row(cls):
        row = ['Pano ID','SV_x','SV_y','Label',
               'Photographer Heading','Heading']
        return row


class Pano(object):
    
    def __init__(self):
        self.feats = {1:[], 2:[], 3:[], 4:[]}
        self.pano_id        = None
        self.photog_heading = None

    def add_feature(self, row):
        feat = Feat(row)
        if self.pano_id is None:
            self.pano_id = feat.pano_id
        assert self.pano_id == feat.pano_id
        
        if self.photog_heading is None:
            self.photog_heading = feat.photographer_heading
        
        self.feats[feat.label_type].append( feat )
            
    def __hash__(self):
        return hash( self.pano_id )
    
    def all_feats(self):
        ''' iterate over all features, regardless of type '''
        for label, features in self.feats.iteritems():
            for feature in features:
                yield feature
    
    def __str__(self):
        s = 'pano{}\n'.format(self.pano_id)
        for feat in self.all_feats():
            s += '{}\n'.format(feat)
        return s
    
    def __len__(self):
        ''' return the total number of feats in this pano '''
        c = 0
        for _ in self.all_feats():
            c += 1
        return c