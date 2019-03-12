# use this to make sliding window crops
from point import Point
from utils import *

class Feat(object):
    def __init__(self, row):
        self.pano_id = row[0]
        self.sv_image_x = float(row[1])
        self.sv_image_y = float(row[2])
        self.label_type = int(row[3])
        self.photographer_heading = float(row[4]) if row[4] is not None else None
        self.heading = float(row[5]) if row[5] is not None else None
        self.label_id = int(row[7])  if row[7] is not None else None
        
    def to_row(self):
        row =[]
        row.append(self.pano_id)
        row.append(self.sv_image_x)
        row.append(self.sv_image_y)
        row.append(self.label_type)
        row.append(self.photographer_heading)
        row.append(self.heading)
        row.append(self.label_id)
        return row
    
    def point(self):
        return Point( self.sv_image_x, self.sv_image_y )
    
    def __str__(self):
        label = GSVutils.utils.label_from_int[self.label_type-1]
        return '{} at {}'.format(label, self.point() )
    
    @classmethod
    def header_row(cls):
        row = ['Pano ID','SV_x','SV_y','Label',
               'Photographer Heading','Heading','Label ID']
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


def sliding_window(pano, stride=100, bottom_space=1600, side_space=300, cor_thresh=70):
    ''' take in a pano and produce a set of feats, ready for writing to a file
        labels assigned if the crop is within cor_thresh of a true label
        
        try cor_thresh = stride/sqrt(2)
    '''

    x, y = side_space, 0
    while(y > - (GSV_IMAGE_HEIGHT/2 - bottom_space)):
        while(x < GSV_IMAGE_WIDTH - side_space):
            # do things in one row
            
            # check if there's any features near this x,y point
            p = Point(x,y)
            
            label = 8 # for null
            for feat in pano.all_feats():
                if p.dist( feat.point() ) <= cor_thresh:
                    if label == 8:
                        label = feat.label_type
                    else:
                        if label != feat.label_type:
                            #print "Found conflicting labels, skipping."
                            continue
            row = [pano.pano_id, x, y, label, pano.photog_heading, None,None,None]
            yield Feat(row)
            
            x += stride
        y -= stride # jump down a row
        x = side_space
