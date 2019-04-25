import math

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

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def from_str(cls, s):
        x,y = map(int, s.split(','))
        return cls(x,y)