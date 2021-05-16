class Face:

    def __init__(self, id, centroid, estimated_centroid, bounding_box):
        self.id = id
        self.centroid = centroid
        self.estimated_centroid = estimated_centroid
        self.bounding_box = bounding_box
