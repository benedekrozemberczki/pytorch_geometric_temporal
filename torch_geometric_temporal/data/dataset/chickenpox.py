import io
import json
import numpy as np
from six.moves import urllib

class ChickenpoxDataset(object):

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/chickenpox.json"
        dataset = json.loads(urllib.request.urlopen(url).read())
        edges = np.array(dataset["edges"]).T
        print(edges.shape)
 
        
