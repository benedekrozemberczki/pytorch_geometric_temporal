import io
import json
from six.moves import urllib

class ChickenpoxDataset(object):

    def __init__(self):
        return self

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/discrete/chickenpox.json"
        dataset = json.loads(urllib.request.urlopen(url).read())
        
 
        
