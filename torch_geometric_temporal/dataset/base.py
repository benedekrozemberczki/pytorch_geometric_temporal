import os
import urllib.request
import json
import ssl


class AbstractDataLoader(object):
    SRC_PREFIX = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/"

    def __init__(self, fname: str, datadir: str = None):
        if not datadir:
            datadir = os.path.join(os.getcwd(), "data")
        os.makedirs(datadir, exist_ok=True)
        self.fname = fname
        self.datadir = datadir

    def _set_fname(self, fname: str):
        self.fname = fname

    def _set_src_prefix(self, src_prefix: str):
        self.SRC_PREFIX = src_prefix

    def _save_to_local(self):
        with open(os.path.join(self.datadir, self.fname), "w") as f:
            ssl._create_default_https_context = ssl._create_unverified_context
            data = urllib.request.urlopen(self.SRC_PREFIX + self.fname).read()
            data = json.loads(data.decode())
            json.dump(data, f, ensure_ascii=False)

    def _load(self):
        if not os.path.exists(os.path.join(self.datadir, self.fname)):
            self._save_to_local()
        return json.loads(open(os.path.join(self.datadir, self.fname)).read())
