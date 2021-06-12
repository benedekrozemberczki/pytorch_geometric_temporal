from typing import List
import json
import numpy as np
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

class MTMDatasetLoader():
    """
    A dataset of `Methods-Time Measurement-1 <https://en.wikipedia.org/wiki/Methods-time_measurement>`_ 
    (MTM-1) motions, signalled as consecutive video frames of 21 3D hand keypoints, acquired via 
    `MediaPipe Hands <https://google.github.io/mediapipe/solutions/hands.html>`_ from RGB-Video 
    material. Vertices are the finger joints of the human hand and edges are the bones connecting 
    them. The targets are manually labeled for each frame, according to one of the five MTM-1 
    motions (classes :math:`C`): Grasp, Release, Move, Reach, Position plus a negative class for 
    frames without graph signals (no hand present). This is a classification task where :math:`T` 
    consecutive frames need to be assigned to the corresponding class :math:`C`. The data x is 
    returned in shape :obj:`(3, 21, T)`, the target is returned one-hot-encoded in shape :obj:`(T, 6)`.
    """

    def __init__(self):
        self._read_web_data()

    def _read_web_data(self):
        url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/mtm_1.json"
        self._dataset = json.loads(urllib.request.urlopen(url).read())
        
    def _get_edges(self):
        self._edges = np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        self._edge_weights = np.array([1 for d in self._dataset["edges"]]).T

    def _get_features(self):
        dic = self._dataset
        joints = [str(n) for n in range(21)]
        dataset_length =len(dic["0"].values())
        features = np.zeros((dataset_length, 21, 3))
        
        for j, joint in enumerate(joints):
            for t, xyz in enumerate(dic[joint].values()):
                xyz_tuple = list(map(float, xyz.strip('()').split(',')))
                features[t, j, :] = xyz_tuple

        self.features = [
            features[i : i + self.frames, :].T
            for i in range(len(features) - self.frames)
        ]

    def _get_targets(self):
        #target eoncoding: {0 : 'Grasp', 1 : 'Move', 2 : 'Negative', 
        #                   3 : 'Position', 4 : 'Reach', 5 : 'Release'}
        targets = []
        for _, y in self._dataset["LABEL"].items():
            targets.append(y)

        n_values = np.max(targets) + 1
        targets_ohe = np.eye(n_values)[targets]

        self.targets = [
            targets_ohe[i:i + self.frames, :]
            for i in range(len(targets_ohe) - self.frames)
        ]

    def get_dataset(self, frames: int = 16) -> StaticGraphTemporalSignal:
        """Returning the MTM-1 motion data iterator.

        Args types:
            * **frames** *(int)* - The number of consecutive frames T, default 16. 
        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The MTM-1 dataset.
        """
        self.frames = frames
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()

        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets
        )
        return dataset
