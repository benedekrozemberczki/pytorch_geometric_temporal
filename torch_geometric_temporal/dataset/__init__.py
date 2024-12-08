from .chickenpox import ChickenpoxDatasetLoader
from .pedalme import PedalMeDatasetLoader
from .metr_la import METRLADatasetLoader
from .pems_bay import PemsBayDatasetLoader
from .wikimath import WikiMathsDatasetLoader
from .windmilllarge import WindmillOutputLargeDatasetLoader
from .windmillmedium import WindmillOutputMediumDatasetLoader
from .windmillsmall import WindmillOutputSmallDatasetLoader
from .encovid import EnglandCovidDatasetLoader
from .twitter_tennis import TwitterTennisDatasetLoader
from .montevideo_bus import MontevideoBusDatasetLoader
from .mtm import MTMDatasetLoader

DATALOADERS = [
    ChickenpoxDatasetLoader,
    PedalMeDatasetLoader,
    METRLADatasetLoader,
    PemsBayDatasetLoader,
    WikiMathsDatasetLoader,
    WindmillOutputLargeDatasetLoader,
    WindmillOutputMediumDatasetLoader,
    WindmillOutputSmallDatasetLoader,
    EnglandCovidDatasetLoader,
    TwitterTennisDatasetLoader,
    MontevideoBusDatasetLoader,
    MTMDatasetLoader,
]