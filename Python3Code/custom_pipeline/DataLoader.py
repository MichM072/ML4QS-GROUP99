from ..Chapter2.CreateDataset import CreateDataset
from ..util.VisualizeDataset import VisualizeDataset
from ..util import util
from pathlib import Path
import copy
import os
import sys


if os.path.exists(Path('./datasets/ML4QS/')):
    DATASET_PATH = Path('./datasets/ML4QS/')
else:
    DATASET_PATH = Path('/local/data/mmr497/datasets/ML4QS/')