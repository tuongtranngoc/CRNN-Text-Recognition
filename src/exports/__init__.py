from .. import config as cfg
from ..utils.logger import Logger
from ..models.crnn import CRNN
from ..utils.metrics import map_char2id
from ..data.transformation import TransformCRNN