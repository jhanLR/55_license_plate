from models.models import convert
# from models.models2 import convert
from utils.google_utils import *
from utils.layers import *
from utils.parse_config import *
from utils import torch_utils


if __name__ == "__main__":
    cfg = 'yolov7.cfg'
    pt = 'yolov7.pt'
    saveto = 'yolov7.weights'
    convert(cfg, pt, saveto) # models.py
    # convert(cfg, pt) # models2.py
    