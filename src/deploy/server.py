from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from fastapi import FastAPI, Request, Response
from src.predict import Predictor


app = FastAPI()
predictor = Predictor()