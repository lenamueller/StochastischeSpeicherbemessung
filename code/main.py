import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from read_data import read_data
from setup import pegelname, tu_mediumblue, tu_red, tu_grey


raw_path = f"Daten_{pegelname}.txt"
raw = read_data(raw_path)
