import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import report_path, image_path, pegelname

from utils.data_structures import read_data, check_path
import utils.statistics as st
from utils.plotting import plot_raw, plot_trend, plot_components, \
    plot_spectrum, plot_sin_waves, plot_characteristics, plot_acf, plot_dsk, \
    plot_breakpoint, pairplot


check_path(image_path)
check_path(report_path)

df = read_data(f"data/others/Daten_{pegelname}.txt")


# -----------------------------------------
#               Statistics
# -----------------------------------------


