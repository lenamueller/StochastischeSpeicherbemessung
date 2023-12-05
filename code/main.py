import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import pegelname, image_path

from utils.data_structures import read_data, read_gen_data, check_path
check_path(image_path)

from utils.plotting import plot_raw

from utils.checks.consistency_check import consistency_check
from utils.checks.homogenity_check import homogenity_check
from utils.checks.stationarity_check import stationarity_check

from utils.components.trend_comp import trend_comp
from utils.components.seasonal_comp import seasonal_comp
from utils.components.autocorr_comp import autocorr_comp
from utils.components.irregular_comp import irregular_comp

from utils.statistics import statistics
from utils.thomasfiering import thomasfiering

from utils.dimensioning.fsa import fsa
from utils.dimensioning.fit_capacity import fit_capacity
from utils.dimensioning.simulation import run_simulation


# Agenda
CHECK_DATA = False
CALC_COMPONENTS = False
CALC_STATS = False
FIT_CAPACITIES = False
SIMULATION = True

# -----------------------------------------
#               read data
# -----------------------------------------
 
df = read_data(f"data/{pegelname}_raw.txt")
plot_raw(df)

# -----------------------------------------
#               check data
# -----------------------------------------
if CHECK_DATA:
    consistency_check(df)
    homogenity_check(test_pegel=df, ref_pegel=read_data(f"data/Rothenthal_raw.txt"))
    stationarity_check(df, "Durchfluss_m3s")

# -----------------------------------------
#       calc time series components
# -----------------------------------------

if CALC_COMPONENTS:
    trend_comp(df)
    seasonal_comp(df)
    autocorr_comp(df)
    irregular_comp(df)

    df.to_csv(f"data/{pegelname}_components.csv", index=False)

# -----------------------------------------
#           calc. statistics
# -----------------------------------------

if CALC_STATS:
    statistics(df)

# -----------------------------------------
#   generate time series (Thomas Fiering)
# -----------------------------------------

GEN_TIMESERIES = False      # ! Don't generate new data
if GEN_TIMESERIES:
    thomasfiering(df)

# -----------------------------------------
#           calc. capacity (FSA)
# -----------------------------------------

CALC_CAPACITIES = False     # ! Don't generate new data
if CALC_CAPACITIES:
    fsa()

# -----------------------------------------
#      fit distribution to capactities
# -----------------------------------------

if FIT_CAPACITIES:
    fit_capacity()

# -----------------------------------------
#            simulate storage 
# -----------------------------------------

if SIMULATION:
    cap_hist = 22.896
    cap90 = 26.006
    cap95 = 28.718
    cap_large = 100.000
    
    # Kapazität und Anfangsfüllung gem. Aufgabenstellung
    run_simulation(var="original", cap=cap90, initial_storage=0.5*cap90)
    
    # Variation: leere Anfangsfüllung
    run_simulation(var="original", cap=cap90, initial_storage=0)
    
    # Variation: 95%-Kapazität, sehr große Kapazität, unbegrenzte Kapazität
    run_simulation(var="original", cap=cap_hist, initial_storage=0)
    run_simulation(var="original", cap=np.inf, initial_storage=0)
    run_simulation(var="original", cap=cap95, initial_storage=0)
    run_simulation(var="original", cap=cap_large, initial_storage=0)
    
    # generated data
    run_simulation(var="G001", cap=cap90, initial_storage=0.5*cap_hist)

print("\n--------------------------------------")
