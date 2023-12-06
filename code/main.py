import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import pegelname, image_path

from utils.data_structures import read_data, check_path
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
from utils.dimensioning.speicherausbaugrad import speicherausbaugrad
from utils.dimensioning.reliability import reliability

# Agenda
CHECK_DATA = False          # Finished
CALC_COMPONENTS = False     # TODO Check
CALC_STATS = False          # TODO Check
FIT_CAPACITIES = False      # Finished
SIMULATION = True           # Finished
CALC_BETA = False           # Finished
RELIABILITY = False         # Finished


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
    fsa(raw_data=df)

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
    cap90 = 28.873
    cap95 = 32.061
    cap_min_gen = 11.923
    cap_max_gen = 41.820
    
    # Kapazität und Anfangsfüllung gem. Aufgabenstellung
    run_simulation(var="original", cap=cap90, initial_storage=0.5*cap90)

    # Variation der Anfangsfüllung
    run_simulation(var="original", cap=cap90, initial_storage=0)
    run_simulation(var="original", cap=cap90, initial_storage=cap90)
    
    # Variation der Kapazität
    run_simulation(var="original", cap=cap_hist, initial_storage=0.5*cap_hist)
    run_simulation(var="original", cap=cap95, initial_storage=0.5*cap95)
    run_simulation(var="original", cap=100.000, initial_storage=0.5*100.00)
    run_simulation(var="original", cap=cap_min_gen, initial_storage=0.5*cap_min_gen)
    run_simulation(var="original", cap=cap_max_gen, initial_storage=0.5*cap_max_gen)

    # unendlicher Speicher
    run_simulation(var="original", cap=np.inf, initial_storage=0)
    
# -----------------------------------------
#           Speicherausbaugrad
# -----------------------------------------

if CALC_BETA:
    mq_hm3 = 1.188 * 60*60*24*365/1000000
    speicherausbaugrad(var="original", mq=mq_hm3)

# -----------------------------------------
#          Zuverlässigkeit
# -----------------------------------------

if RELIABILITY:
    fn_storage_sim = [
        "data/Klingenthal_storagesim_original_0_28.873.csv",
        "data/Klingenthal_storagesim_original_0_inf.csv",
        "data/Klingenthal_storagesim_original_5.962_11.923.csv",
        "data/Klingenthal_storagesim_original_11.448_22.896.csv",
        "data/Klingenthal_storagesim_original_14.437_28.873.csv",
        "data/Klingenthal_storagesim_original_16.03_32.061.csv",
        "data/Klingenthal_storagesim_original_20.91_41.82.csv",
        "data/Klingenthal_storagesim_original_28.873_28.873.csv",
        "data/Klingenthal_storagesim_original_50.0_100.0.csv"
    ]
    reliability(fn_storage_sim)

print("\n--------------------------------------")
