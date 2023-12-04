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


# Agenda
CHECK_DATA = False
CALC_COMPONENTS = False
CALC_STATS = False
FIT_CAPACITIES = True

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
    gen_data = thomasfiering(df)
    

# -----------------------------------------
#           calc. capacity (FSA)
# -----------------------------------------

CALC_CAPACITIES = False     # ! Don't generate new data
if CALC_CAPACITIES:
    fsa(raw_data=df, gen_data=gen_data)

# -----------------------------------------
#      fit distribution to capactities
# -----------------------------------------

if FIT_CAPACITIES:
    fit_capacity()

# -----------------------------------------
#            simulate storage 
# -----------------------------------------


print("\n--------------------------------------")
