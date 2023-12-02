import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import report_path, image_path, pegelname, \
    N_TIMESERIES, N_YEARS, MONTH, MONTH_HYD_YEAR

from utils.data_structures import read_data, check_path
from utils.plotting import plot_monthly_fitting, plot_thomasfierung_eval
from utils.thomasfiering import parameter_xp, parameter_sp, parameter_rp, thomasfiering


check_path(image_path)
check_path(report_path)

raw_data = read_data(f"data/others/Daten_{pegelname}.txt")

# -----------------------------------------
# Fit model parameters (Thomas-Fiering model)
# -----------------------------------------

tf_pars = pd.DataFrame(
    data={
        # "Monat": np.arange(1, 13),
        "Mittelwert": [parameter_xp(raw_data, i) for i in MONTH_HYD_YEAR],
        "Standardabweichung": [parameter_sp(raw_data, i) for i in MONTH_HYD_YEAR],
        "Korrelationskoeffizient": [parameter_rp(raw_data, i) for i in MONTH_HYD_YEAR]
    },
    index=[MONTH[i] for i in MONTH_HYD_YEAR]
)

tf_pars = tf_pars.round(4)
tf_pars.to_csv(f"data/{pegelname}_tomasfiering_parameters.csv", index=True)
tf_pars.to_latex(f"data/{pegelname}_tomasfiering_parameters.tex", index=True)

# -----------------------------------------
# Fit Gamma and LogNV to raw data
# -----------------------------------------

plot_monthly_fitting(raw_data)

# -----------------------------------------
# Generate time series
# -----------------------------------------

gen_arr = np.array([thomasfiering(raw_data, n_years=N_YEARS) for _ in range(N_TIMESERIES)])

gen_data = pd.DataFrame(
    data=gen_arr.transpose(),
    index=raw_data["Monat"],
    columns=[f"G{str(i).zfill(3)}" for i in range(1, N_TIMESERIES+1)]
    )

gen_data.to_csv(
    f"data/{pegelname}_thomasfiering_timeseries.csv", index=True)
gen_data.iloc[:].round(3).to_latex(
    f"data/{pegelname}_thomasfiering_timeseries.tex", index=True)
gen_data.iloc[:, :10].round(3).to_latex(
    f"data/{pegelname}_thomasfiering_timeseries_first10.tex", index=True)

plot_thomasfierung_eval(raw_data, gen_data)