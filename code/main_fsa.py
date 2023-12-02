import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from config import report_path, image_path, pegelname, MONTH_HYD_YEAR_TXT, \
    N_TIMESERIES, SEC_PER_MONTH

from utils.data_structures import read_data, read_gen_data, check_path
from utils.plotting import plot_monthly_discharge, plot_storage, plot_fsa
from utils.fsa import monthly_discharge, calc_storage, calc_capacity


check_path(image_path)
check_path(report_path)


raw_data = read_data(f"data/others/Daten_{pegelname}.txt")
gen_data = read_gen_data()
gen_data.reset_index(inplace=True)

# -----------------------------------------
# calculate monthly discharge (Soll-Abgabe)
# -----------------------------------------

monthly_dis = {}

# original data
monthly_dis["original"] = monthly_discharge(raw_data["Durchfluss_m3s"].to_numpy())

# generated data
for i in range(N_TIMESERIES):
    monthly_dis[f"G{str(i+1).zfill(3)}"] = monthly_discharge(
        gen_data[f"G{str(i+1).zfill(3)}"].to_numpy())

# save data
monthly_dis = pd.DataFrame.from_dict(monthly_dis)
monthly_dis.index = MONTH_HYD_YEAR_TXT

monthly_dis = monthly_dis.transpose()
monthly_dis.round(3).to_csv(f"data/{pegelname}_monthly_discharge.csv", index=True)
monthly_dis.round(3).to_latex(f"data/{pegelname}_monthly_discharge.tex", index=True)

plot_monthly_discharge(monthly_dis)

# -----------------------------------------
# Calculate capacity for original data
# -----------------------------------------

raw_data["Durchfluss_hm3"] = raw_data["Durchfluss_m3s"] * SEC_PER_MONTH/1000000
gen_data = gen_data.transpose()
capacities = {}

# Capacity of unlimited reservoir
q_in = raw_data["Durchfluss_hm3"].to_numpy()
q_out = np.tile(monthly_dis.loc["original", :].to_numpy(), 40)
storage, _, _, _ = calc_storage(q_in, q_out, initial_storage=0, max_cap=np.inf)
plot_fsa(storage)
cap, _, _, _, _ = calc_capacity(storage)
capacities["original"] = cap

# Plot unlimited reservoir
storage, deficit, overflow, q_out_real = calc_storage(
    q_in, q_out, initial_storage=0, max_cap=np.inf)
plot_storage(q_in, q_out, q_out_real, storage, deficit, overflow, 
             fn_ending="unlimited")

# Plot limited reservoir
storage, deficit, overflow, q_out_real = calc_storage(
    q_in, q_out, initial_storage=0, max_cap=cap)
plot_storage(q_in, q_out, q_out_real, storage, deficit, overflow, 
             fn_ending=str(round(cap, 3)))

# -----------------------------------------
# Calculate capacity for generated data
# -----------------------------------------

for i in range(N_TIMESERIES):
    q_in = gen_data.loc[f"G{str(i+1).zfill(3)}", :]
    q_out = np.tile(monthly_dis.loc[f"G{str(i+1).zfill(3)}", :].to_numpy(), 40)
    storage, _, _, _ = calc_storage(q_in, q_out, initial_storage=0, max_cap=np.inf)
    cap, _, _, _, _ = calc_capacity(storage)
    capacities[f"G{str(i+1).zfill(3)}"] = cap

df_capacities = pd.DataFrame()
df_capacities["Zeitreihe"] = capacities.keys()
df_capacities["Kapazität"] = capacities.values()

df_capacities.to_csv(f"data/{pegelname}_capacities.csv", index=False)
df_capacities.to_latex(f"data/{pegelname}_capacities.tex", index=False)