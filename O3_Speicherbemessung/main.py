import numpy as np
import pandas as pd
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import warnings
warnings.filterwarnings("ignore")

from settings import PEGEL_NAMES, N_GEN_YEARS, N_RAW_YEARS, vars
from utils.read import read_raw_data
from sequent_peak_algorithm import soll_abgabe, calc_storage_simulation, calc_capacity, \
    calc_maxima, calc_minima
from fit_capacity import fit_lognv
from reliability import rel_monthly, rel_amount, rel_yearly
from plotting_func import plot_pu, plot_qq, plot_capacities_hist, plot_fsa, \
    plot_monthly_discharge, plot_deficit_overflow, plot_storage_simulation


for PEGEL in PEGEL_NAMES:
    print(f"PEGEL: {PEGEL}")
    
    paths = [
        "O3_Speicherbemessung/results/",
        f"O3_Speicherbemessung/results/{PEGEL}",
        f"O3_Speicherbemessung/results/{PEGEL}/plots"
        ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)

    # ---------------------------------------
    # Rohdaten und generierte Daten einlesen
    # ---------------------------------------

    raw_data = read_raw_data(f"Rohdaten/Daten_{PEGEL}.txt")
    gen_data = pd.read_csv(f"O2_Zeitreihengenerierung/results/{PEGEL}/GenerierteZeitreihen.csv")


    def data(var):
        if var == "original":
            return raw_data["Durchfluss_m3s"].to_list()
        else:
            return gen_data[var].to_list()

    # ---------------------------------------
    # Berechnung der Soll-Abgabe
    # ---------------------------------------

    abgaben = pd.DataFrame()
    for var in vars:
        abgaben[var] = soll_abgabe(timeseries=data(var))
    abgaben.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/SollAbgabe.csv", index=False)

    plot_monthly_discharge(
        df_dis=abgaben,
        fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/MonatlicheAbgabe.png"
        )

    # ---------------------------------------
    # Berechnung der Speicherkapazitäten
    # ---------------------------------------

    capacities = pd.DataFrame()
    c = []
    for var in vars:
        storage, _, _, _ = calc_storage_simulation(
            q_in=data(var),
            q_out_soll=np.tile(abgaben[var].to_numpy(), N_GEN_YEARS),
            initial_storage=0, max_cap=np.inf,
            q_in_convert=True, q_out_soll_convert=False
            )
        cap, cap_min_index, cap_min, cap_max_index, cap_max = calc_capacity(storage)
        c.append(cap)
        
        # Plot FSA only for original time series
        if var == "original":
            
            max_vals, max_indices = calc_maxima(cum_storage=storage)
            min_vals, min_indices = calc_minima(cum_storage=storage, max_indices=max_indices) 
            
            plot_fsa(
                storage=storage, 
                max_vals=max_vals,
                max_indices=max_indices,
                min_vals=min_vals,
                min_indices=min_indices,
                cap=cap,
                cap_min_index=cap_min_index,
                cap_min=cap_min,
                cap_max=cap_max,
                fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/FSA_{var}.png"
                )
                    
    capacities["Zeitreihe"] = vars
    capacities["Speicherkapazität [hm³]"] = c
    capacities.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/Speicherkapazitaet.csv", index=False)

    chist = capacities.iloc[0]["Speicherkapazität [hm³]"]

    # ---------------------------------------
    # Berechnung von K(90%)
    # ---------------------------------------

    gen_capacities = capacities.iloc[1:]
    gen_capacities = gen_capacities["Speicherkapazität [hm³]"].to_list()
    capacities_lognv, fixed_quantiles = fit_lognv(gen_capacities, print_parameters=False)
    capacities_lognv.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/Speicherkapazitaet_Fit.csv", index=False)
    fixed_quantiles.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/Speicherkapazitaet_FixedQuantiles.csv", index=False)

    def cap_fixed_quantile(q: float):
        """Return the capacity for a fixed quantile."""
        pu = fixed_quantiles["Pu [-]"].to_list()
        caps = fixed_quantiles["Kapazität [hm³]"].to_list()
        for i in range(len(pu)):
            if pu[i] == q:
                return caps[i]

    c50 = cap_fixed_quantile(0.5)
    c90 = cap_fixed_quantile(0.9)
    c95 = cap_fixed_quantile(0.95)

    plot_capacities_hist(
        capacities=gen_capacities,
        hist_cap=chist,
        fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/Speicherkapazitaet_Histogramm.png"
        )
        
    plot_pu(
        capacities_sort=sorted(gen_capacities),
        pu_emp=capacities_lognv["empirische Pu [-]"].to_list(),
        pu_theo=capacities_lognv["theoretische Pu [-]"].to_list(),
        cap_90=c90,
        fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/Pu.png"
        )

    plot_qq(
        emp=capacities_lognv["empirische Quantile [hm³]"].to_list(),
        theo=capacities_lognv["theoretische Quantile [hm³]"].to_list(),
        fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/QQ.png"
        )

    # ---------------------------------------
    # Speichersimulation
    # ---------------------------------------

    scenarios = [
        (1, "original", 0.5*c90, c90),
        
        # Variation Anfangsfüllung
        (2, "original", 0.00*c90, c90),
        (3, "original", 0.25*c90, c90),
        (4, "original", 0.75*c90, c90),
        (5, "original", 1.00*c90, c90),
        
        # Variation Kapazität
        (6, "original", 0.5*c50, c50),
        (7, "original", 0.5*c95, c95),
        (8, "original", 0.5*chist, chist),
        
        # fiktive Szenerien
        (9, "original", 50.0, 100.00),
        (10, "original", 0.0, np.inf),
    ]

    for (scen_i, var, initial_storage, max_cap) in scenarios:
        print(f"""\tSimulation {scen_i} von {len(scenarios)}: \t Zeitreihe: {var}, Anfangsfüllung: {round(initial_storage, 3)}, Kapazität: {round(max_cap, 3)}""")
        
        # Simulation
        q_in = data(var)
        q_out_soll = np.tile(abgaben[var].to_numpy(), N_RAW_YEARS)
        storage, deficit, overflow, q_out_ist = calc_storage_simulation(
            q_in=q_in, q_out_soll=q_out_soll, initial_storage=initial_storage, 
            max_cap=max_cap, q_in_convert=True, q_out_soll_convert=False)

        sim = pd.DataFrame()
        sim["Zufluss [hm³]"] = q_in
        sim["Soll-Abgabe [hm³]"] = q_out_soll
        sim["Ist-Abgabe [hm³]"] = q_out_ist
        sim["Speicherinhalt [hm³]"] = storage
        sim["Defizit [hm³]"] = deficit
        sim["Überschuss [hm³]"] = overflow
        sim.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/Simulation_{scen_i}.csv", index=False)

        plot_storage_simulation(
            q_in=q_in,
            q_out=q_out_soll,
            q_out_real=q_out_ist,
            storage=storage,
            deficit=deficit,
            overflow=overflow,
            var=var,
            cap=max_cap,
            initial_storage=initial_storage,
            xticklabels=raw_data["Monat"].to_list(),
            fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/Speichersimulation_{scen_i}.png"
            )

        plot_deficit_overflow(
            deficit=deficit,
            overflow=overflow,
            months=raw_data["Monat"].to_list(),
            fn=f"O3_Speicherbemessung/results/{PEGEL}/plots/Defizit_Ueberschuss_{scen_i}.png"
            )

        # ---------------------------------------
        # Zuverlässigkeitsprüfung
        # ---------------------------------------

        reliability = {
            "Zeitreihe": var,
            "R(Monat, Defizit)": rel_monthly(data=deficit),
            "R(Monat, Überschuss)": rel_monthly(data=overflow),
            "R(Jahr, Defizit)": rel_yearly(data=deficit),
            "R(Jahr, Überschuss)": rel_yearly(data=overflow),
            "R(Menge, Defizit)": rel_amount(data=deficit, soll_abgabe=q_out_soll),
            "R(Menge, Überschuss)": rel_amount(data=overflow, soll_abgabe=q_out_soll)
            }

        result = pd.DataFrame()
        result["Metrik"] = reliability.keys()
        result["Zuverlässigkeit"] = reliability.values()
        result.to_csv(f"O3_Speicherbemessung/results/{PEGEL}/Zuverlaessigkeit_{scen_i}.csv", index=False)