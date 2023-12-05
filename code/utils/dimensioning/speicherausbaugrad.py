import numpy as np
import pandas as pd


def speicherausbaugrad(var: str, mq: float):
    """Calculate beta for a given MQ and different capacities."""
    
    print("\n--------------------------------------")
    print(f"\nSpeicherausgleichsgrad für Zeitreihe {var}")
        
    cap = np.arange(5, 55, 5)
    beta = [i/mq for i in cap]
    
    pd.DataFrame(data={"Kapazität [hm³]": cap, f"Speicherausbaugrad für MQ {round(mq, 3)} hm³ [-]": beta}).to_csv(
        f"data/speicherausbaugrad_{var}.csv", index=False)

    pd.DataFrame(data={"Kapazität [hm³]": cap, f"Speicherausbaugrad für MQ {round(mq, 3)} hm³ [-]": beta}).to_latex(
        f"data/speicherausbaugrad_{var}.tex", index=False, float_format="%.3f")
    
    print(f"\n-> data/speicherausbaugrad_{var}.csv")