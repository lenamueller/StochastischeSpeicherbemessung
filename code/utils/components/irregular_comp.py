import pandas as pd


def irregular_comp(df: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nBestimmung der Zufallskomponente\n")
    
    # calculate components
    df["zufall"] = df["normiert"] - df["autokorr"]