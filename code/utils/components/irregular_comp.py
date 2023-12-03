import pandas as pd

from utils.plotting  import plot_components, plot_characteristics, pairplot


def irregular_comp(df: pd.DataFrame):
    print("\n--------------------------------------")
    print("\nBestimmung der Zufallskomponente\n")
    
    # calculate components
    df["zufall"] = df["normiert"] - df["autokorr"]

    # plot comparison of components    
    plot_components(df)
    plot_characteristics(df)
    pairplot(df)
