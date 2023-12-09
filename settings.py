# --------------------------------------
# Pegelinformationen
# --------------------------------------

PEGEL_NAMES = [
    "Ammelsdorf",
    "Klingenthal",
    "Rothenthal",
    "Seifhennersdorf",
    "Tannenberg",
    "Wiesa",
    ]

MQ_M3S = {
    "Ammelsdorf":       0.946,
    "Klingenthal":      1.188,
    "Rothenthal":       1.320,
    "Seifhennersdorf":  0.885,
    "Tannenberg":       1.450,
    "Wiesa":            1.610,
    }

MQ_HM3 = {
    "Ammelsdorf":       MQ_M3S["Ammelsdorf"] * 60*60*24*365/1000000,
    "Klingenthal":      MQ_M3S["Klingenthal"] * 60*60*24*365/1000000,
    "Rothenthal":       MQ_M3S["Rothenthal"] * 60*60*24*365/1000000,
    "Seifhennersdorf":  MQ_M3S["Seifhennersdorf"] * 60*60*24*365/1000000,
    "Tannenberg":       MQ_M3S["Tannenberg"] * 60*60*24*365/1000000,
    "Wiesa":            MQ_M3S["Wiesa"] * 60*60*24*365/1000000,
    }

# --------------------------------------
# Sonstiges
# --------------------------------------

SEC_PER_MONTH = (365/12)*24*60*60   

# --------------------------------------
# Zeitreihengenerierung
# --------------------------------------

N_GEN_TIMESERIES = 100
N_GEN_YEARS = 80
N_RAW_YEARS = 40
SPEICHERAUSGLEICHSGRAD = 0.7

# Monatsanteile der Soll-Abgaben [%]
ABGABEN = {                         
    11: 7, 12: 7, 1: 5, 2: 4, 3: 3, 4: 7,
    5: 7, 6: 10, 7: 12, 8: 14, 9: 14, 10:10
    }

vars = ["original"] + [f"G{str(i+1).zfill(3)}" for i in range(N_GEN_TIMESERIES)]