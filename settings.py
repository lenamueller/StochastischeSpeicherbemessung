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

MQ_M3S = 1.188
MQ_HM3 = MQ_M3S * 60*60*24*365/1000000

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

# --------------------------------------
# TU CD colors
# --------------------------------------

tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)