pegelname = "Klingenthal"

# --------------------------------------
# Pfade
# --------------------------------------

image_path = "images/"
report_path = "reports/"
fn_results = f"data/Daten_{pegelname}_results.csv"

# --------------------------------------
# Variablennamen
# --------------------------------------

var_remapper = {
        "Durchfluss_m3s": "Rohdaten",
        "trendber": "Trendbereinigt",
        "saisonfigur_mean": "Saisonale Komp. (Mittel)",
        "saisonfigur_std": "Saisonale Komp. (Standardabweichung)",
        "saisonber": "Saisonbereinigt",
        "normier": "Normiert",
        "autokorr_saisonfigur": "Autokorr. (Saisonfigur)",
        "autokorr": "Autokorr. Komp.",
        "zufall": "Zufallskomp."
    }

MONTH = {
    1: "Januar",
    2: "Februar",
    3: "März",
    4: "April",
    5: "Mai",
    6: "Juni",
    7: "Juli",
    8: "August",
    9: "September",
    10: "Oktober",
    11: "November",
    12: "Dezember"
}

MONTH_ABB = ["N", "D", "J", "F", "M", "A", "M", "J", "J", "A", "S", "O"]
MONTH_HYD_YEAR = [11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9 ,10]
MONTH_HYD_YEAR_TXT = ["Nov", "Dez", "Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep" ,"Okt"]

# --------------------------------------
# TU CD colors
# --------------------------------------

tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)

# --------------------------------------
# Thomas Fiering model
# --------------------------------------

T = 12                  # max i index
N_TIMESERIES = 100       # Anzahl generierte Zeitreihen
N_YEARS = 40            # Anzahl Jahre pro generierte Zeitreihe

# --------------------------------------
# Folgescheitelalgorithmus (FSA)
# --------------------------------------

# Sekunden pro Monat
SEC_PER_MONTH = (365/12)*24*60*60   

# Speicherausgleichsgrad
ALPHA = 0.7

# Monatsanteile [%]
ABGABEN = {                         
    11: 7,
    12: 7,
    1: 5,
    2: 4,
    3: 3,
    4: 7,
    5: 7,
    6: 10,
    7: 12,
    8: 14,
    9: 14,
    10:10
}