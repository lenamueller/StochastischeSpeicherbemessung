pegelname = "Klingenthal"

image_path = "images/"
report_path = "reports/"

fn_results = f"data/Daten_{pegelname}_results.csv"

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

tu_darkblue = (0/255, 48/255, 93/255)
tu_mediumblue = (0/255, 105/255, 180/255)
tu_grey = (114/255, 119/255, 119/255)
tu_red = (181/255, 28/255, 28/255)