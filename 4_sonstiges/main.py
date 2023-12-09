import numpy as np
import os
import sys
sys.path.insert(1, '/home/lena/Dokumente/FGB/StochastischeSpeicherbemessung/')

from speicherausbaugrad import speicherausbaugrad
from settings import MQ_HM3, PEGEL_NAMES


for PEGEL in PEGEL_NAMES:

    paths = [
        "4_sonstiges/results/",
        f"4_sonstiges/results/{PEGEL}"
        ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


    # ---------------------------------------
    # Speicherausbaugrad
    # ---------------------------------------

    beta = speicherausbaugrad(cap=np.arange(5, 55, 5), mq=MQ_HM3[PEGEL])
    beta.to_csv(f"4_sonstiges/results/{PEGEL}/Speicherausbaugrad.csv", index=False)
