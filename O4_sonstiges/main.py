import numpy as np
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from speicherausbaugrad import speicherausbaugrad
from settings import MQ_HM3, PEGEL_NAMES


for PEGEL in PEGEL_NAMES:

    paths = [
        "O4_sonstiges/results/",
        f"O4_sonstiges/results/{PEGEL}"
        ]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)


    # ---------------------------------------
    # Speicherausbaugrad
    # ---------------------------------------

    beta = speicherausbaugrad(cap=np.arange(5, 55, 5), mq=MQ_HM3[PEGEL])
    beta.to_csv(f"O4_sonstiges/results/{PEGEL}/Speicherausbaugrad.csv", index=False)
