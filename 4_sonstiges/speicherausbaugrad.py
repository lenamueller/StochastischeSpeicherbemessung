import pandas as pd


def speicherausbaugrad(cap: list[float], mq: float):
    """Calculate beta for different capacities and a given MQ [hm³]."""
    
    result = pd.DataFrame()
    result["Speicherkapazität [hm³]"] = cap
    result["MQ [hm³]"] = [mq]*len(cap)
    result["Speicherausbaugrad [-]"] = [i/mq for i in cap]
    return result