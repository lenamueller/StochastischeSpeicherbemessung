from read_data import read_data
from process_data import accumulate_yearly
from plot_data import visualize_discharge


fn = "Daten_Ammelsdorf.txt"
raw = read_data(fn)
print(raw)

visualize_discharge(
    monthly_data=raw,
    yearly_data=accumulate_yearly(raw),
    fn="Durchfluss_Ammelsdorf.png"
)