from pyabf import ABF
from analyze_abf import CurrentStepsData


ABF_LOCATION = r'C:\Users\mattisj\Desktop\9-Patching\GC patching Cl free internal\adult WT\JM20190327_0034.abf'

abf = ABF(ABF_LOCATION)
experiment = CurrentStepsData(abf)

for sweep in experiment.sweeps:
    ap_count = sweep.find_output_peaks()
    base_level, hat_level, hat_mid, hat_width = sweep.fit_input_tophat(verify=True)
    current_step = hat_level - base_level
    print('Current: {}pA, APs: {}'.format(current_step, len(ap_count)))


