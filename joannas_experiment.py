import os
from pyabf import ABF
from analyze_abf import CurrentStepsData, VCTestData

ABF_LOCATION = r'C:\Users\mattisj\Desktop\9-Patching\GC adult Scn1a\IC steps'
CURRENT_VS_APS_OUTPUT_FILE = r'C:\Users\mattisj\Desktop\9-Patching\GC adult Scn1a\current_vs_aps.csv'
ANALYSIS_OUTPUT_FILE = r'C:\Users\mattisj\Desktop\9-Patching\GC adult Scn1a\IC steps GC adult Scn1a.csv'

# RHEOBASE_OUTPUT_FILE = r'C:\Users\mattisj\Desktop\9-Patching\GC adult WT\rheobase.csv'

if os.path.isdir(ABF_LOCATION):
    abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
else:
    abf_files = [ABF_LOCATION]

# Print the files we're analyzing as a sanity check
print('Analyzing the following files:\n{}'.format(abf_files))

# Gathering data from the abf files
current_vs_aps_output = {}
ap_half_width_output = {}
ap_amplitude_output = {}
ap_rise_time_output = {}
ap_threshold_1_output = {}
rheobase_output = {}
max_instantaneous_firing_frequency_output = {}
max_steady_state_firing_frequency_output = {}
spike_frequency_adaptation_output = {}

for filepath in abf_files:
    abf = ABF(filepath)
    experiment = CurrentStepsData(abf)

    filename = os.path.basename(filepath)
    print('Analyzing {}'.format(filename))
    current_vs_aps_output[os.path.basename(filename)] = []

    print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))

    current_vs_aps_output[filename] = list(zip(
        experiment.get_current_step_sizes(), experiment.get_ap_counts()
    ))

    # individual AP characteristics
    ap_half_width_output[filename] = experiment.get_ap_half_width()
    ap_amplitude_output[filename] = experiment.get_ap_amplitude()
    ap_rise_time_output[filename] = experiment.get_ap_rise_time()
    ap_threshold_1_output[filename] = experiment.get_ap_threshold_1()

    # characteristics of spike train
    rheobase_output[filename] = experiment.get_rheobase()
    max_instantaneous_firing_frequency_output[filename] = experiment.get_max_instantaneous_firing_frequency()
    max_steady_state_firing_frequency_output[filename] = experiment.get_max_steady_state_firing_frequency(verify=True)
    spike_frequency_adaptation_output[filename] = experiment.get_spike_frequency_adaptation()

# Writing the % spikes per current step data to output file
max_sweeps = len(max(current_vs_aps_output.values(), key=lambda x: len(x)))
filenames = sorted(current_vs_aps_output.keys())
print('max_sweeps is {}'.format(max_sweeps))
with open(CURRENT_VS_APS_OUTPUT_FILE, 'w') as f:
    f.write(','.join(['{}, '.format(s) for s in filenames]))
    f.write('\n')

    for i in range(max_sweeps):
        for filename in filenames:
            try:
                f.write('{},{},'.format(*current_vs_aps_output[filename][i]))
            except IndexError:
                f.write(',,')
        f.write('\n')

# Writing the additional analysis to output file
with open(ANALYSIS_OUTPUT_FILE, 'w') as f:
    f.write("filename, AP_half_width, AP_amp, AP_rise_time, AP_thresh, rheobase, max_IFF, max_SSFF, SFA10, SFAn\n")
    for filename in ap_half_width_output:
        f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
            filename,
            ap_half_width_output[filename],
            ap_amplitude_output[filename],
            ap_rise_time_output[filename],
            ap_threshold_1_output[filename],
            rheobase_output[filename],
            max_instantaneous_firing_frequency_output[filename],
            max_steady_state_firing_frequency_output[filename],
            spike_frequency_adaptation_output[filename]
        ))

# with open(RHEOBASE_OUTPUT_FILE, 'w') as f:
#     for filename, rheobase in rheobase_output.items():
#         f.write('{}, {}\n'.format(filename, rheobase))