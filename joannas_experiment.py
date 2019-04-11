import os
from pyabf import ABF
from analyze_abf import CurrentStepsData


# ABF_LOCATION = r'C:\Users\mattisj\Desktop\test_files'
ABF_LOCATION = '.\\test\\KG173_02_01_19_0075.abf'
CURRENT_VS_APS_OUTPUT_FILE = r'C:\Users\Colin\Desktop\cuurent_vs_aps.csv'
RHEOBASE_OUTPUT_FILE = r'C:\Users\Colin\Desktop\rheobase.csv'

if os.path.isdir(ABF_LOCATION):
    abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
else:
    abf_files = [ABF_LOCATION]

# Print the files we're analyzing as a sanity check
print('Analyzing the following files:\n{}'.format(abf_files))

# Gathering data from the abf files
current_vs_aps_output = {}
rheobase_output = {}
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

    rheobase_output[filename] = experiment.get_rheobase()

# Writing the data to output file
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


with open(RHEOBASE_OUTPUT_FILE, 'w') as f:
    for filename, rheobase in rheobase_output.items():
        f.write('{}, {}\n'.format(filename, rheobase))
