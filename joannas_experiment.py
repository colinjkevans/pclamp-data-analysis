import os
from collections import defaultdict
from pyabf import ABF
from analyze_abf import CurrentStepsData


ABF_LOCATION = r'C:\Users\mattisj\Desktop\test_files'
OUTPUT_FILE = r'C:\Users\mattisj\Desktop\output_file.csv'


if os.path.isdir(ABF_LOCATION):
    abf_files = [os.path.join(ABF_LOCATION, f) for f in os.listdir(ABF_LOCATION) if f.endswith('.abf')]
else:
    abf_files = [ABF_LOCATION]

# Print the files we're analyzing as a sanity check
print('Analyzing the following files:')
for f in abf_files:
    print(f)

# Gathering data from the abf files
output = {}
for filepath in abf_files:
    abf = ABF(filepath)
    experiment = CurrentStepsData(abf)
    filename = os.path.basename(filepath)
    print('Analyzing {}'.format(filename))
    output[os.path.basename(filename)] = []

    print('{} contains {} sweeps'.format(filename, len(experiment.sweeps)))
    for sweep in experiment.sweeps:
        aps = sweep.find_output_peaks()
        base_level, hat_level, hat_mid, hat_width = sweep.fit_input_tophat()
        current_step = hat_level - base_level
        output[filename].append((current_step, len(aps)))
        print('Current: {}pA, APs: {}'.format(current_step, len(aps)))

# Writing the data to output file
max_sweeps = len(max(output.values(), key=lambda x: len(x)))
filenames = sorted(output.keys())
print('max_sweeps is {}'.format(max_sweeps))
with open(OUTPUT_FILE, 'w') as f:
    f.write(','.join(['{}, '.format(s) for s in filenames]))
    f.write('\n')

    for i in range(max_sweeps):
        for filename in filenames:
            try:
                f.write('{},{},'.format(*output[filename][i]))
            except IndexError:
                f.write(',,')
        f.write('\n')



