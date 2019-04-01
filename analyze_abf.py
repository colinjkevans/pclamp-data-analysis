import os
import logging
import numpy as np
from pyabf import ABF

from trace_analysis import fit_tophat, find_peaks
from config import ABF_LOCATION, AP_THRESHOLD

ABF_FILE_EXTENSION = '.abf'
EXPERIMENT_TYPE_CURRENT_STEPS = 'current_steps'

EXPERIMENT_TYPES = [
    EXPERIMENT_TYPE_CURRENT_STEPS
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Sweep(object):
    """
    The data related to one sweep in an `ExperimentData` (i.e. in an abf file)
    """

    def __init__(
            self,
            time_steps,
            input_signal,
            output_signal,
            time_steps_units=None,
            input_signal_units=None,
            output_signal_units=None,
            metadata=None):
        """

        :param time_steps: The time steps of the sweep (list)
        :param input_signal: The input signal values (list)
        :param output_signal: The output signal values (list)
        :param time_steps_units: units of `time`
        :param input_signal_units: units of `input_signal`
        :param output_signal_units: units of `output_signal`
        :param metadata: Optional metadata about the sweep (dict)
        """
        # time, input signal and output signal must have the same number of
        # data points
        assert len(time_steps) == len(input_signal)
        assert len(time_steps) == len(output_signal)

        self.time_steps = time_steps
        self.input_signal = input_signal
        self.output_signal = output_signal
        self.time_steps_units = time_steps_units
        self.input_signal_units = input_signal_units
        self.output_signal_units = output_signal_units
        self.metadata = {} if metadata is None else metadata
        self.analysis_cache = {}

        # TODO cache results of analyses already completed

    def __str__(self):
        return 'Data from a single sweep, containing {} data points'.format(len(self.time_steps))


class ExperimentData(object):
    """The set of traces in one abf file (a colloquial, not mathematical set)"""
    def __init__(self, abf, experiment_type=None):
        """

        :param abf: The abf file, as loaded by pyabf
        :param experiment_type: The type of experiment this data is from. One of EXPERIMENT_TYPES
        """
        self.abf = abf
        self.filename = os.path.basename(abf.abfFilePath)
        self.sweep_count = abf.sweepCount
        if experiment_type is None:
            # TODO Try to get it out of abf metadata (tagComments perhaps?)
            self.experiment_type = EXPERIMENT_TYPE_CURRENT_STEPS  # Default to "current steps"
        else:
            self.experiment_type = experiment_type

        # Extract all the sweeps into
        self.sweeps = []
        for sweep_num in self.abf.sweepList:
            if experiment_type == EXPERIMENT_TYPE_CURRENT_STEPS:
                self.abf.setSweep(sweep_num, channel=1)
                input_signal = self.abf.sweepY
                input_signal_units = self.abf.sweepUnitsY

                self.abf.setSweep(sweep_num, channel=0)
                output_signal_units = self.abf.sweepUnitsY
                output_signal = self.abf.sweepY

                time_steps = self.abf.sweepX
                time_units = self.abf.sweepUnitsX
                self.sweeps.append(
                    Sweep(
                        time_steps,
                        input_signal,
                        output_signal,
                        time_units,
                        input_signal_units,
                        output_signal_units,
                        metadata={'filename': self.filename, 'index': sweep_num}
                    )
                )

    def __str__(self):
        return('Experiment data from {} containing {} sweeps of {} data'.format(
            self.filename, self.sweep_count, self.experiment_type
        ))


class CurrentStepsData(ExperimentData):
    """Subclass of Experiment data including functions to get relevant metrics"""
    def get_rheobase(self):
        """
        Get the rheobase - the minimum voltage that elicits at least one peak
        :return:
        """

        drive_voltages = []
        peak_count = []
        for sweep in self.sweeps:
            # Find the voltage of the driving signals for all sweeps
            tophat_params = fit_tophat(
                sweep.time_steps,
                sweep.input_signal,
            )
            drive_voltages.append(tophat_params[1] - tophat_params[0])

            # count the peaks
            peak_count.append(len(find_peaks(sweep.time_steps, sweep.output_signal)))

        peaks_at_voltage = list(zip(peak_count, drive_voltages))
        peaks_at_voltage.sort(key=lambda x: x[1])  # Sort by voltage
        for sweep in peaks_at_voltage:  # Iterate over sorted sweeps
            if sweep[0] > 0:
                return sweep[1]  # return voltage of fist sweep with a peak
        else:
            logger.info('No sweep had a peak in this experiment')
            return None  # return None if there are no APs in the experiment

    def get_spike_frequency_adaptation(self):
        """
        Spike frequency adaptation: ratio of first to 10th interspike interval
        (ISI1/ISI10) and ratio of first to last interspike interval
        (ISI1/ISIn) for the first suprathreshold current injection to elicit
        sustained firing

        TODO What is sustained firing?

        :return: (isi_1/isi_10, isi_1/isi_n)
        """
        peaks_by_sweep = []
        for sweep in self.sweeps:
            peaks_by_sweep.append(find_peaks(sweep.time_steps, sweep.output_signal))

        peaks_by_sweep.sort(key=lambda peaks: len(peaks))
        for peaks in peaks_by_sweep:
            if len(peaks) >= 11:
                isi_1 = peaks[1][0] - peaks[0][0]
                isi_10 = peaks[10][0] - peaks[9][0]
                isi_n = peaks[-1][0] - peaks[-2][0]
                return isi_1/isi_10, isi_1/isi_n
        else:
            logger.info('Data had no sweeps with sustained firing')
            return None


def get_file_list(abf_location):
    """
    Figure out which file(s) to analyze based ont eh location(s) specified in the config file.
    Locations can be strings with the name of a file / folder path, or a list of strings.

    :param abf_location: The abf location as extracted from config
    :return:
    """
    if isinstance(abf_location, list):
        abf_location_list = abf_location
    else:
        abf_location_list = [abf_location]

    abf_files = []
    error = False
    for path in abf_location_list:
        if not os.path.exists(path):
            logger.error('File {} not found'.format(path))
            error = True
        if os.path.isdir(path):
            abf_files += [f for f in os.listdir(path) if f.endswith(ABF_FILE_EXTENSION)]
        elif os.path.isfile(path):
            if path.endswith(ABF_FILE_EXTENSION):
                abf_files.append(path)

    if error:
        raise ValueError('Specified location for abd files does not exist')

    logger.info('Found {} files to analyze'.format(len(abf_files)))
    return abf_files


if __name__ == '__main__':

    abf_files = get_file_list(ABF_LOCATION)
    for filename in abf_files:
        logger.info('Filename: {}'.format(filename))
        abf = ABF(filename)
        # for field in dir(abf):
        #     print('#################### {}'.format(field))
        #     exec('print(abf.{})'.format(field))

        experiment = CurrentStepsData(abf, EXPERIMENT_TYPE_CURRENT_STEPS)
        print(experiment)
        print(experiment.sweeps[0])

        # rheobase = experiment.get_rheobase()
        # print('Rheobase of {} is {}mV'.format(experiment.filename, rheobase))

        sfa = experiment.get_spike_frequency_adaptation()
        print('SFA is {}'.format(sfa))

        for sweep in experiment.sweeps:
            pass
            # fit_tophat(
            #     sweep.time_steps,
            #     sweep.input_signal,
            #     verify='offline',
            #     verify_file='{}_{}.png'.format(sweep.metadata['filename'][:-4], sweep.metadata['index'])
            # )
            # find_peaks(
            #     sweep.time_steps,
            #     sweep.output_signal,
            #     verify='offline',
            #     verify_file='{}_{}_peaks.png'.format(sweep.metadata['filename'][:-4], sweep.metadata['index'])
            # )
        exit()
        # input: sweepY, channel=1
        # output: sweepY, channel=0
        # time: sweepX?
        # print(dir(abf))
        # print(abf.sweepList)
        logger.info(dir(abf))
        logger.info('File contains {} sweeps'.format(abf.sweepCount))

        for sweep_num in abf.sweepList:
            logger.info('Sweep number {} has {} data points'.format(sweep_num, len(abf.sweepY)))
            abf.setSweep(sweep_num, channel=1)
            drive_signal = abf.sweepY

            abf.setSweep(sweep_num, channel=0)
            response_signal = abf.sweepY

            time = abf.sweepX

            base_level, hat_level, hat_mid, hat_width = fit_tophat(time, drive_signal)
            actions_potentials = count_peaks(
                time,
                response_signal,
                threshold=AP_THRESHOLD,
                verify='offline',
                verify_file='{}.png'.format(sweep_num)
            )

            print('dV={}, AP count: {}'.format(hat_level - base_level, len(actions_potentials)))



