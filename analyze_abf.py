import os
import logging
from pyabf import ABF
import matplotlib.pyplot as plt
import numpy as np

from trace_analysis import fit_tophat, find_peaks, get_derivative
from config import ABF_LOCATION, AP_THRESHOLD
from sys import float_info

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
            sweep_name=None):
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
        self.sweep_name = sweep_name
        self.analysis_cache = {}  #TODO replace this with functools.lru_cache

        logger.info('{} input units {}'.format(sweep_name, input_signal_units))
        logger.info('{} output units {}'.format(sweep_name, output_signal_units))

        # TODO cache results of analyses already completed

    def __str__(self):
        return 'Data from a single sweep, containing {} data points'.format(len(self.time_steps))

    def fit_input_tophat(
            self, verify=False, max_fitting_passes=2, verify_file='verfication.png'):
        """
        Fit a tophat function to the input signal and cache the result

        :return: (base_level, hat_level, hat_mid, hat_width)
        """
        if 'fit_input_tophat' in self.analysis_cache:
            logger.info('Found tophat params in cache')
            return self.analysis_cache['fit_input_tophat']

        tophat_params = fit_tophat(
            self.time_steps,
            self.input_signal,
            verify=verify,
            max_fitting_passes=max_fitting_passes,
            verify_file=verify_file)
        self.analysis_cache['fit_input_tophat'] = tophat_params

        return tophat_params

    def find_output_peaks(self, threshold=0, verify=False, verify_file='verification.png'):
        """
        Find the peaks in the output and return a list of (t, V) values

        :param threshold:
        :param verify:
        :param verify_file:
        :return: list of tuples (peak time, peak value)
        """
        if 'find_output_peaks' in self.analysis_cache:
            logger.info('Found peak counts in cache')
            return self.analysis_cache['find_output_peaks']

        peaks = find_peaks(
            self.time_steps,
            self.output_signal,
            threshold=threshold,
            verify=verify,
            verify_file=verify_file)
        self.analysis_cache['find_output_peaks'] = peaks

        return peaks

    def get_output_derivative(self):
        """
        return dV/dt values of output signal. List indices correspond to
        self.time_steps

        :return: list of dV/dt values
        """
        if 'get_output_derivative' in self.analysis_cache:
            logger.info('Found output derivative in cache')
            return self.analysis_cache['get_output_derivative']

        dV_dt = get_derivative(self.output_signal, self.time_steps)
        self.analysis_cache['get_output_derivative'] = dV_dt
        return dV_dt

    def get_output_second_derivative(self):
        """
        return d2V/dt2 values of output signal. List indices correspond to
        self.time_steps

        :return: list of d2V/dt2 values
        """
        if 'get_output_second_derivative' in self.analysis_cache:
            logger.info('Found output second derivative in cache')
            return self.analysis_cache['get_output_second_derivative']

        d2V_dt2 = get_derivative(self.get_output_derivative(), self.time_steps)
        self.analysis_cache['get_output_second_derivative'] = d2V_dt2
        return d2V_dt2

    def show_plot(self):
        """
        Plot input vs time and output vs time on overlapping axes
        :return: None
        """
        fig, ax1 = plt.subplots()
        ax1.plot(self.time_steps, self.input_signal, color='red')
        ax2 = ax1.twinx()
        ax2.plot(self.time_steps, self.output_signal)

        plt.show()


class ExperimentData(object):
    """The set of traces in one abf file (a colloquial, not mathematical set)"""
    def __init__(self, abf):
        """

        :param abf: The abf file, as loaded by pyabf
        :param experiment_type: The type of experiment this data is from. One of EXPERIMENT_TYPES
        """
        self.abf = abf
        self.filename = os.path.basename(abf.abfFilePath)
        self.sweep_count = abf.sweepCount
        logger.info('{} sweeps in {}'.format(self.sweep_count, self.filename))

        # Extract all the sweeps into
        self.sweeps = []
        for sweep_num in self.abf.sweepList:
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
                    sweep_name='{}_{}'.format(self.filename[:-4], sweep_num)
                )
            )

    def __str__(self):
        return('Experiment data from {} containing {} sweeps of {} data'.format(
            self.filename, self.sweep_count, self.experiment_type
        ))


class VCTestData(ExperimentData):
    """Functions to get relevant metrics for 'VC test' experiments"""
    def get_input_resistance(self):
        """
        Input resistance: calculate using change in steady state current
        in response to small hyperpolarizing voltage step

        :return:
        """
        resistances = []
        for sweep in self.sweeps:
            voltage_base, applied_voltage, voltage_mid, voltage_width = \
                sweep.fit_input_tophat()  # Voltage base is should always be ~0
            voltage_start = voltage_mid - voltage_width / 2
            logger.info('Current starts at t={}'.format(voltage_start))
            voltage_end = voltage_mid + voltage_width / 2
            start_idx = None
            end_idx = None
            for idx, t in enumerate(sweep.time_steps):
                if t > voltage_start and start_idx is None:
                    start_idx = idx
                if t > voltage_end and end_idx is None:
                    end_idx = idx

            # Measure current for the middle half of the driven part of the sweep
            logger.info('Driven slice is {} to {}'.format(start_idx, end_idx))
            measurement_slice_start = start_idx + (end_idx - start_idx) // 4
            measurement_slice_end = start_idx + 3 * (end_idx - start_idx) // 4

            mean_current_in_measurement_slice = np.mean(
                sweep.output_signal[measurement_slice_start: measurement_slice_end]
            )

            # Measure current for the middle half post drive part of the sweep
            last_idx = len(sweep.input_signal) - 1
            resting_slice_start = end_idx + (last_idx - end_idx) // 4
            resting_slice_end = end_idx + 3 * (last_idx - end_idx) // 4

            mean_current_in_resting_slice = np.mean(
                sweep.output_signal[resting_slice_start: resting_slice_end]
            )

            logger.info('Applied voltage: {} {}'.format(
                applied_voltage, sweep.input_signal_units))
            logger.info('Mean driven current: {} {}'.format(
                mean_current_in_measurement_slice, sweep.output_signal_units))
            logger.info('Resting current is: {} {}'.format(
                mean_current_in_resting_slice, sweep.input_signal_units))

            change_in_current = mean_current_in_measurement_slice - mean_current_in_resting_slice
            resistance = applied_voltage / change_in_current
            resistances.append(resistance)

        return resistances


class CurrentClampGapFreeData(ExperimentData):
    """Functions to get relevant metrics for 'current clamp gap free' experiments"""
    def get_resting_potential(self):
        """
        Resting potential is in the output trace. Just average it. There should
        be jsut one trace

        :return:
        """
        assert len(self.sweeps) == 1

        return np.mean(self.sweeps[0].output_signal)



class CurrentStepsData(ExperimentData):
    """Functions to get relevant metrics for 'current steps' experiments"""

    def get_rheobase(self):
        """
        Get the rheobase - the minimum voltage that elicits at least one peak
        :return:
        """
        drive_voltages = []
        peak_count = []
        for sweep in self.sweeps:
            # Find the voltage of the driving signals for all sweeps
            tophat_params = sweep.fit_input_tophat()
            drive_voltages.append(tophat_params[1] - tophat_params[0])

            # count the peaks
            peak_count.append(len(sweep.find_output_peaks()))

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

        TODO What counts as sustained firing? First sweep with 11+ peaks?

        :return: (isi_1/isi_10, isi_1/isi_n)
        """
        peaks_by_sweep = []
        for sweep in self.sweeps:
            peaks_by_sweep.append(sweep.find_output_peaks())

        for peaks in peaks_by_sweep:
            if len(peaks) >= 11:
                isi_1 = peaks[1][0] - peaks[0][0]
                isi_10 = peaks[10][0] - peaks[9][0]
                isi_n = peaks[-1][0] - peaks[-2][0]
                return isi_1/isi_10, isi_1/isi_n
        else:
            logger.info('Data had no sweeps with sustained firing')
            return None, None

    def get_max_steady_state_firing_frequency(self):
        """
         Max steady state firing frequency:
         max mean firing frequency in response to current injection with no
         failures (AP amplitude at least 40mV and overshooting 0mV)

        # TODO what should be returned. frequency. Driving voltage eliciting that frequency?
        # TODO do we have to check for "missing" peaks
        :return: frequency, inverse of timesteps units
        """
        peaks_by_sweep = []
        for sweep in self.sweeps:
            # Set threshold = 0 to fulfil "overshooting 0mV" criterion
            peaks_by_sweep.append(sweep.find_output_peaks(threshold=0))

        frequencies = []
        for peaks in peaks_by_sweep:
            invalid_sweep = False
            if len(peaks) < 2:
                logger.info('Not enough peaks in sweep to calculate a frequency')
                invalid_sweep = True

            for peak in peaks:
                if peak[1] < 40:  # Fulfil amplitude > 40mV criterion
                    logger.info('One of the peaks was too low')
                    invalid_sweep = True

            if not invalid_sweep:
                frequency = len(peaks)/(peaks[-1][0] - peaks[0][0])
                frequencies.append(frequency)

        return max(frequencies)

    def get_max_instantaneous_firing_frequency(self):
        """
        Max instantaneous firing frequency:
        inverse of smallest interspike interval in response to current
        injection (AP amplitude at least 40mV and overshooting 0mV)


        :return:
        """
        minimum_peak_interval = float_info.max
        for sweep in self.sweeps:
            # Set threshold = 0 to fulfil "overshooting 0mV" criterion
            peaks = sweep.find_output_peaks(threshold=0)
            for idx, peak in enumerate(peaks[1:]):
                peak_interval = peak[0] - peaks[idx][0]
                logger.debug('{}, {} to {} peak interval is {}'.format(
                    sweep.sweep_name, idx, idx + 1, peak_interval
                ))
                if peak_interval < minimum_peak_interval:
                    logger.info(
                        'Found a smaller peak interval in {}, peaks {} to {}'.format(
                            sweep.sweep_name, idx, idx + 1
                        )
                    )
                    minimum_peak_interval = peak_interval

        return 1/minimum_peak_interval

    def _get_ap_threshold_1_details(self):
        """
        AP threshold #1:
        for first spike obtained at suprathreshold current injection, the
        voltage at which first derivative (dV/dt) of the AP waveform reaches
        10V/s = 10000mV/s

        :return: sweep number of threshold measurement, V or threshold, t of threshold
        """
        gradient_threshold = 10000  # V/s  TODO handle units properly

        # iterate through sweeps and peaks until we find the first peak. We will
        # return a result based on that peak.
        for sweep_num, sweep in enumerate(self.sweeps):
            for peak in sweep.find_output_peaks():
                dVdt = sweep.get_output_derivative()
                for idx, gradient in enumerate(dVdt):
                    if gradient > gradient_threshold:
                        # return the value of the voltage at the timestamp that
                        # we cross the threshold in gradient
                        logger.info('Found AP threshold 1 in {}'.format(sweep.sweep_name))
                        return sweep_num, sweep.output_signal[idx], sweep.time_steps[idx]

    def _get_ap_threshold_1_time(self):
        """


        :return:
        """
        return self._get_ap_threshold_1_details()[2]

    def get_ap_threshold_1(self):
        """

        :return:
        """
        return self._get_ap_threshold_1_details()[1]

    def get_ap_threshold_2(self):
        """
        AP threshold #2:
        for first spike obtained at suprathreshold current injection, the
        voltage at which second derivative (d2V/dt2) reaches 5% of maximal
        value

        # TODO this return value is dubious for the test data
        # TODO it occurs outside the driving voltage step.
        # TODO Would we get a reasonable result by looking at the slice in the
        # TODO driving tophat

        :return:
        """
        raise NotImplementedError
        # for sweep in self.sweeps:
        #     for peak in sweep.find_output_peaks():
        #         d2V_dt2 = sweep.get_output_second_derivative()
        #         d2V_dt2_peaks = find_peaks(sweep.time_steps, d2V_dt2)
        #         max_first_d2V_dt2_peak = d2V_dt2_peaks[0][1]
        #         for idx, d2V_dt2_value in enumerate(d2V_dt2):
        #             if d2V_dt2_value > 0.05 * max_first_d2V_dt2_peak:
        #                 logger.info('Found AP threshold 2 in {}'.format(sweep.sweep_name))
        #                 logger.info('Found AP threshold 2 at t={}'.format(sweep.time_steps[idx]))
        #                 return sweep.output_signal[idx]

    def get_ap_rise_time(self):
        """
        AP rise time:
        for first spike obtained at suprathreshold current injection, time
        from AP threshold 1 to peak

        :return:
        """
        sweep_num, ap_threshold, ap_threshold_time = self._get_ap_threshold_1_details()
        sweep = self.sweeps[sweep_num]
        ap_peak_time = sweep.find_output_peaks()[0][0]
        return ap_peak_time - ap_threshold_time

    def get_ap_amplitude(self):
        """
        AP amplitude:
        for first spike obtained at suprathreshold current injection, change
        in mV from AP threshold #1 to peak

        :return:
        """
        sweep_num, ap_threshold_voltage, ap_threshold_time = \
            self._get_ap_threshold_1_details()
        sweep = self.sweeps[sweep_num]
        ap_peak_voltage = sweep.find_output_peaks()[0][1]
        return ap_peak_voltage - ap_threshold_voltage

    def get_ap_half_width(self):
        """
        AP half-width:
        for first spike obtained at suprathreshold current injection, width
        of the AP (in ms) at 1/2 maximal amplitude, using AP threshold #1 and
        AP amplitude

        :return:
        """
        for sweep in self.sweeps:
            for peak in sweep.find_output_peaks():
                logger.info('Found first peak in {}'.format(sweep.sweep_name))
                half_peak_voltage = 0.5 * (peak[1] - self.get_ap_threshold_1())
                peak_time = peak[0]
                peak_index = list(sweep.time_steps).index(peak_time)
                logger.info('Peak time is {}'.format(peak_time))
                logger.info('Peak is at data point number {}'.format(peak_index))

                voltage_at_time = dict(zip(sweep.time_steps, sweep.output_signal))
                # Iterate back through the data to find the half peak time
                for time_step in sweep.time_steps[peak_index::-1]:
                    if voltage_at_time[time_step] < half_peak_voltage:
                        logger.info('Found peak start at {}'.format(time_step))
                        peak_start = time_step
                        break

                # Iterate forward through the data to find the half peak time
                for time_step in sweep.time_steps[peak_index:]:
                    if voltage_at_time[time_step] < half_peak_voltage:
                        logger.info('Found peak end at {}'.format(time_step))
                        peak_end = time_step
                        break

                return peak_end - peak_start

    def get_input_resistance(self):
        """
        Input resistance #1:
        calculate using slope of the linear fit to the plot of the V-I
        relation from subthreshold current steps at/around resting potential

        # TODO do this from the last sweep before threshold?
        # TODO Average gradient for first 20ms?
        # TODO What is resting potential?
        # TODO Need to discuss this one.

        :return:
        """
        raise NotImplementedError
        # Find the sweep which elicits the first AP
        for sweep_num, sweep in enumerate(self.sweeps):
            if len(sweep.find_output_peaks()) > 0:
                first_suprathreshold_sweep = sweep_num
                break

        last_subthreshold_sweep_num = first_suprathreshold_sweep - 1

        # Find time of current step in last subthreshold peak
        last_subthreshold_sweep = self.sweeps[last_subthreshold_sweep_num]
        tophat_params = last_subthreshold_sweep.fit_input_tophat()
        current_step_time = tophat_params[2] - tophat_params[3] / 2
        drive_current = tophat_params[1]

        # Get the derivative of the output function
        dV_dt = last_subthreshold_sweep.get_output_derivative()

    def plot_v_vs_i(self, sweep_num):
        """
        Just for testing

        :param sweep_num:
        :return:
        """
        sweep = self.sweeps[sweep_num]
        fig, ax1 = plt.subplots()
        ax1.plot(sweep.output_signal, sweep.input_signal)
        plt.show()



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

        ############################   CURRENT STEPS
        experiment = CurrentStepsData(abf)
        for i, sweep in enumerate(experiment.sweeps):
            if len(sweep.find_output_peaks()) == 0:
                experiment.plot_v_vs_i(i)


        #
        # rheobase = experiment.get_rheobase()
        # print('Rheobase of {} is {}mV'.format(experiment.filename, rheobase))
        #
        # sfa = experiment.get_spike_frequency_adaptation()
        # print('SFA is {}'.format(sfa))
        #
        # max_ssff = experiment.get_max_steady_state_firing_frequency()
        # print('Max steady state firing frequency is {}'.format(max_ssff))
        #
        # max_iff = experiment.get_max_instantaneous_firing_frequency()
        # print('Max instantaneous firing frequency is {}'.format(max_iff))
        #
        # ap_threshold_1 = experiment.get_ap_threshold_1()
        # print('AP threshold 1 is {}'.format(ap_threshold_1))
        #
        # try:
        #     ap_threshold_2 = experiment.get_ap_threshold_2()
        #     print('AP threshold 2 is {}'.format(ap_threshold_2))
        # except NotImplementedError:
        #     logger.warning("I don't know how to do that")
        #
        # ap_half_width = experiment.get_ap_half_width()
        # print('AP half width is {}'.format(ap_half_width))
        ############################   \CURRENT STEPS

        ############################   VC TEST
        # experiment = VCTestData(abf)
        # print('time units: {}, input units: {}, output units: {}'.format(
        #     experiment.sweeps[0].time_steps_units,
        #     experiment.sweeps[0].input_signal_units,
        #     experiment.sweeps[0].output_signal_units
        # ))
        # input_resistances = experiment.get_input_resistance()
        # print('Input resistances: {}'.format(input_resistances))
        # print('Input resistance is {} {}/{}'.format(
        #     np.mean(input_resistances),
        #     experiment.sweeps[0].input_signal_units,
        #     experiment.sweeps[0].output_signal_units
        # ))
        # print('mV / pA is GOhm')
        ############################   \VC TEST

        ############################   current clamp gap free
        # experiment = CurrentClampGapFreeData(abf)
        # resting_potential = experiment.get_resting_potential()
        # print('Resting potential is: {} {}'.format(
        #     resting_potential, experiment.sweeps[0].output_signal_units))
        # for sweep in experiment.sweeps:
        #     sweep.show_plot()
        #     print('Input: {}'.format(sweep.input_signal_units))
        #     print('Output: {}'.format(sweep.output_signal_units))

        ############################   /current clamp gap free