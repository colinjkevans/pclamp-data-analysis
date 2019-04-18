import os
import logging
from functools import lru_cache
from collections import defaultdict
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
logger.setLevel(logging.WARNING)

class InvalidSweep(Exception):
    pass


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

    def get_output_derivative(self, verify=False):
        """
        return d/dt values of output signal. List indices correspond to
        self.time_steps

        :return: list of dV/dt values
        """
        if 'get_output_derivative' in self.analysis_cache:
            logger.info('Found output derivative in cache')
            return self.analysis_cache['get_output_derivative']

        d_dt_output = get_derivative(self.output_signal, self.time_steps)
        self.analysis_cache['get_output_derivative'] = d_dt_output

        if verify:
            plt.close('all')
            plt.figure(figsize=(16, 10))
            fig, ax1 = plt.subplots()
            ax1.plot(self.time_steps, self.output_signal)

            ax2 = ax1.twinx()
            ax2.plot(self.time_steps, d_dt_output)
            plt.show()

        return d_dt_output

    def get_output_second_derivative(self):
        """
        return d2V/dt2 values of output signal. List indices correspond to
        self.time_steps

        :return: list of d2V/dt2 values
        """
        if 'get_output_second_derivative' in self.analysis_cache:
            logger.info('Found output second derivative in cache')
            return self.analysis_cache['get_output_second_derivative']

        d2_dt2_output = get_derivative(self.get_output_derivative(), self.time_steps)
        self.analysis_cache['get_output_second_derivative'] = d2_dt2_output
        return d2_dt2_output

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


class CurrentStepsSweep(Sweep):
    """
    Functions to extract relevant data from a current steps sweep
    """

    def __init__(self, good_ap_amplitude=0, failed_ap_amplitude=-20, *args, **kwargs):
        self._has_failed_aps = None
        self.good_ap_amplitude = good_ap_amplitude
        self.failed_ap_amplitude = failed_ap_amplitude
        super().__init__(*args, **kwargs)

    def has_aps(self):
        return len(self.get_aps()) > 0

    def has_failed_aps(self):
        """
        Boolean specifying whether the sweep has failed aps. See get_aps for
        definition.
        """
        if self._has_failed_aps is None:
            # Call get_aps with default args to ensure _has_failed_aps is set
            logger.warning('Checking for failed APs with default threshold values')
            self.get_aps()

        return self._has_failed_aps

    @lru_cache
    def _get_output_peaks(self):
        """
        Get all peaks from the output signal that are above threshold

        :param threshold:
        :return:
        """
        return find_peaks(
            self.time_steps, self.output_signal, threshold=self.failed_ap_amplitude)

    def _get_aps_with_idx(self):
        """
        Like get_aps, but the returned list includes list index of the peaks

        :return: [(idx, time, voltage), ...]
        """
        low_boundary_peaks = self._get_output_peaks()
        aps = [peak for peak in low_boundary_peaks if peak[2] > self.good_ap_amplitude]

        if len(low_boundary_peaks) > len(aps):
            logger.warning('Sweep at {}{} has failed APs'.format(
                self.get_drive_current(),
                self.input_signal_units
            ))
            self._has_failed_aps = True
        else:
            self._has_failed_aps = False

        return aps

    def get_aps(self, verify=False):
        """
        Look for peaks in the output voltage. Any peak above ap_fail_boundary
        is an AP. Peaks that pass low_boundary, but not ap_fail_boundary are
        failed peaks, which make the sweep invalid.

        :param failed_ap_amplitude:
        :param good_ap_amplitude:
        :return: [(ap_time, ap_voltage), ...]
        """
        if verify:
            raise NotImplementedError

        aps_with_idx = self._get_aps_with_idx()
        return [ap[1:3] for ap in aps_with_idx]

    @lru_cache
    def get_drive_current(self, verify=False):
        """
        Drive current is a tophat function.
        :return:
        """
        base_level, hat_level, hat_start, hat_end = fit_tophat(
            self.time_steps, self.input_signal, verify=verify)
        current_step = hat_level - base_level
        return current_step

    def get_ap_count(self):
        return len(self.get_aps())

    # def get_ap_times(self):
    #     pass
    #
    # def get_ap_amplitudes(self):
    #     pass

    @lru_cache
    def get_steady_state_ap_frequency(self):
        if self._has_failed_aps:
            raise InvalidSweep('Steady state firing frequency is invalid for sweeps with failed APs')

        aps = self.get_aps()

        if len(aps) < 2:
            raise InvalidSweep('Not enough APs to calculate a firing freqency')

        freq = (aps[-1][0] - aps[0][0]) / (len(aps) - 1)  # -1 so we are dividing by intervals, not peaks

        return freq

    def get_max_instantaneous_ap_frequency(self):
        """
        Get the frequency corresponding to the shortest gap between APs in this sweep

        :return:
        """
        if self.get_ap_count() < 2:
            raise InvalidSweep('Cannot get a frequency from < 2 APs')

        aps = self.get_aps()
        minimum_ap_interval = float_info.max
        minimum_ap_interval_first_ap = None
        for idx, ap in enumerate(aps[1:]):
            ap_interval = ap[0] - aps[idx][0]
            logger.debug('{}, {} to {} peak interval is {}'.format(
                self.sweep_name, idx, idx + 1, ap_interval))
            if ap_interval < minimum_ap_interval:
                logger.debug('Found a smaller peak interval in {}, peaks {} to {}'.format(
                        self.sweep_name, idx, idx + 1))
                minimum_ap_interval = ap_interval
                minimum_ap_interval_first_ap = idx

        return 1 / minimum_ap_interval, minimum_ap_interval_first_ap

    def get_ap_frequency_adaptation(self, min_ap_count):
        """
        Spike frequency adaptation: ratio of first to Nth interspike interval
        (ISI1/ISI10) and ratio of first to last interspike interval
        (ISI1/ISIn) for the first suprathreshold current injection to elicit
        sustained firing

        :return:
        """
        if self.get_ap_count() < min_ap_count:
            raise InvalidSweep('Not enough APs for spike frequency adaptation')

        if self.has_failed_aps():
            raise InvalidSweep('Sweep has failed APs')

        aps = self.get_aps()
        isi_1 = aps[1][0] - aps[0][0]
        isi_10 = aps[10][0] - aps[9][0]
        isi_n = aps[-1][0] - aps[-2][0]

        return isi_1/isi_10, isi_1/isi_n

    def get_first_ap_peak_data_index(self):
        """
        Get the index of the time_steps / input_signal / output_signal iterables
        where the peak of the first AP occurs

        :return:
        """
        if self.get_ap_count() < 1:
            raise InvalidSweep('No APs in {}'.format(self.sweep_name))

        aps_with_idx = self._get_aps_with_idx()
        return aps_with_idx[0][0]

    def get_first_ap_amplitude(self):
        """

        :return:
        """
        threshold_voltage, threshold_time = self.get_first_ap_threshold()
        first_peak_voltage = self.get_aps()[0][1]

        return first_peak_voltage - threshold_voltage

    def get_first_ap_rise_time(self):
        """

        :return:
        """
        threshold_voltage, threshold_time = self.get_first_ap_threshold()
        first_peak_time = self.get_aps()[0][0]

        return first_peak_time - threshold_time

    @lru_cache
    def get_first_ap_threshold(self, threshold):
        """
        Get the AP threshold of the first AP. Defined as the point at which its
        rate of change is greated than a threshold value.

        :param threshold
        :return:
        """
        if self.get_ap_count() == 0:
            raise InvalidSweep('Sweep {} has not APs'.format(self.sweep_name))

        dv_dt = self.get_output_derivative()
        for idx, gradient in enumerate(dv_dt):
            if gradient > threshold:
                # return the value of the voltage at the timestamp that
                # we cross the threshold in gradient, and the time it happened
                return self.output_signal[idx], self.time_steps[idx]

    def get_first_ap_half_width(self):
        """
        Full width, half maximum amplitude for first AP in the sweep. Amplitude measured
        from threshold to peak

        :return: width in sweep time units
        """


    def show_plot(self):
        pass


class ExperimentData(object):
    """The set of traces in one abf file (a colloquial, not mathematical set)"""
    SWEEP_CLASS = Sweep

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
                CurrentStepsSweep(
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
    def get_resting_potential(self, verify=True):
        """
        Resting potential is in the output trace. Just average it. There should
        be jsut one trace

        :return:
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            plt.plot(self.sweeps[0].time_steps, self.sweeps[0].output_signal)
            plt.axhline(mean_voltage)
            plt.show()

        assert len(self.sweeps) == 1
        mean_voltage = np.mean(self.sweeps[0].output_signal)

        if verify:
            verification_plot()

        return mean_voltage


class CurrentStepsData(ExperimentData):
    """Functions to get relevant metrics for 'current steps' experiments"""
    SWEEP_CLASS = CurrentStepsSweep

    def get_current_step_sizes(self, verify=False):
        """
        Get a list of the step sizes of the driving current in the same order
        as self.sweeps

        :return: list of floats
        """
        step_sizes = []
        for sweep in self.sweeps:
            base_level, hat_level, hat_start, hat_end = sweep.get_drive_current(verify)
            current_step = hat_level - base_level
            step_sizes.append(current_step)

        return step_sizes

    def get_ap_counts(self, verify=False):
        """
        Get a list of the number of action potentials in the output in the
        same order as self.sweeps

        :return: list of ints
        """
        ap_counts = []
        for sweep in self.sweeps:
            ap_counts.append(sweep.get_ap_count(verify=verify))

        return ap_counts

    def get_rheobase(self, verify=False):
        """
        Get the rheobase - the minimum voltage that elicits at least one peak
        :return:
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            for i, sweep in enumerate(self.sweeps):
                offset = 140 * i  # TODO Derive offset from data
                plt.plot(sweep.time_steps, sweep.output_signal + offset)
                text_height = sweep.output_signal[0] + offset
                plt.text(0, text_height, '{:.0f}'.format(sweep.get_drive_current()), fontsize=8)
                if i == rheobase_sweep_num:
                    plt.text(
                        .95,
                        text_height,
                        'Rheobase from this sweep',
                        horizontalalignment='right',
                        fontsize=8
                    )

            plt.gca().get_yaxis().set_visible(False)
            plt.show()

        rheobase_sweep_num = self._get_rheobase_sweep_num()
        rheobase = self.sweeps[rheobase_sweep_num].get_drive_current()

        if verify:
            verification_plot()

        return rheobase

    def _get_rheobase_sweep_num(self):
        """
        get the sweep number that the rheobase is obtained from. i.e. the first
        sweep with an AP
        :return:
        """
        for idx, sweep in enumerate(self.sweeps):
            if sweep.get_ap_count() > 0:
                return idx
        else:
            logger.warning('No sweep in {} had a peak'.format(self.filename))
            return None

    def get_spike_frequency_adaptation(self, min_ap_count=11, verify=False):
        """
        Spike frequency adaptation: ratio of first to 10th interspike interval
        (ISI1/ISI10) and ratio of first to last interspike interval
        (ISI1/ISIn) for the Sweep with most APs and at least `min_ap_count` APs

        :return: (isi_1/isi_10, isi_1/isi_n)
        """
        # Find the sweep with most APs. Breaking ties with higher sweep number.
        ap_counts = defaultdict(list)
        for idx, sweep in enumerate(self.sweeps):
            try:
                ap_counts[sweep.get_ap_count()].append(idx)
            except InvalidSweep:
                pass
        max_aps = max(ap_counts)
        if max_aps < min_ap_count:
            logger.info('{} had no sweeps with sustained firing'.format(self.filename))
            return None, None

        sfa_sweep = max(ap_counts[max_aps])
        sfa = self.sweeps[sfa_sweep].get_ap_frequency_adaptation(min_ap_count)

        if verify:
            raise NotImplementedError

        return sfa

    def get_max_steady_state_firing_frequency(self, verify=False):
        """
         Max steady state firing frequency:
         max mean firing frequency in response to current injection with no
         failures (AP amplitude at least 40mV and overshooting 0mV)

        # TODO what should be returned. frequency. Driving voltage eliciting that frequency?
        # TODO do we have to check for "missing" peaks
        :return: frequency, inverse of timesteps units
        """
        def verification_plot():
            plt.figure(figsize=(32, 20))
            plt.close('all')
            for i, sweep in enumerate(self.sweeps):
                offset = 140 * i  # TODO Derive offset from data
                text_height = sweep.output_signal[0] + offset
                plot_color = 'r' if i in invalid_sweeps else 'b'
                for freq, idx in frequencies.items():
                    if idx == i:
                        plt.text(0, text_height, '{:.2f}Hz'.format(freq), fontsize=8)
                if i == max_frequency_idx:
                    max_freq_peaks = self.sweeps[i].get_aps()
                    first_ap_time = max_freq_peaks[0][0]
                    last_ap_time = max_freq_peaks[-1][0]
                    time_diff = last_ap_time - first_ap_time
                    plt.text(
                        .95,
                        text_height,
                        'Max SSFF: {} APs in {:.2f}s = {:.0f}Hz'.format(
                            len(max_freq_peaks), time_diff, max_frequency),
                        horizontalalignment='right',
                        fontsize=8
                    )
                    plt.axvline(first_ap_time)
                    plt.axvline(last_ap_time)
                    plot_color = 'g'
                plt.plot(sweep.time_steps, sweep.output_signal + offset, color=plot_color)

            plt.gca().get_yaxis().set_visible(False)
            plt.show()

        # Start of function
        frequencies = {}  # {<frequency>: sweep_num, ... }
        invalid_sweeps = []
        for i, sweep in enumerate(self.sweeps):
            try:
                sweep_ap_freq = sweep.get_steady_state_ap_frequency()
            except InvalidSweep:
                invalid_sweeps.append(i)
                continue

            frequencies[sweep_ap_freq] = i

        if len(frequencies) == 0:
            logger.warning('No sweep had enough peaks to calculate a frequency')
            return 0.0
        else:
            max_frequency = max(frequencies)
            max_frequency_idx = frequencies[max_frequency]
            logger.info('Max SSFF is {} from sweep {}'.format(max_frequency, max_frequency_idx))

        if verify:
            verification_plot()

        return max_frequency

    def get_max_instantaneous_firing_frequency(self, verify=False):
        """
        Max instantaneous firing frequency:
        inverse of smallest interspike interval in response to current
        injection (AP amplitude at least 40mV and overshooting 0mV)


        :return:
        """
        max_frequency = float_info.min
        max_frequency_sweep = None
        for i, sweep in enumerate(self.sweeps):
            sweep_max_freq, max_freq_ap_num = sweep.get_max_instantaneous_ap_frequency()
            if sweep_max_freq > max_frequency:
                max_frequency = sweep_max_freq
                max_frequency_sweep = i

        if verify:
            logger.info('max frequency is in sweep {}'.format(max_frequency_sweep))
            raise NotImplementedError

        return max_frequency

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
            try:
                threshold_voltage, threshold_time = sweep.get_first_ap_threshold(gradient_threshold)
                return sweep_num, threshold_voltage, threshold_time
            except InvalidSweep:
                pass
        else:
            # If there are no peaks in any sweep we will hit this. This shouldn't happen.
            raise Exception('No sweep had an AP')

    def _get_ap_threshold_1_time(self):
        """


        :return:
        """
        return self._get_ap_threshold_1_details()[2]

    def get_ap_threshold(self):
        """

        :return:
        """
        return self._get_ap_threshold_1_details()[1]

    def get_ap_rise_time(self, verify=False):
        """
        AP rise time:
        for first spike obtained at suprathreshold current injection, time
        from AP threshold 1 to peak

        :return:
        """
        sweep_num, ap_threshold, ap_threshold_time = self._get_ap_threshold_1_details()
        rise_time = self.sweeps[sweep_num].get_first_ap_rise_time()

        if verify:
            raise NotImplementedError

        return rise_time

    def get_ap_amplitude(self, verify=False):
        """
        AP amplitude:
        for first spike obtained at suprathreshold current injection, change
        in mV from AP threshold #1 to peak

        :return:
        """
        sweep_num, ap_threshold_voltage, ap_threshold_time = \
            self._get_ap_threshold_1_details()
        ap_amplitude = self.sweeps[sweep_num].get_first_ap_amplitude()

        if verify:
            raise NotImplementedError

        return ap_amplitude

    def get_ap_half_width(self, verify=False):
        """
        AP half-width:
        for first spike obtained at suprathreshold current injection, width
        of the AP (in ms) at 1/2 maximal amplitude, using AP threshold #1 and
        AP amplitude

        :return:
        """
        def verification_plot():
            plt.close('all')
            plot_slice_start_idx = peak_start_idx - 3 * (peak_end_idx - peak_start_idx)
            plot_slice_end_idx = peak_end_idx + 5 * (peak_end_idx - peak_start_idx)
            plt.plot(
                rheobase_sweep.time_steps[plot_slice_start_idx: plot_slice_end_idx],
                rheobase_sweep.output_signal[plot_slice_start_idx: plot_slice_end_idx]
            )

            # TODO add text for threshold, peak, half, horizontal lines
            plt.axhline(threshold_voltage, color='b')
            plt.axhline(rheobase_sweep.output_signal[ap_idx], color='b')

            half_width_amp = 0.5 * (rheobase_sweep.output_signal[ap_idx] + threshold_voltage)
            plt.axhline(half_width_amp, color='r')

            plt.axvline(rheobase_sweep.time_steps[peak_start_idx])
            plt.axvline(rheobase_sweep.time_steps[peak_end_idx])

            plt.text(
                rheobase_sweep.time_steps[plot_slice_end_idx],
                0.5 * (ap_peak_voltage + half_width_amp),
                'AP half width ={:.2g}'.format(ap_half_width),
                horizontalalignment='right',
                verticalalignment='top')

            plt.show()

        rheobase_sweep = self.sweeps[self._get_rheobase_sweep_num()]
        half_width_ap = rheobase_sweep.get_aps()[0]
        threshold_voltage = self.get_ap_threshold()
        ap_time, ap_peak_voltage = half_width_ap
        half_peak_voltage = 0.5 * (ap_peak_voltage + threshold_voltage)
        ap_idx = rheobase_sweep.get_first_ap_peak_data_index()
        voltage_at_time = dict(zip(rheobase_sweep.time_steps, rheobase_sweep.output_signal))

        # Iterate back through the data to find the half peak time
        for idx_diff, time_step in enumerate(rheobase_sweep.time_steps[ap_idx::-1]):
            if voltage_at_time[time_step] < half_peak_voltage:
                logger.info('Found peak start at {}'.format(time_step))
                peak_start = time_step
                peak_start_idx = ap_idx - idx_diff
                break

        # Iterate forward through the data to find the half peak time
        for idx_diff, time_step in enumerate(rheobase_sweep.time_steps[ap_idx:]):
            if voltage_at_time[time_step] < half_peak_voltage:
                logger.info('Found peak end at {}'.format(time_step))
                peak_end = time_step
                peak_end_idx = ap_idx + idx_diff
                break

        ap_half_width = peak_end - peak_start

        if verify:
            verification_plot()

        return ap_half_width

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
        # for sweep in experiment.sweeps:
        #     sweep.show_plot()


        # rheobase = experiment.get_rheobase(verify=True)
        # print('Rheobase of {} is {}mV'.format(experiment.filename, rheobase))
        # exit()
        #
        # sfa = experiment.get_spike_frequency_adaptation()
        # print('SFA is {}'.format(sfa))

        max_ssff = experiment.get_max_steady_state_firing_frequency(verify=True)
        print('Max steady state firing frequency is {}'.format(max_ssff))

        # max_iff = experiment.get_max_instantaneous_firing_frequency()
        # print('Max instantaneous firing frequency is {}'.format(max_iff))
        #
        # ap_threshold_1 = experiment.get_ap_threshold()
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