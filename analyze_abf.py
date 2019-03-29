import logging
import numpy as np
from scipy.optimize import minimize
from pyabf import ABF
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def smooth_data(data, window=2):
    """
    Moving average smoothing of data

    :param data: (time, measured output, driving signal)
    :param window: half-width of moving average window
    :return: The smoothed data in teh same format as data
    """
    smoothed_data = []
    for idx, data_point in enumerate(data):
        # if idx < window:
        #     smoothed_data.append(data_point)
        # elif idx > len(data) - 2:
        #     smoothed_data.append(data_point)
        # if idx - window < 0 or idx + window > len(data):
        #     logger.warning('Discarding data that cannot be smoothed, t={}'.format(data_point[0]))  # TODO fix level
        #     continue
        slice_lower_bound = idx - window if idx - window >= 0 else None
        slice_upper_bound = idx + window + 1 if idx + window <= len(data) - 1 else None
        data_slice = data[slice_lower_bound: slice_upper_bound]
        smoothed_data.append(
            (data_point[0], np.mean([d[1] for d in data_slice]), np.mean([d[2] for d in data_slice]))
        )

    return smoothed_data


def fit_tophat(x, y, verify=False, max_fitting_passes=2):
    """
    Fit the x and y data to a tophat function, returning:
    base_level - the y-value inside the tophat
    hat_level - the y-value outside the tophat
    hat_mid - centre of the tophat
    hat_width - width of the tophat

    Fitting will be repeated until the optimizer exits with success, or is run `max_fitting_passes` times

    :param x: iterable of x values
    :param y: corresponding iterable of y values
    :param verify: Show a plot of the fit, blocking progress until it is dismissed
    :param max_fitting_passes: max how many rounds of mimimzation of residuals to do.
    :return: (base_level, hat_level, hat_mid, hat_width)
    """

    def top_hat(x, base_level, hat_level, hat_mid, hat_width):
        return np.where((hat_mid-hat_width/2.0 < x) & (x < hat_mid+hat_width/2.0), hat_level, base_level)
    
    def objective(params, x, y):
        return np.sum(np.abs((top_hat(x, *params) - y)))

    assert len(x) == len(y)  # There must be the same amount of x as y values

    # Chose initial guesses
    base_level = np.mean(y[0:100])
    hat_level = min(y) if (abs(min(y) - base_level) > abs(max(y) - base_level)) else max(y)
    hat_mid = x[len(x) // 2]  # centre of the trace
    hat_width = x[3 * len(x) // 4] - x[len(x) // 4]  # the middle half of the x-range (of data points, not values)

    # Miminize the residuals. The second round is needed to get the base level right
    params = (base_level, hat_level, hat_mid, hat_width)
    for _ in range(max_fitting_passes):
        res = minimize(objective, params, args=(x, y), method='Nelder-Mead')
        logger.debug('Optimizer message: {}'.format(res.message))
        logger.debug('Optimizer status: {}'.format(res.status))
        if res.status == 0:
            break
        params = res.x
    else:
        logger.warning('Optimizer did not finish fitting tophat successfully')

    # print('base V: {}\ndrive V: {}\ndV: {}\nstart time: {}\nend time: {}\n'.format(
    #     res.x[0], res.x[1], res.x[1] - res.x[0], res.x[2] - res.x[3] / 2, res.x[2] + res.x[3] / 2))

    if verify:
        plt.close('all')
        plt.figure(figsize=(8, 5))
        # fig, ax = plt.subplots()
        plt.plot(x, y)
        #plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
        plt.plot(x, top_hat(x, *res.x))
        plt.show()

    return res.x


def count_peaks(x, y, threshold=0, verify=False):
    """
    Count signficant spikes in y values above threshold, for some definition
    of "significant".

    :param x: list of x values
    :param y: list of y values
    :param threshold: y threshold that neesd to be crossed to count as a peak
    :param verify: If True, a plot showing peaks found will be shown and block progress
    :return: list of (x, y) values of peaks at point of maximum y
    """
    in_peak = False
    peak = []  # list of data points that are part of a peak
    peaks = []  # list or peaks
    for data_point in zip(x, y):
        if in_peak:
            if data_point[1] > threshold:
                peak.append(data_point)
            else:
                in_peak = False
                peaks.append(peak)
        elif data_point[1] > threshold:
            in_peak = True
            peak = [data_point]

    # print([peak[0] for peak in peaks])
    # if len(peaks) > 0:
    #     print([max(peak, key=lambda d: d[1]) for peak in peaks])
    # else:
    #     print('No peaks')
    # print(len(peaks))

    maximums = [max(peak, key=lambda d: d[1]) for peak in peaks]

    if verify:
        plt.close(fig='all')  # Make sure there are no unwanted figures
        plt.figure(figsize=(16, 10))
        plt.plot(x, y)
        for m in maximums:
            plt.axvline(x=m[0], color='red')
        plt.show()

    return maximums


if __name__ == '__main__':
    filename = '.\\test\\KG173_02_01_19_0075.abf'
    abf = ABF(filename)
    # input: sweepY, channel=1
    # output: sweepY, channel=0
    # time: sweepX?
    # print(dir(abf))
    # print(abf.sweepList)
    logger.info(dir(abf))
    logger.info('Filename: {}'.format(filename))
    logger.info('File contains {} sweeps'.format(abf.sweepCount))

    for sweep_num in abf.sweepList:
        logger.info('Sweep number {} has {} data points'.format(sweep_num, len(abf.sweepY)))
        abf.setSweep(sweep_num, channel=1)
        drive_signal = abf.sweepY

        abf.setSweep(sweep_num, channel=0)
        response_signal = abf.sweepY

        time = abf.sweepX

        base_level, hat_level, hat_mid, hat_width = fit_tophat(time, drive_signal)
        actions_potentials = count_peaks(time, response_signal)
        print('dV={}, AP count: {}'.format(hat_level - base_level, len(actions_potentials)))



