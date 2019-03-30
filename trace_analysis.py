from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        return np.where((hat_mid - hat_width / 2.0 < x) & (x < hat_mid + hat_width / 2.0), hat_level, base_level)

    def objective(params, x, y):
        return np.sum(np.abs((top_hat(x, *params) - y)))

    assert len(x) == len(y)  # There must be the same amount of x as y values

    # Chose initial guesses
    base_level = np.mean(y[0:100])
    hat_level = min(y) if (abs(min(y) - base_level) > abs(max(y) - base_level)) else max(y)
    hat_mid = x[len(x) // 2]  # centre of the trace
    hat_width = x[3 * len(x) // 4] - x[len(x) // 4]  # the middle half of the x-range (of data points, not values)

    # Miminize the residuals. Keep going until it completes or we hit the max passes
    params = (base_level, hat_level, hat_mid, hat_width)
    for _ in range(max_fitting_passes):
        res = minimize(objective, params, args=(x, y), method='Nelder-Mead')
        logger.debug('Optimizer message: {}'.format(res.message))
        logger.debug('Optimizer status: {}'.format(res.status))
        if res.status == 0:
            break
        params = res.x
    else:
        # Executes if for-loop exits without a "break"
        logger.warning('Optimizer did not finish fitting tophat successfully')

    # print('base V: {}\ndrive V: {}\ndV: {}\nstart time: {}\nend time: {}\n'.format(
    #     res.x[0], res.x[1], res.x[1] - res.x[0], res.x[2] - res.x[3] / 2, res.x[2] + res.x[3] / 2))

    if verify:
        plt.close('all')
        plt.figure(figsize=(8, 5))
        # fig, ax = plt.subplots()
        plt.plot(x, y)
        # plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
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
    logger.info('Peak threshold is {}'.format(threshold))

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
