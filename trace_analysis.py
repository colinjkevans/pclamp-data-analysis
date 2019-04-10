from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import logging
import os.path

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

VERIFICATION_DIR = 'verification'


def verify_offline(plot, filename, verification_dir=VERIFICATION_DIR):
    """
    TODO move this into Sweep class?
    Write a plot to disk showing output of trace analysis for verification.
    The file will be output to verification/filename

    :param plot: a matplotlib pyplot
    :param filename: filename of verification file
    :return: None
    """
    if not os.path.isdir(verification_dir):
        os.mkdir(verification_dir)

    filepath = os.path.join(verification_dir, filename)
    logger.info('Saving verification plot to {}'.format(filepath))
    plot.savefig(filepath)


# def fit_tophat(x, y, verify=False, max_fitting_passes=2, verify_file='verfication.png'):
#     """
#     Fit the x and y data to a tophat function, returning:
#     base_level - the y-value inside the tophat
#     hat_level - the y-value outside the tophat
#     hat_mid - centre of the tophat
#     hat_width - width of the tophat
#
#     Fitting will be repeated until the optimizer exits with success, or is
#     run `max_fitting_passes` times
#
#     :param x: iterable of x values
#     :param y: corresponding iterable of y values
#     :param verify: Show a plot of the fit, blocking progress until it is dismissed
#     :param max_fitting_passes: max how many rounds of mimimzation of residuals to do.
#     :param verify_file:
#     :return: (base_level, hat_level, hat_mid, hat_width)
#     """
#
#     def top_hat(x, base_level, hat_level, hat_mid, hat_width):
#         return np.where((hat_mid - hat_width / 2.0 < x) & (x < hat_mid + hat_width / 2.0), hat_level, base_level)
#
#     def objective(params, x, y):
#         return np.sum(np.abs((top_hat(x, *params) - y)))
#
#     assert len(x) == len(y)  # There must be the same amount of x as y values
#
#     # Chose initial guesses
#     base_level = np.mean(y[0:100])
#     hat_level = min(y) if (abs(min(y) - base_level) > abs(max(y) - base_level)) else max(y)
#     # hat_mid = x[len(x) // 2]  # centre of the trace
#     hat_mid = x[list(y).index(hat_level)]  # x value of hat_mid estimate
#     # the middle half of the x-range (of data points not values), or hat_mid,
#     # whichever is less
#     hat_width = min(x[3 * len(x) // 4] - x[len(x) // 4], hat_mid)
#
#     # Miminize the residuals. Keep going until it completes or we hit the max passes
#     params = (base_level, hat_level, hat_mid, hat_width)
#     for i, _ in enumerate(range(max_fitting_passes)):
#         res = minimize(objective, params, args=(x, y), method='Nelder-Mead')
#         logger.debug('Optimizer message: {}'.format(res.message))
#         logger.debug('Optimizer status: {}'.format(res.status))
#         params = res.x
#         if res.status == 0:
#             logger.info('Tophat fit in {} passes. Height: {}, width: {}'.format(
#                 i+1,
#                 params[1] - params[0],
#                 params[3]))
#             break
#     else:
#         # Executes if for-loop exits without a "break"
#         logger.warning('Optimizer did not finish fitting tophat successfully in {} passes'.format(i+1))
#
#     # print('base V: {}\ndrive V: {}\ndV: {}\nstart time: {}\nend time: {}\n'.format(
#     #     res.x[0], res.x[1], res.x[1] - res.x[0], res.x[2] - res.x[3] / 2, res.x[2] + res.x[3] / 2))
#
#     if verify:
#         plt.close('all')
#         plt.figure(figsize=(8, 5))
#         # fig, ax = plt.subplots()
#         plt.plot(x, y)
#         # plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
#         plt.plot(x, top_hat(x, *res.x))
#         if verify == 'offline':
#             verify_offline(plt, verify_file)
#         else:
#             plt.show()
#
#     return res.x

def fit_tophat(x, y, verify=False, max_fitting_passes=2, verify_file='verfication.png'):
    """

    :param x:
    :param y:
    :param verify:
    :param max_fitting_passes:
    :param verify_file:
    :return:
    """
    def top_hat(x, base_level, hat_level, hat_mid, hat_width):
        return np.where((hat_mid - hat_width / 2.0 < x) & (x < hat_mid + hat_width / 2.0), hat_level, base_level)

    gradient = list(get_derivative(y, x))
    max_gradient = max(gradient)
    min_gradient = min(gradient)

    max_gradient_index = gradient.index(max_gradient)
    min_gradient_index = gradient.index(min_gradient)
    step_indices = (max_gradient_index, min_gradient_index)

    max_gradient_x = x[max_gradient_index]
    min_gradient_x = x[min_gradient_index]
    step_xs = (max_gradient_x, min_gradient_x)

    first_step_x = min(max_gradient_x, min_gradient_x)
    second_step_x = max(max_gradient_x, min_gradient_x)

    base_level = np.mean(y[:min(step_indices)])
    hat_level = np.mean(y[min(*step_indices):max(*step_indices)])
    hat_mid = np.mean(step_xs)
    hat_width = max(*step_xs) - min(*step_xs)

    if verify:
        plt.close('all')
        plt.figure(figsize=(8, 5))
        # fig, ax = plt.subplots()
        plt.plot(x, y)
        # plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
        plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
        if verify == 'offline':
            verify_offline(plt, verify_file)
        else:
            plt.show()

    return base_level, hat_level, hat_mid, hat_width



def find_peaks(x, y, threshold=0, verify=False, verify_file='verification.png'):
    """
    Count signficant spikes in y values above threshold, for some definition
    of "significant". This is a very naive approach - any time the trace goes
    then below the threshold, find the highest value while it is above. That
    is the peak. It works so far. scipy.signal.findpeaks may provide a more
    robust approach if needed, using a peaks "prominence".

    :param x: list of x values
    :param y: list of y values
    :param threshold: y threshold that neesd to be crossed to count as a peak
    :param verify: If True, a plot showing peaks found will be shown, or saved to file if verify=='offline'
    :param verify_file: name of file for offline verification
    :return: list of (x, y) values of peaks at point of maximum y
    """
    logger.debug('Peak threshold is {}'.format(threshold))

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
        if verify == 'offline':
            verify_offline(plt, verify_file)
        else:
            plt.show()

    logger.info('Found {} peaks'.format(len(maximums)))
    return maximums


def get_derivative(y, x):
    """
    Return the numerical derivatice of the data dy/dx
    :param y: y values list
    :param x: x values list
    :return: dy/dx
    """
    return np.gradient(y, x)

