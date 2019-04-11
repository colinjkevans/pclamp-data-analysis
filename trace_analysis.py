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


def fit_tophat(x, y, verify=False, verify_file='verfication.png'):
    """
    Fit the x and y data to a tophat function, returning:
    base_level - the y-value inside the tophat
    hat_level - the y-value outside the tophat
    hat_mid - centre of the tophat
    hat_width - width of the tophat

    :param x: iterable of x values
    :param y: corresponding iterable of y values
    :param verify: Show a plot of the fit, blocking progress until it is dismissed
    :param verify_file:
    :return: (base_level, hat_level, hat_mid, hat_width)
    """
    def top_hat(x, base_level, hat_level, hat_mid, hat_width):
        return np.where((hat_mid - hat_width / 2.0 < x) & (x < hat_mid + hat_width / 2.0), hat_level, base_level)

    gradient = list(get_derivative(y, x))
    max_gradient = max(gradient)
    min_gradient = min(gradient)

    # The tophat could be upside down, so we don't know which of these comes
    # in the x direction
    max_gradient_index = gradient.index(max_gradient)
    min_gradient_index = gradient.index(min_gradient)
    step_indices = (max_gradient_index, min_gradient_index)

    max_gradient_x = x[max_gradient_index]
    min_gradient_x = x[min_gradient_index]
    step_xs = (max_gradient_x, min_gradient_x)

    base_level = np.mean(y[:min(step_indices)])
    hat_level = np.mean(y[min(*step_indices):max(*step_indices)])
    hat_mid = np.mean(step_xs)
    hat_width = max(*step_xs) - min(*step_xs)

    if verify:
        plt.close('all')
        plt.figure(figsize=(8, 5))
        plt.plot(x, y)
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

