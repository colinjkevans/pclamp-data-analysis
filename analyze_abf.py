import logging
import numpy as np
from scipy.optimize import minimize
from pyabf import ABF
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def fit_tophat(x, y):
    """
    Fit the x and y data to a tophat function, returning:
    base_level - the y-value inside the tophat
    hat_level - the y-value outside the tophat
    hat_mid - centre of the tophat
    hat_width - width of the tophat

    :param x: iterable of x values
    :param y: corresponding iterable of y values
    :return: (base_level, hat_level, hat_mid, hat_width)
    """

    def top_hat(x, base_level, hat_level, hat_mid, hat_width):
        return np.where((hat_mid-hat_width/2.0 < x) & (x < hat_mid+hat_width/2.0), hat_level, base_level)
    
    def objective(params, x, y):
        return np.sum(np.abs((top_hat(x, *params) - y)))

    assert len(x) == len(y)  # There must be the same amount of x as y values

    # Chose initial guesses
    base_level = y[0]
    hat_level = min(y) if (abs(min(y) - base_level) > abs(max(y) - base_level)) else max(y)
    hat_mid = x[len(x) // 2]  # centre of the trace
    hat_width = x[3 * len(x) // 4] - x[len(x) // 4]  # the middle half of the x-range (of data points, not values)

    params = (base_level, hat_level, hat_mid, hat_width)
    res = minimize(objective, params, args=(x, y), method='Nelder-Mead')

    plt.figure(figsize=(8, 5))
    # fig, ax = plt.subplots()
    plt.plot(x, y)
    #plt.plot(x, top_hat(x, base_level, hat_level, hat_mid, hat_width))
    plt.plot(x, top_hat(x, *res.x))
    print('base V: {}\ndrive V: {}\ndV: {}\nstart time: {}\nend time: {}'.format(
        res.x[0], res.x[1], res.x[1] - res.x[0],res.x[2] - res.x[3]/2, res.x[2] + res.x[3]/2))
    # plt.text(0.95, 0.95, 'text', fontsize=10, bbox=dict(facecolor='red', alpha=0.5),transform=ax.transAxes)
    plt.show()





def find_edges(data, window=2, threshold=50000):
    """
    Check slope of data in window, find instances where slope is greater than threshold

    :param data: (timestamp, data value)
    :param window: half width of window for slope calculation
    :param threshold: steepness of slope that counts as an edge
    :return:
    """
    slopes = []
    for idx, data_point in enumerate(data):
        # if idx - window < 0 or idx + window > len(data):
        #     logger.warning('Discarding data that cannot give slope, t={}'.format(data_point[0]))  # TODO fix level
        #     continue
        data_slice = data[idx - window: idx + window + 1]
        t_values = [d[0] for d in data_slice]
        data_values = [d[1] for d in data_slice]
        timestamp = data_point[0]

        slope = np.linalg.lstsq(list(zip(t_values, np.ones(len(t_values)))), data_values, rcond=None)[0][0]
        slopes.append((timestamp, slope))
        # if idx % 100 == 0:
        #     print(timestamp, slope)
        if abs(slope) > threshold:
            print(timestamp, slope)

    print(max([s[1] for s in slopes]))
    print(min([s[1] for s in slopes]))


if __name__ == '__main__':
    abf = ABF('.\\test\\KG173_02_01_19_0075.abf')
    # input: sweepY, channel=1
    # output: sweepY, channel=0
    # time: sweepX?



    sweeps = abf.sweepCount
    logger.info(sweeps)
    abf.setSweep(10, channel=1)
    # print(abf.data[0])
    # for c in [0, 1]:
    #     for i, d in enumerate(abf.data[c]):
    #         if i % 10000 == 0:
    #             print(d)

    logger.info('{}, {}'.format(len(abf.sweepX), abf.sweepX))

    abf.setSweep(10, channel=1)
    input = abf.sweepY

    abf.setSweep(10, channel=0)
    output = abf.sweepY

    time = abf.sweepX

    fit_tophat(time, input)
    exit(0)

    data = list(zip(time, output, input))
    smoothed_data = smooth_data(data, window=2)


    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('output', color=color)
    ax1.plot([d[0] for d in smoothed_data], [d[1] for d in smoothed_data], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('input', color=color)  # we already handled the x-label with ax1
    abf.setSweep(10, channel=1)
    ax2.plot([d[0] for d in smoothed_data], [d[2] for d in smoothed_data], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    #smoothed_data = smooth_data(data)
    #find_edges([(d[0], d[2]) for d in smoothed_data])

    # for i in range(100):
    #     print(smoothed_data[i], data[i])

