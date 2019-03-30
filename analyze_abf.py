import os
import logging
import numpy as np
from pyabf import ABF

from trace_analysis import fit_tophat, count_peaks
from config import ABF_LOCATION, AP_THRESHOLD

ABF_FILE_EXTENSION = '.abf'

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
            actions_potentials = count_peaks(time, response_signal, threshold=AP_THRESHOLD)
            print('dV={}, AP count: {}'.format(hat_level - base_level, len(actions_potentials)))



