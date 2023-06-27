import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_numeric_dtype
from RRest.common.rpeak_detection import PeakDetector
from hrvanalysis import get_nn_intervals

OPERAND_MAPPING_DICT = {
    ">": 5,
    ">=": 4,
    "=": 3,
    "<=": 2,
    "<": 1
}


def get_nn(s, wave_type='ppg', sample_rate=100, rpeak_method=7,
           remove_ectopic_beat=False):
    if wave_type == 'ppg':
        detector = PeakDetector(wave_type='ppg')
        peak_list, trough_list = detector.ppg_detector(s, detector_type=rpeak_method)
    else:
        detector = PeakDetector(wave_type='ecg')
        peak_list, trough_list = detector.ecg_detector(s, detector_type=rpeak_method)

    rr_list = np.diff(peak_list) * (1000 / sample_rate)
    if not remove_ectopic_beat:
        return rr_list
    nn_list = get_nn_intervals(rr_list)
    nn_list_non_na = np.copy(nn_list)
    nn_list_non_na[np.where(np.isnan(nn_list_non_na))[0]] = -1
    return nn_list_non_na


def check_valid_signal(x):
    """Check whether signal is valid, i.e. an array_like numeric, or raise errors.

    Parameters
    ----------
    x :
        array_like, array of signal

    Returns
    -------


    """
    if isinstance(x, dict) or isinstance(x, tuple):
        raise ValueError("Expected array_like input, instead found {"
                         "0}:".format(type(x)))
    if len(x) == 0:
        raise ValueError("Empty signal")
    types = []
    x = list(x)
    for i in range(len(x)):
        types.append(str(type(x[i])))
    type_unique = np.unique(np.array(types))
    if len(type_unique) != 1 and (type_unique[0].find("int") != -1 or
                                  type_unique[0].find("float") != -1):
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    if type_unique[0].find("int") == -1 and type_unique[0].find("float") == -1:
        raise ValueError("Invalid signal: Expect numeric array, instead found "
                         "array with types {0}: ".format(type_unique))
    return True


def cut_segment(df, milestone):
    """
    Spit Dataframe into segments, base on the pair of start and end indices.

    Parameters
    ----------
    df :
        Signal dataframe .
    milestone :
        Indices dataframe

    Returns
    -------
    The list of split segments.
    """
    assert isinstance(milestone, pd.DataFrame), \
        "Please convert the milestone as dataframe " \
        "with 'start' and 'end' columns. " \
        ">>> from vital_sqi.common.utils import format_milestone" \
        ">>> milestones = format_milestone(start_milestone,end_milestone)"
    start_milestone = np.array(milestone.iloc[:, 0])
    end_milestone = np.array(milestone.iloc[:, 1])
    processed_df = []
    for start, end in zip(start_milestone, end_milestone):
        processed_df.append(df.iloc[int(start):int(end)])
    return processed_df


def format_milestone(start_milestone, end_milestone):
    """

    Parameters
    ----------
    start_milestone :
        array-like represent the start indices of segment.
    end_milestone :
        array-like represent the end indices of segment.

    Returns
    -------
    a dataframe of two columns.
    The first column indicates the start indices, the second indicates the end indices
    """
    assert len(start_milestone) == end_milestone, "The list of  start indices and end indices must equal size"
    df_milestones = pd.DataFrame()
    df_milestones['start'] = start_milestone
    df_milestones['end'] = end_milestone
    return df_milestones


def check_signal_format(s):
    assert isinstance(s, pd.DataFrame), 'Expected a pd.DataFrame.'
    assert len(s.columns) == 2, 'Expect a datafram of only two columns.'
    assert isinstance(s.iloc[0, 0], pd.Timestamp), \
        'Expected type of the first column to be pd.Timestamp.'
    assert is_numeric_dtype(s.iloc[0, 1]), \
        'Expected type of the second column to be float'
    return True


def create_rule_def(sqi_name, upper_bound=0, lower_bound=1):
    json_rule_dict = {}
    json_rule_dict[sqi_name] = {
        "name": sqi_name,
        "def": [
            {"op": ">", "value": str(lower_bound), "label": "accept"},
            {"op": "<=", "value": str(lower_bound), "label": "reject"},
            {"op": ">=", "value": str(upper_bound), "label": "reject"},
            {"op": "<", "value": str(upper_bound), "label": "accept"},
        ],
        "desc": "",
        "ref": ""
    }
    return json_rule_dict
