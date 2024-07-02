import numpy as np
import pandas as pd


def _generate_raincloud_random_data(
    min_max,
    n_grp_samples,
    features_label,
    group_label,
    group_names,
    data_file,
    group_nans=None,
):
    rng = np.random.default_rng(1234)

    if group_nans is None:
        group_nans = [None] * len(min_max)

    # Create some random data in the [min_val, max_val) half-open interval
    values = np.array([])
    names = []
    for group_min_max, name, nans in zip(min_max, group_names, group_nans):
        min_val = group_min_max[0]
        max_val = group_min_max[1]
        range_size = max_val - min_val
        _values = rng.random(n_grp_samples) * range_size + min_val

        values = np.concatenate((values, _values), axis=0)
        names.extend([name] * n_grp_samples)

        if nans:
            values = np.concatenate((values, [np.nan] * nans), axis=0)
            names.extend([name] * nans)

    df = pd.DataFrame(np.vstack([values, names]).T, columns=[features_label, group_label])

    df.to_csv(data_file, sep="\t")
