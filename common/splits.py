"""Subject-wise splits. Windows from one patient (any day) never appear in
more than one of train / validation / test."""
import numpy as np


def patient_split(patient_ids, n_folds=5, test_fold=0, seed=0):
    """Assign patients to folds (seeded shuffle, round-robin). Fold ``test_fold``
    is the test set, the next fold is validation, the rest is training.

    Returns a dict ``{"train": [...], "val": [...], "test": [...]}`` of patient IDs.
    """
    patients = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    folds = np.arange(len(patients)) % n_folds
    val_fold = (test_fold + 1) % n_folds
    return {
        "train": sorted(patients[(folds != test_fold) & (folds != val_fold)].tolist()),
        "val": sorted(patients[folds == val_fold].tolist()),
        "test": sorted(patients[folds == test_fold].tolist()),
    }
