import numpy as np
from src.fc import get_connectome
from src.data_utils import load_data
from nilearn.connectome import sym_matrix_to_vec
import pandas as pd


def qc_fc(fc, mean_rms_vec) -> np.ndarray:
    """
    Compute the quality control functional connectivity (QC-FC) matrix.

    This function calculates the correlation between each element of the 
    functional connectivity matrix and the mean root mean squared 
    (RMS) vector. The resulting matrix represents the QC-FC values.

    Parameters
    ----------
    fc : np.ndarray
        A 3D numpy array representing the functional connectivity data 
        with shape (subjects, regions, regions).
    mean_rms_vec : np.ndarray
        A 1D numpy array containing the mean RMS values for each subject.
        It should have the same length as the number of subjects in the 
        functional connectivity data. 
        The order of subjects in fc matrix should be the same as mean_rms_vec.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the QC-FC matrix with shape 
        (regions, regions), where each element is the correlation 
        coefficient between the FC data and the mean RMS vector.
    """

    qc_mat = np.zeros((fc.shape[1], fc.shape[2]))

    for i in range(fc.shape[1]):
        for t in range(fc.shape[2]):
            # Calculate the correlation between the FC measure and motion estimates
            qc_mat[i, t] = np.corrcoef(fc[:, i, t], mean_rms_vec)[0, 1]

    return np.abs(qc_mat).flatten()


def icc_matrix(data, mask=False) -> np.ndarray:
    """
    Computes the ICC coefficient for all edges in the correlation matrices.
    
    Parameters:
    - data: np.ndarray of shape (K, N_subjects, N_rois, N_rois)
            where K is the number of sessions (2 in this case),
            N_subjects is the number of subjects,
            and N_rois is the number of regions of interest (ROIs).
    
    Returns:
    - ICC matrix of shape (N_rois, N_rois) containing ICC values for all edges.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    K, N_subjects, N_rois, _ = data.shape

    # Compute the grand mean across all subjects and conditions for each edge
    grand_mean = data.mean(axis=(0, 1))  # Shape: (N_rois, N_rois)

    # Compute the subject means for each edge
    subject_means = data.mean(axis=0)  # Shape: (N_subjects, N_rois, N_rois)

    # Compute Between Mean Squares (BMS)
    BMS = K * np.sum((subject_means - grand_mean) ** 2, axis=0) / (N_subjects - 1)  # Shape: (N_rois, N_rois)

    # Compute Within Mean Squares (WMS)
    deviations = data - subject_means[np.newaxis, :, :, :]  # Shape: (K, N_subjects, N_rois, N_rois)
    WMS = np.sum(deviations ** 2, axis=(0, 1)) / (N_subjects * (K - 1))  # Shape: (N_rois, N_rois)

    # Compute ICC for all edges
    ICC = (BMS - WMS) / (BMS + (K - 1) * WMS + 0.0000001)
    np.fill_diagonal(ICC, 1.0)

    if mask:
        threshold = np.percentile(ICC.mean(axis=0), 98)
        ICC = np.where(np.abs(ICC) > threshold, ICC, np.nan)

    return ICC



def strategies_comparison(closed_path, open_path, fc, atlas) -> pd.DataFrame:
    """
    Compute the mean correlation coefficient between all pairs of subjects 
    for the 6 strategies in China dataset.

    Parameters
    ----------

    Notes
    -----
    Strategies should be ordered: '24P', 'aCompCor+12P', 'aCompCor50+12P', 
    'aCompCor+24P', 'aCompCor50+24P', 'a/tCompCor50+24P' + same with GSR

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mean correlation coefficients between all pairs of subjects 
        for the 6 strategies in China dataset.
    """
    data = {}
    strategies = ['24P', 'aCompCor+12P', 'aCompCor50+12P', 
                  'aCompCor+24P', 'aCompCor50+24P', 'atCompCor50+24P']
    
    for i in range(len(strategies)):
        cl, op = load_data(closed_path, open_path, fc=fc, atlas=atlas, strategy=i+1, gsr='noGSR', icc_session=None)
        data[f'close_{strategies[i]}'] = sym_matrix_to_vec(cl)
        data[f'open_{strategies[i]}'] = sym_matrix_to_vec(op)

        cl_g, op_g = load_data(closed_path, open_path, fc=fc, atlas=atlas, strategy=i+1, gsr='GSR', icc_session=None)
        data[f'close_{strategies[i]}_GSR'] = sym_matrix_to_vec(cl_g)
        data[f'open_{strategies[i]}_GSR'] = sym_matrix_to_vec(op_g)
    
    k = sorted(data.keys())
    subs = len(data['close_24P_GSR'])
    out = np.zeros((24, 24))

    for en, i in enumerate(k):
        for en2, t in enumerate(k):
            out[en, en2] = np.mean(
                [np.corrcoef(data[i][sub],
                             data[t][sub])[0, 1] for sub in range(subs)], axis=0)
            
    # Create DataFrame for visualization         
    df = pd.DataFrame(data=out, columns=k, index=k)
    
    return df


    
    