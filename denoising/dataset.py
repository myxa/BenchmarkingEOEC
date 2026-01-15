from pathlib import Path

import bids
import pandas as pd


class Dataset:
    """
    Class to handle fmriprep output

    Attributes
    ----------
    bids_layout
        Returns dataset layout
    sub_labels
        Returns a list of subject labels
    
    Methods
    -------
    get_func_files(sub)
        Returns list of functional files paths

    """

    def __init__(self, derivatives_path: str, TR: float,
                 sessions: int, runs: int, task: str):
        """ 
        Parameters
        ----------
        derivatives_path: str
            Path to derivatives directory
        runs: int
            Number of runs in one folder
        task: str
            Task name
        """

        self.derivatives = Path(derivatives_path).as_posix()
        self.runs = runs
        self.sessions = sessions
        self.task = task
        self.t_r = TR
        self.bids_layout = bids.BIDSLayout(
            self.derivatives, validate=False, config=['bids', 'derivatives'])
        self.sub_labels = self.bids_layout.get_subjects() #ok


    def get_confounds_one_subject(self, sub=None):
        """
        Get confounds dataframe

        Parameters
        ----------
        sub: str
            Subject label
        
        Returns
        -------
        list of pd.DataFrame
            List with dataframes for each run for one subject
        """
        conf_paths = self.bids_layout.get(subject=sub,
                                          extension='tsv',
                                          return_type='file')
        #print(len(conf_paths))
        return [pd.read_csv(conf_paths[i], sep='\t') for i in range(len(conf_paths))]

    def get_func_files(self, sub=None):
        """
        Get functional files' paths

        Parameters
        ----------
        sub: list of str, optional
            Subject labels to get files for. If None, all subjects are returned
        
        Returns
        -------
        list of str
        """

        if sub is None:
            sub = self.sub_labels

        return self.bids_layout.get(subject=sub,
                                    datatype='func',
                                    task=self.task,
                                    suffix="bold",
                                    extension='nii.gz',
                                    return_type='file')
    
    
    def get_mask_files(self, sub=None):
        """
        Get functional files' paths

        Parameters
        ----------
        sub: list of str, optional
            Subject labels to get files for. If None, all subjects are returned
        
        Returns
        -------
        list of str
        """

        if sub is None:
            sub = self.sub_labels
        return self.bids_layout.get(subject=sub,
                                    datatype='func',
                                    task=self.task,
                                    desc='brain',
                                    #space='MNI152NLin2009cAsym',
                                    extension='nii.gz',
                                    return_type='file')