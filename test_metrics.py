import numpy as np
import pandas as pd
from src.metrics import icc_matrix, strategies_comparison
from nilearn.connectome import vec_to_sym_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def test_strategies_comparison():
    data = np.random.rand(20, 4950)
    data = vec_to_sym_matrix(data, diagonal=np.zeros((20, 100)))
    
    data = np.array([data + np.random.normal(0, i/100, (100, 100)) for i in range(1, 13)])

    test_matrix = pd.read_excel('/home/tm/projects/BenchmarkingEOEC/tests/modularity.xlsx', names=[i for i in range(100)])
    cond1 = data[:, :10] * test_matrix.values
    cond2 = data[:, 10:] * ~test_matrix.values

    result = strategies_comparison(cond1, cond2)


test_strategies_comparison()