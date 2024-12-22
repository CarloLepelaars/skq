import numpy as np
from sklearn.pipeline import Pipeline

from src.quantum_info import Statevector
from src.layers import SuperpositionLayer
from src.transformers import PhaseOracleTransformer, GroverDiffusionTransformer
from src.utils import _check_quantum_state_array


class GroverSearch(Pipeline):
    def __init__(self, n_qubits: int, target_state: np.ndarray, n_iterations: int):
        self.n_qubits = n_qubits
        self.target_state = Statevector(target_state) 
        self.n_iterations = n_iterations
        pipeline = self.create_pipeline()
        super().__init__(steps=pipeline.steps)

    def create_pipeline(self):
        steps = []
        qubits_list = list(range(self.n_qubits))
        steps.append(('superposition', SuperpositionLayer(n_qubits=self.n_qubits)))
        for _ in range(self.n_iterations):
            steps.append(('oracle', PhaseOracleTransformer(qubits=qubits_list, target_state=self.target_state)))
            steps.append(('diffusion', GroverDiffusionTransformer(qubits=qubits_list)))
        return Pipeline(steps=steps)

    def fit(self, X, y=None):
        _check_quantum_state_array(X)
        return super().fit(X, y)
    
    def transform(self, X):
        _check_quantum_state_array(X)
        return super().transform(X)
    