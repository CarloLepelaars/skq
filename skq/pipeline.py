import numpy as np
from sklearn.pipeline import Pipeline, _name_estimators, FeatureUnion

class QuantumPipeline(Pipeline):
    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def visualize_bloch_spheres(self):
        # Get Bloch spheres after each quantum transformer for each qubit
        ...

    def _sk_visual_block_(self):
        ...

class QuantumFeatureUnion(FeatureUnion):
    def __init__(self, transformer_list, **kwargs):
        super().__init__(transformer_list, **kwargs)

    def transform(self, X):
        combined_gate = self.transformer_list[0][1].gate
        for _, transformer in self.transformer_list[1:]:
            combined_gate = np.kron(combined_gate, transformer.gate)
        
        # Apply the combined gate to the input state vectors
        transformed_X = np.array([combined_gate @ x for x in X])
        return transformed_X

def make_quantum_pipeline(*steps, memory=None, verbose=False) -> QuantumPipeline:
    """ 
    Convenience function for creating a QuantumPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    :return: QuantumPipeline object
    """
    return QuantumPipeline(_name_estimators(steps), memory=memory, verbose=verbose)

def make_quantum_union(*transformers, n_jobs=None, verbose=False) -> QuantumFeatureUnion:
    """
    Convenience function for creating a QuantumFeatureUnion.
    :param transformers: List of (name, transform) tuples (implementing fit/transform) that are concatenated.
    :param kwargs: Additional arguments to pass to the QuantumFeatureUnion constructor.
    :return: QuantumFeatureUnion object
    """
    return QuantumFeatureUnion(_name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
