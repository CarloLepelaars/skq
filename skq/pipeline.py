from sklearn.pipeline import Pipeline, _name_estimators

class QuantumPipeline(Pipeline):
    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def visualize_bloch_spheres(self):
        # Get Bloch spheres after each quantum transformer for each qubit
        ...

    def _sk_visual_block_(self):
        ...


def make_quantum_pipeline(*steps, memory=None, verbose=False) -> QuantumPipeline:
    """ 
    Convenience function for creating a QuantumPipeline. 
    :param steps: List of (name, transform) tuples (implementing fit/transform) that are chained.
    :param memory: Used to cache the fitted transformers of the pipeline.
    :param verbose: If True, the time elapsed while fitting each step will be printed as it is completed.
    :return: QuantumPipeline object
    """
    return QuantumPipeline(_name_estimators(steps), memory=memory, verbose=verbose)
