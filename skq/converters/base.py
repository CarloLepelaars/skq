from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

from skq.transformers import SingleQubitTransformer, MultiQubitTransformer, MeasurementTransformer


class QuantumCircuitConverter:
    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
    
    def process_transformer(self, transformer, *args, **kwargs):
        """ Recursive setup to convert skq circuit to a target framework. """
        if isinstance(transformer, ColumnTransformer):
            raise NotImplementedError("The usage of ColumnTransformer for converting circuits is not supported yet.")
        elif isinstance(transformer, Pipeline):
            for _, step in transformer.steps:
                self.process_transformer(step, *args, **kwargs)
        elif isinstance(transformer, FeatureUnion):
            for _, step in transformer.transformer_list:
                self.process_transformer(step, *args, **kwargs)
        elif isinstance(transformer, (SingleQubitTransformer, MultiQubitTransformer)):
            if transformer.__class__.__name__ == "ITransformer":
                return
            self.handle_gate(transformer, *args, **kwargs)
        elif isinstance(transformer, MeasurementTransformer):
            self.handle_measurement(*args, **kwargs)
        else:
            return
    
    def handle_gate(self, transformer, *args, **kwargs):
        """ Convert and add skq gates to the target framework. """
        raise NotImplementedError("handle_gate must be implemented by converter.")

    def handle_measurement(self, *args, **kwargs):
        """ Convert and add skq measurements to the target framework. """
        raise NotImplementedError("handle_measurement must be implemented by converter.")

    def convert(self, *args, **kwargs):
        for _, step in self.pipeline.steps:
            self.process_transformer(step, *args, **kwargs)
        self.finalize_conversion(*args, **kwargs)

    def finalize_conversion(self, *args, **kwargs):
        pass
