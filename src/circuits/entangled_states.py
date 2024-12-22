from sklearn.pipeline import Pipeline, make_pipeline

from src.layers import QuantumLayer
from src.transformers import XTransformer, HTransformer, ZTransformer, CXTransformer

# TODO Convert to fastcore Transform/Pipeline
class BellStates:
    """
    Convenience class for generating Bell States as skq Pipelines.
    More information on defining Bell States:
    - https://quantumcomputinguk.org/tutorials/introduction-to-bell-states
    - https://quantumcomputing.stackexchange.com/a/2260
    """
    def get_bell_state(self, configuration: int = 1) -> Pipeline:
        """
        Return the pipeline for the Bell State based on the configuration.
        :param configuration: Configuration of the Bell State.
        Configuration 1: |Φ+⟩ =|00> + |11> / sqrt(2)
        Configuration 2: |Φ-⟩ =|00> - |11> / sqrt(2)
        Configuration 3: |Ψ+⟩ =|01> + |10> / sqrt(2)
        Configuration 4: |Ψ-⟩ =|01> - |10> / sqrt(2)
        :return: Pipeline for the Bell State.
        NOTE: The Bell State pipeline is returned without a MeasurementTransformer at the end.
        """
        assert configuration in [1, 2, 3, 4], f"Invalid Bell State configuration: {configuration}. Configurations are: 1: |Φ+⟩, 2: |Φ-⟩, 3: |Ψ+⟩, 4: |Ψ-⟩"
        config_mapping = {
            1: self.get_bell_state_omega_plus,
            2: self.get_bell_state_omega_minus,
            3: self.get_bell_state_phi_plus,
            4: self.get_bell_state_phi_minus,
        }
        pipe = config_mapping[configuration]()
        return pipe

    def get_bell_state_omega_plus(self) -> Pipeline:
        """
        Return pipeline for the entangled state |Φ−⟩ =|00> + |11> / sqrt(2).
        This corresponds to the 1st bell state.
        :return: Circuit for creating the 1st Bell State.
        """
        return Pipeline([
        ('H', QuantumLayer([
            ('H', HTransformer(qubits=[0])),
        ], n_qubits=2),
        ),
        ('CNOT', CXTransformer(qubits=[0, 1])),
        ])
    
    def get_bell_state_omega_minus(self) -> Pipeline:
        """
        Return pipeline for the entangled state |Φ−⟩ =|00> - |11> / sqrt(2).
        This corresponds to the 2nd bell state.
        :return: Circuit for creating the 2nd Bell State
        
        """
        omega_plus = self.get_bell_state_omega_plus()
        phase_flip = Pipeline([
            ('Z', QuantumLayer([
                ('Z', ZTransformer(qubits=[0])),
            ], n_qubits=2),
            ),
        ])
        return make_pipeline(omega_plus, phase_flip)
    
    def get_bell_state_phi_plus(self) -> Pipeline:
        """
        Return pipeline for the entangled state  |Ψ+⟩ =|01> + |10> / sqrt(2).
        This corresponds to the 3rd bell state.
        :return: Circuit for creating the 3rd Bell State
        """
        omega_plus = self.get_bell_state_omega_plus()
        bit_flip = Pipeline([
            ('X', QuantumLayer([
                ('X', XTransformer(qubits=[1])),
            ], n_qubits=2),
            ),
        ])
        return make_pipeline(omega_plus, bit_flip)
    
    def get_bell_state_phi_minus(self) -> Pipeline:
        """
        Return pipeline for the entangled state |Ψ−⟩ =|01> - |10> / sqrt(2).
        This corresponds to the 4th bell state.
        :return: Circuit for creating the 4th Bell State
        """
        omega_plus = self.get_bell_state_omega_plus()
        phase_flip = Pipeline([
            ('ZX', QuantumLayer([
                ('Z', ZTransformer(qubits=[0])),
                ('X', XTransformer(qubits=[1])),
            ], n_qubits=2),
            ),
        ])
        return make_pipeline(omega_plus, phase_flip)

class GHZStates:
    """
    Generalization of Bell States to 3 or more qubits.
    Greenberger, Horne, and Zeilinger states.
    """
    # TODO Implement GHZ states for arbitrary number of qubits
    ...


class WState:
    """
    1 / sqrt(3) (|001⟩ + |010⟩ + |100⟩)
    """
    # TODO Implement W state 
    ...
