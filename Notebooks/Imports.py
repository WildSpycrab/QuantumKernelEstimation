from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit_aer.primitives import SamplerV2
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_aer.primitives import SamplerV2 as Sampler
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel
