"""
Analysis methods registry.

Each method module must have:
  - run(svd_data, weight_types, n_layers, output_dir): main function
  - DESCRIPTION: one-line string describing the method

To add a new method:
  1. Create methods/your_method.py
  2. Import it below
  3. Add it to AVAILABLE_METHODS
"""

from . import structure
from . import spectra
from . import stable_rank
from . import effective_rank
from . import alpha
from . import distributions
from . import norms
from . import principal_angles
from . import alignment
from . import cumulative_energy
from . import moe_experts

# Method registry — order determines default execution order
AVAILABLE_METHODS = {
    "structure":         structure,
    "spectra":           spectra,
    "stable_rank":       stable_rank,
    "effective_rank":    effective_rank,
    "alpha":             alpha,
    "distributions":     distributions,
    "norms":             norms,
    "principal_angles":  principal_angles,
    "alignment":         alignment,
    "cumulative_energy": cumulative_energy,
    "moe_experts":       moe_experts,
}
