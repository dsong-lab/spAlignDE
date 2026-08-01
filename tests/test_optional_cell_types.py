import inspect
import unittest

import pandas as pd

from spAlignDE import PreparedInference, fit_local_de


class OptionalCellTypeTests(unittest.TestCase):
    def test_cell_type_adjustment_is_opt_in(self):
        parameter = inspect.signature(fit_local_de).parameters[
            "cell_type_adjustment"
        ]
        self.assertIs(parameter.default, False)

    def test_cell_type_adjustment_requires_available_annotations(self):
        prepared = PreparedInference(
            shared={},
            data=pd.DataFrame(),
            genes=("gene_a",),
            reference="reference",
            library_size=None,
            density_energy_share=0.25,
            alignment_uncertainty_key=None,
            metadata={"cell_type_available": False},
        )

        with self.assertRaisesRegex(ValueError, "complete cell-type annotation"):
            fit_local_de(
                prepared,
                genes=["gene_a"],
                cell_type_adjustment=True,
            )


if __name__ == "__main__":
    unittest.main()
