import unittest


class ImportTest(unittest.TestCase):
    def test_public_imports(self):
        import girsanov_uq
        from girsanov_uq.integrators import LangevinOBABO
        from girsanov_uq.post_processing.reweighting import Reweightor
        from girsanov_uq.potentials import (
            OneDimensionalPotentialCalculator,
            Polynomial1DPotential,
            RuggedMullerBrown,
            SolvatedDimer,
            TargetFunction1DPotential,
        )

        self.assertIsNotNone(girsanov_uq)
        self.assertIsNotNone(LangevinOBABO)
        self.assertIsNotNone(Reweightor)
        self.assertIsNotNone(OneDimensionalPotentialCalculator)
        self.assertIsNotNone(Polynomial1DPotential)
        self.assertIsNotNone(RuggedMullerBrown)
        self.assertIsNotNone(SolvatedDimer)
        self.assertIsNotNone(TargetFunction1DPotential)


if __name__ == "__main__":
    unittest.main()
