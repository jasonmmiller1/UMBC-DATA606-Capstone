from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.retrieval.retrieve import _policy_section_multiplier


class PolicyRetrievalWeightingTest(unittest.TestCase):
    def test_low_signal_sections_are_downranked_by_default(self) -> None:
        payload = {"source_type": "policy_pdf", "section_type": "purpose"}
        self.assertAlmostEqual(_policy_section_multiplier("access provisioning requirements", payload), 0.6)

    def test_low_signal_downrank_is_disabled_when_explicitly_requested(self) -> None:
        payload = {"source_type": "policy_pdf", "section_type": "scope"}
        self.assertAlmostEqual(_policy_section_multiplier("what is the scope of this policy", payload), 1.0)

    def test_high_signal_sections_are_upweighted(self) -> None:
        payload = {"source_type": "policy_pdf", "section_type": "requirements"}
        self.assertAlmostEqual(_policy_section_multiplier("least privilege controls", payload), 1.1)

    def test_non_policy_source_is_not_weighted(self) -> None:
        payload = {"source_type": "oscal_control", "section_type": "controls"}
        self.assertAlmostEqual(_policy_section_multiplier("AC-2 requirements", payload), 1.0)

    def test_weighting_can_be_disabled_via_env(self) -> None:
        payload = {"source_type": "policy_pdf", "section_type": "purpose"}
        with patch.dict(os.environ, {"RETRIEVAL_POLICY_SECTION_WEIGHTING_ENABLED": "0"}, clear=False):
            self.assertAlmostEqual(_policy_section_multiplier("purpose", payload), 1.0)


if __name__ == "__main__":
    unittest.main()
