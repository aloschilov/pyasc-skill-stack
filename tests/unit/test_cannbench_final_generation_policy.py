"""Final publication gate rejects review-mutated and non-GLM candidates."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from test_cannbench_resume import build_recorded_iteration, make_catalog, driver, generation_gate


class FinalGenerationPolicyTests(unittest.TestCase):
    def check(self, mutation=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = make_catalog(["exp"])
            iteration, provenance = build_recorded_iteration(root / "exp-generate", catalog["operators"][0])
            provenance['readonly_review'] = True
            for phase in ('design', 'implement', 'review'):
                provenance['models'][phase] = 'gateway/glm-5.3'
                trace_path = iteration / (phase + '.skill-trace.attempt1.json')
                trace = json.loads(trace_path.read_text())
                trace['model'] = 'gateway/glm-5.3'
                trace_path.write_text(json.dumps(trace))
            if mutation:
                mutation(provenance, iteration)
            (iteration / "provenance.json").write_text(json.dumps(provenance))
            qualified = root / "locally_qualified"
            for sub in ["cann_bench", "provenance"]:
                (qualified / sub).mkdir(parents=True)
            shutil.copy2(iteration / "candidate.py", qualified / "cann_bench/exp.py")
            shutil.copy2(iteration / "provenance.json", qualified / "provenance/exp.json")
            (root / "summary.json").write_text(json.dumps({"requested_operators": ["exp"], "locally_qualified_operators": ["exp"], "catalog_sha256": catalog["catalog_sha256"]}))
            return generation_gate.validate_generation(catalog, root, driver.SKILLS_ROOT)

    def test_unchanged_all_glm_candidate_accepted(self):
        self.assertIn("exp", self.check())

    def test_review_mutation_rejected_even_when_final_compile_passed(self):
        with self.assertRaisesRegex(ValueError, "unchanged GLM"):
            self.check(lambda p, _: p["reviewed_compile"].update(candidate_sha256="different-before-review"))

    def test_non_glm_implementation_rejected_with_valid_independent_trace(self):
        def mutate(p, iteration):
            p["models"]["implement"] = "gateway/other"
            path = iteration / "implement.skill-trace.attempt1.json"
            trace = json.loads(path.read_text())
            trace["model"] = "gateway/other"
            path.write_text(json.dumps(trace))
        with self.assertRaisesRegex(ValueError, "unchanged GLM"):
            self.check(mutate)

    def test_explicit_implementation_binding_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "unchanged GLM"):
            self.check(lambda p, _: p.update(implementation_sha256="different-implementation"))


if __name__ == "__main__":
    unittest.main()
