"""Run: python3 -m unittest test_audit"""
import unittest
from audit import Workload, analyse, cost, load_pricing, normalize_model

M = load_pricing()["models"]

def wl(**kw):
    base = dict(feature="f", model_raw=kw.get("model", "claude-sonnet-4-6"), model=kw.pop("model", "claude-sonnet-4-6"),
                requests=1000, avg_in=2000, avg_out=500)
    base.update(kw)
    return Workload(**base)

class AuditTest(unittest.TestCase):
    def test_cost_matches_hand_math(self):
        # 1000 calls x (2000 in x $3 + 500 out x $15) / 1e6 = $13.50
        self.assertAlmostEqual(cost(wl(), "claude-sonnet-4-6", M), 13.5)
        self.assertAlmostEqual(cost(wl(), "claude-sonnet-4-6", M, batch=True), 6.75)

    def test_cached_tokens_priced_at_cache_rate(self):
        w = wl(cached_share=0.5)  # 1000 fresh x $3 + 1000 cached x $0.30
        self.assertAlmostEqual(cost(w, "claude-sonnet-4-6", M), (1000 * 3 + 1000 * 0.3 + 500 * 15) / 1000)

    def test_normalize_snapshot_names(self):
        self.assertEqual(normalize_model("claude-sonnet-4-6-20260115", M), "claude-sonnet-4-6")
        self.assertEqual(normalize_model("gpt-4o-mini-2024-07-18", M), "gpt-4o-mini")
        self.assertIsNone(normalize_model("llama-3-70b", M))

    def test_never_downshifts_hard_tasks(self):
        f = analyse(wl(model="claude-opus-5-5", task_type="agent"), M)
        self.assertFalse(any(l.kind == "model" for l in f.levers))

    def test_downshift_only_in_high_estimate(self):
        f = analyse(wl(task_type="classification", latency_sensitive=True), M)
        lever = next(l for l in f.levers if l.kind == "model")
        self.assertTrue(lever.needs_test)
        self.assertLess(lever.low, lever.high)

    def test_savings_never_exceed_spend(self):
        f = analyse(wl(task_type="classification", latency_sensitive=False, static_prefix=1500), M)
        self.assertLessEqual(f.high, f.current)
        self.assertLessEqual(f.low, f.high)

    def test_short_prefix_not_cached(self):
        f = analyse(wl(static_prefix=500, latency_sensitive=True), M)
        self.assertFalse(any(l.kind == "cache" for l in f.levers))

if __name__ == "__main__":
    unittest.main()
