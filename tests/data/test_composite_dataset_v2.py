import os
import tempfile
import unittest
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf

from optimus_dl.core.registry import RegistryConfig
from optimus_dl.modules.data.datasets import register_dataset
from optimus_dl.modules.data.datasets.base import BaseDataset
from optimus_dl.modules.data.datasets.composite import (
    CompositeDataset,
    CompositeDatasetConfig,
    DatasetConfig,
    StopCriteria,
)
from optimus_dl.modules.data.datasets.txt_lines import TxtLinesDatasetConfig


# Define a mock dataset for testing
@dataclass
class MockDatasetConfig(RegistryConfig):
    size: int = 10
    start_value: int = 0


@register_dataset("mock_dataset", MockDatasetConfig)
class MockDataset(BaseDataset):
    def __init__(self, cfg, rank=0, world_size=1, **kwargs):
        super().__init__(cfg)
        self.size = cfg.size
        self.start_value = cfg.start_value
        self.current = 0

    def reset(self, initial_state=None):
        super().reset(initial_state)
        if initial_state:
            self.current = initial_state["current"]
        else:
            self.current = 0

    def next(self):
        if self.current >= self.size:
            raise StopIteration

        val = self.start_value + self.current
        self.current += 1
        return val

    def get_state(self):
        return {"current": self.current}


class TestCompositeDataset(unittest.TestCase):
    def setUp(self):
        # Configure datasets
        # Wrap with OmegaConf.structured so they are treated as configs by registry.build
        self.ds1_cfg = OmegaConf.structured(
            MockDatasetConfig(size=5, start_value=0, _name="mock_dataset")
        )
        self.ds2_cfg = OmegaConf.structured(
            MockDatasetConfig(size=3, start_value=100, _name="mock_dataset")
        )

        self.comp_cfg = CompositeDatasetConfig(
            datasets={
                "ds1": DatasetConfig(dataset=self.ds1_cfg, weight=1.0, cycle=False),
                "ds2": DatasetConfig(dataset=self.ds2_cfg, weight=1.0, cycle=False),
            },
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,
            strict_load=True,
        )
        self.temp_files = []

    def tearDown(self):
        for f in self.temp_files:
            if os.path.exists(f):
                os.remove(f)

    def test_weighted_sampling_and_exhaustion(self):
        # ds1 has 5 items (0-4), ds2 has 3 items (100-102)
        # Weights equal. Should mix.
        # cycle=False, so they should exhaust.

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            while True:
                items.append(dataset.next())
        except StopIteration:
            pass

        # Check if we got all items
        ds1_items = [i for i in items if i < 100]
        ds2_items = [i for i in items if i >= 100]

        self.assertEqual(len(ds1_items), 5)
        self.assertEqual(len(ds2_items), 3)
        self.assertEqual(sorted(ds1_items), list(range(5)))
        self.assertEqual(sorted(ds2_items), list(range(100, 103)))

    def test_cycling(self):
        # ds1 cycles, ds2 does not.
        # ds1: 2 items, ds2: 2 items.
        # stop when FIRST_DATASET_EXHAUSTED (ds2 will exhaust)

        self.comp_cfg.datasets["ds1"].cycle = True
        # Update wrapped config
        self.comp_cfg.datasets["ds1"].dataset.size = 2
        self.comp_cfg.datasets["ds2"].dataset.size = 2
        self.comp_cfg.stop_criteria = StopCriteria.FIRST_DATASET_EXHAUSTED

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            for _ in range(10):  # Limit to avoid infinite loop if bug
                items.append(dataset.next())
        except StopIteration:
            pass

        # Expect ds2 to exhaust after 2 items.
        # ds1 might yield more than 2 items due to cycling.

        [i for i in items if i < 100]
        ds2_items = [i for i in items if i >= 100]

        self.assertEqual(len(ds2_items), 2)

    def test_state_restoration(self):
        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Consume some items
        items_1 = [dataset.next() for _ in range(3)]

        # Save state
        state = dataset.get_state()

        # Create new dataset and load state
        dataset2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Continue
        items_2 = []
        try:
            while True:
                items_2.append(dataset2.next())
        except StopIteration:
            pass

        # Verify total items
        total_items = items_1 + items_2
        ds1_items = [i for i in total_items if i < 100]
        ds2_items = [i for i in total_items if i >= 100]

        self.assertEqual(len(ds1_items), 5)
        self.assertEqual(len(ds2_items), 3)

    def test_state_resume_exact_match(self):
        # Test that resuming produces exactly the same sequence as continuing
        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Run N steps
        [dataset.next() for _ in range(3)]

        # Save state
        state = dataset.get_state()

        # Run M more steps (Path A)
        continued_items = [dataset.next() for _ in range(4)]

        # Restore state
        dataset2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Run M more steps (Path B)
        restored_items = [dataset2.next() for _ in range(4)]

        self.assertEqual(
            continued_items,
            restored_items,
            "Resuming from state should match continuing execution exactly",
        )

    def test_exhausted_weight_restoration(self):
        # ds2 has 3 items. Run until it exhausts.
        # It shouldn't cycle.

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Consume until ds2 exhausts.
        # Since we don't know order, let's run until we see 3 items from ds2.
        count_ds2 = 0
        while count_ds2 < 3:
            item = dataset.next()
            if item >= 100:
                count_ds2 += 1

        # Now ds2 might be exhausted or about to check exhaustion on next sample.
        # Force one more sample to ensure exhaustion logic triggers if needed
        # (If ds2 was just yielded, we need to call next() again to hit StopIteration inside sub-dataset)
        # But we can't guarantee next sample is ds2.
        # Let's check state.
        state = dataset.get_state()

        # If ds2 is exhausted, its weight in sampler should be 0.
        # But wait, exhaustion happens lazily when next() is called and raises StopIteration.
        # If we just consumed the 3rd item, the dataset is empty but doesn't know it yet until next() call.

        # So let's consume ALL items.
        try:
            while True:
                dataset.next()
        except StopIteration:
            pass

        # Now all exhausted.
        # If we load this state, weights should be 0.
        state = dataset.get_state()

        dataset2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Check sampler weights in dataset2
        # Accessing private member for verification
        sampler = dataset2._weighted_sampler
        # ds1 (idx 0) and ds2 (idx 1) should be 0.0 because ALL exhausted.
        self.assertTrue(
            torch.all(sampler.weights == 0.0),
            "Weights should be 0.0 for exhausted datasets in restored state",
        )

    def test_cycle_state_restoration(self):
        # ds1 cycles. Run until it cycles at least once.
        self.comp_cfg.datasets["ds1"].cycle = True
        self.comp_cfg.datasets["ds1"].dataset.size = 2

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        # ds1 size 2. Consume 3 ds1 items.
        ds1_count = 0
        while ds1_count < 3:
            item = dataset.next()
            items.append(item)
            if item < 100:
                ds1_count += 1

        # Save state
        state = dataset.get_state()

        # Restore
        dataset2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Continue
        dataset2.next()

        # Verify continuity?
        # Hard to verify strictly without knowing exact sequence, but it shouldn't crash.
        pass

    def test_dynamic_weight_update(self):
        # ds1: 10 items, weight=10
        # ds2: 1 item, weight=1
        # Stop: ALL_DATASETS_EXHAUSTED
        # Initially ds1 picked 10x more likely.
        # Once ds2 exhausts, sampler should ONLY pick ds1.

        self.comp_cfg.datasets["ds1"].weight = 10.0
        self.comp_cfg.datasets["ds1"].dataset.size = 10
        self.comp_cfg.datasets["ds2"].weight = 0.1  # Very low weight
        self.comp_cfg.datasets["ds2"].dataset.size = 1

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            while True:
                items.append(dataset.next())
        except StopIteration:
            pass

        self.assertEqual(len(items), 11)
        self.assertIn(100, items)  # ds2 item

    def test_strict_load(self):
        # Train model with 2 datasets
        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()
        dataset.next()
        state = dataset.get_state()

        # Create new config with 3 datasets
        self.comp_cfg.datasets["ds3"] = DatasetConfig(
            dataset=OmegaConf.structured(
                MockDatasetConfig(size=5, start_value=200, _name="mock_dataset")
            ),
            weight=1.0,
        )

        # Strict load = True -> Should fail
        self.comp_cfg.strict_load = True
        dataset_strict = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        with self.assertRaises(ValueError):
            dataset_strict.reset(state)

        # Strict load = False -> Should succeed
        self.comp_cfg.strict_load = False
        dataset_loose = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        try:
            dataset_loose.reset(state)
        except ValueError:
            self.fail("reset() raised ValueError unexpectedly with strict_load=False")

    def test_seed_determinism(self):
        # Run 1
        ds1 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        ds1.reset()
        # Increased sample size to reduce collision probability
        items1 = [ds1.next() for _ in range(8)]

        # Run 2 (same seed)
        ds2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        ds2.reset()
        items2 = [ds2.next() for _ in range(8)]

        # Run 3 (diff seed)
        ds3 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=421)
        ds3.reset()
        items3 = [ds3.next() for _ in range(8)]

        self.assertEqual(items1, items2, "Same seed should produce same sequence")
        self.assertNotEqual(
            items1, items3, "Different seed should produce different sequence"
        )

    def test_stop_criteria_cycle_forever(self):
        # ds1, ds2 both cycle=True, small size.
        self.comp_cfg.datasets["ds1"].cycle = True
        self.comp_cfg.datasets["ds1"].dataset.size = 2
        self.comp_cfg.datasets["ds2"].cycle = True
        self.comp_cfg.datasets["ds2"].dataset.size = 2
        self.comp_cfg.stop_criteria = StopCriteria.CYCLE_FOREVER

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        # Pull many more than capacity
        for _ in range(20):
            items.append(dataset.next())

        self.assertEqual(len(items), 20)
        # Verify we see items cycling
        self.assertTrue(any(i < 100 for i in items))
        self.assertTrue(any(i >= 100 for i in items))

    def test_single_dataset(self):
        # Only ds1
        single_cfg = CompositeDatasetConfig(
            datasets={
                "ds1": DatasetConfig(dataset=self.ds1_cfg, weight=1.0, cycle=False),
            },
            strict_load=True,
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,  # Fixed criteria
        )
        dataset = CompositeDataset(single_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            while True:
                items.append(dataset.next())
        except StopIteration:
            pass

        self.assertEqual(len(items), 5)
        self.assertEqual(items, list(range(5)))

    def test_sampling_distribution(self):
        # ds1 weight 4.0, ds2 weight 1.0.
        # Both cycling to provide infinite stream.
        self.comp_cfg.datasets["ds1"].weight = 4.0
        self.comp_cfg.datasets["ds1"].cycle = True
        # Update config properly
        self.comp_cfg.datasets["ds1"].dataset.size = 1000
        self.comp_cfg.datasets["ds1"].dataset.start_value = 0

        self.comp_cfg.datasets["ds2"].weight = 1.0
        self.comp_cfg.datasets["ds2"].cycle = True
        self.comp_cfg.datasets["ds2"].dataset.size = 1000
        self.comp_cfg.datasets["ds2"].dataset.start_value = 10000

        self.comp_cfg.stop_criteria = StopCriteria.CYCLE_FOREVER

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        n_samples = 2000
        items = [dataset.next() for _ in range(n_samples)]

        count_ds1 = sum(1 for i in items if i < 10000)
        count_ds2 = sum(1 for i in items if i >= 10000)

        ratio_ds1 = count_ds1 / n_samples
        expected_ratio = 4.0 / 5.0  # 0.8

        # Tolerance 0.05
        self.assertTrue(
            abs(ratio_ds1 - expected_ratio) < 0.05,
            f"Expected ratio ~0.8, got {ratio_ds1:.3f} (ds1={count_ds1}, ds2={count_ds2})",
        )

    def test_real_txt_dataset(self):
        # Create temp files
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f1:
            f1.write("A1\nA2\nA3\n")
            f1_path = f1.name
            self.temp_files.append(f1_path)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f2:
            f2.write("B1\nB2\n")
            f2_path = f2.name
            self.temp_files.append(f2_path)

        # Configure datasets
        txt_cfg1 = OmegaConf.structured(
            TxtLinesDatasetConfig(file_link=f1_path, _name="txt_lines")
        )
        txt_cfg2 = OmegaConf.structured(
            TxtLinesDatasetConfig(file_link=f2_path, _name="txt_lines")
        )

        cfg = CompositeDatasetConfig(
            datasets={
                "txt1": DatasetConfig(dataset=txt_cfg1, weight=1.0, cycle=False),
                "txt2": DatasetConfig(dataset=txt_cfg2, weight=1.0, cycle=False),
            },
            stop_criteria=StopCriteria.ALL_DATASETS_EXHAUSTED,
            strict_load=True,
        )

        dataset = CompositeDataset(cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            while True:
                item = dataset.next()
                items.append(item["text"])
        except StopIteration:
            pass

        # Verify
        expected = sorted(["A1", "A2", "A3", "B1", "B2"])
        actual = sorted(items)
        self.assertEqual(actual, expected)

    def test_mixed_cycling_first_exhausted(self):
        # ds1 cycles (size 2), ds2 no cycle (size 2).
        # StopCriteria: FIRST_DATASET_EXHAUSTED.
        # Should stop when ds2 exhausts (after 2 items).

        self.comp_cfg.datasets["ds1"].cycle = True
        self.comp_cfg.datasets["ds1"].dataset.size = 2
        self.comp_cfg.datasets["ds2"].cycle = False
        self.comp_cfg.datasets["ds2"].dataset.size = 2
        self.comp_cfg.stop_criteria = StopCriteria.FIRST_DATASET_EXHAUSTED

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        try:
            for _ in range(20):
                items.append(dataset.next())
        except StopIteration:
            pass

        ds2_items = [i for i in items if i >= 100]
        self.assertEqual(
            len(ds2_items),
            2,
            "ds2 should yield exactly 2 items before exhaustion triggers stop",
        )

        # ds1 might have yielded > 2 if it was picked often before ds2 exhausted
        # But crucially, we shouldn't infinite loop.

    def test_mixed_cycling_all_exhausted(self):
        # ds1 cycles (size 2), ds2 no cycle (size 2).
        # StopCriteria: ALL_DATASETS_EXHAUSTED.
        # Since ds1 cycles, it never exhausts in the sense of stopping the loop.
        # ds2 should exhaust and stop yielding.
        # Loop should continue indefinitely (we break manually).

        self.comp_cfg.datasets["ds1"].cycle = True
        self.comp_cfg.datasets["ds1"].dataset.size = 2
        self.comp_cfg.datasets["ds2"].cycle = False
        self.comp_cfg.datasets["ds2"].dataset.size = 2
        self.comp_cfg.stop_criteria = StopCriteria.ALL_DATASETS_EXHAUSTED

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        items = []
        # Consume enough to ensure ds2 exhausts and ds1 cycles multiple times
        for _ in range(50):
            items.append(dataset.next())

        ds2_items = [i for i in items if i >= 100]
        self.assertEqual(len(ds2_items), 2, "ds2 should stop yielding after 2 items")

        ds1_items = [i for i in items if i < 100]
        self.assertEqual(len(ds1_items), 48, "ds1 should continue yielding")
        # ds1 size 2 -> items 0,1. 48 items means cycled 24 times.

    def test_mixed_cycling_state_restoration(self):
        # ds1 cycles (size 2), ds2 no cycle (size 2).
        # Run until ds2 exhausted. Save state. Restore.
        # Ensure ds2 remains exhausted and ds1 continues.

        self.comp_cfg.datasets["ds1"].cycle = True
        self.comp_cfg.datasets["ds1"].dataset.size = 2
        self.comp_cfg.datasets["ds2"].cycle = False
        self.comp_cfg.datasets["ds2"].dataset.size = 2
        self.comp_cfg.stop_criteria = StopCriteria.ALL_DATASETS_EXHAUSTED

        dataset = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset.reset()

        # Consume until ds2 exhausted (seen 2 items)
        count_ds2 = 0
        while count_ds2 < 2:
            item = dataset.next()
            if item >= 100:
                count_ds2 += 1

        for _ in range(10):
            item = dataset.next()
            self.assertTrue(
                item < 100, "Should only yield ds1 items after ds2 exhausted"
            )

        # Save state
        state = dataset.get_state()

        # Restore
        dataset2 = CompositeDataset(self.comp_cfg, rank=0, world_size=1, seed=42)
        dataset2.reset(state)

        # Check weights immediately
        sampler = dataset2._weighted_sampler
        # ds1 (idx 0) should be > 0. ds2 (idx 1) should be 0.
        self.assertTrue(sampler.weights[0] > 0, "ds1 weight should be active")
        self.assertTrue(sampler.weights[1] == 0, "ds2 weight should be 0 (exhausted)")

        # Verify continued execution
        for _ in range(10):
            item = dataset2.next()
            self.assertTrue(item < 100, "Restored dataset should only yield ds1 items")


if __name__ == "__main__":
    unittest.main()
