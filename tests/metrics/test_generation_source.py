from unittest.mock import MagicMock

import torch

from optimus_dl.modules.metrics.source import StandardProtocols
from optimus_dl.modules.metrics.sources.generation import (
    GenerationSource,
    GenerationSourceConfig,
)


class TestGenerationSource:
    def test_greedy_generation(self):
        cfg = GenerationSourceConfig(max_new_tokens=5, do_sample=False)
        source = GenerationSource(cfg)

        batch = {"input_ids": torch.tensor([[1, 2, 3]])}

        # Mock model that always predicts the next token as (current_last_token + 1)
        def model_side_effect(**kwargs):
            input_ids = kwargs["input_ids"]
            # logits: [B, T, V], let's say V=20
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 20)
            for b in range(batch_size):
                for t in range(seq_len):
                    next_val = (input_ids[b, t].item() + 1) % 20
                    logits[b, t, next_val] = 10.0
            return {"logits": logits}

        model = MagicMock(side_effect=model_side_effect)

        result = source(dependencies={}, model=model, batch=batch)

        generated = result[StandardProtocols.GENERATED_TOKENS]
        # Input was [1, 2, 3].
        # Step 1: predicts 4. curr_ids=[1, 2, 3, 4]
        # Step 2: predicts 5. curr_ids=[1, 2, 3, 4, 5]
        # ...
        # Generated should be [4, 5, 6, 7, 8]
        expected = torch.tensor([[4, 5, 6, 7, 8]])
        assert torch.equal(generated, expected)
        assert model.call_count == 5

    def test_eos_handling(self):
        # max_new_tokens=10, but should stop at 5 because of EOS
        cfg = GenerationSourceConfig(max_new_tokens=10, do_sample=False, eos_token_id=7)
        source = GenerationSource(cfg)

        # Batch with two sequences
        batch = {"input_ids": torch.tensor([[1, 2, 3], [10, 11, 12]])}

        def model_side_effect(**kwargs):
            input_ids = kwargs["input_ids"]
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros(batch_size, seq_len, 20)
            for b in range(batch_size):
                # Simple logic: next token is last + 1
                next_val = (input_ids[b, -1].item() + 1) % 20
                logits[b, -1, next_val] = 10.0
            return {"logits": logits}

        model = MagicMock(side_effect=model_side_effect)

        result = source(dependencies={}, model=model, batch=batch)

        generated = result[StandardProtocols.GENERATED_TOKENS]
        # Seq 1: 4, 5, 6, 7 (EOS) -> stop extending
        # Seq 2: 13, 14, 15, 16, 17, 18, 19, 0, 1, 2 (doesn't hit 7)
        # Loop should continue until all are finished or max_new_tokens

        assert generated.shape == (2, 10)
        # Check Seq 1 stopped at index 3 (value 7) and padded with 7
        assert torch.equal(generated[0, :4], torch.tensor([4, 5, 6, 7]))
        assert (generated[0, 4:] == 7).all()

        # Check Seq 2 continued
        assert generated[1, 0].item() == 13

    def test_sampling_basic(self):
        # We can't easily test multinomial output deterministically,
        # but we can check if it runs without error.
        cfg = GenerationSourceConfig(
            max_new_tokens=2, do_sample=True, temperature=0.8, top_k=5, top_p=0.9
        )
        source = GenerationSource(cfg)

        batch = {"input_ids": torch.tensor([[1, 2, 3]])}
        model = MagicMock(return_value={"logits": torch.randn(1, 3, 20)})

        # Update mock to handle growing sequence
        def side_effect(**kwargs):
            t = kwargs["input_ids"].shape[1]
            return {"logits": torch.randn(1, t, 20)}

        model.side_effect = side_effect

        result = source(dependencies={}, model=model, batch=batch)
        assert result[StandardProtocols.GENERATED_TOKENS].shape == (1, 2)
