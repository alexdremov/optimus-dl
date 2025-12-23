import importlib.util
import unittest

import pytest
import torch

from optimus_dl.modules.model.llama2 import Llama, LlamaConfig

LIGER_AVAILABLE = importlib.util.find_spec("liger_kernel") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not LIGER_AVAILABLE, reason="Liger Kernel not installed")
class TestLigerEquivalence(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda")
        self.config = LlamaConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            vocab_size=1000,
            sequence_length=128,
            use_liger_rmsnorm=False,
            use_liger_swiglu=False,
            use_liger_rope=False,
        )
        # Create base model (standard PyTorch)
        self.base_model = Llama(self.config).to(self.device)
        self.base_model.eval()

    def test_rmsnorm_equivalence(self):
        """Test that Liger RMSNorm produces same output as PyTorch implementation."""
        liger_config = LlamaConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            vocab_size=1000,
            sequence_length=128,
            use_liger_rmsnorm=True,
            use_liger_swiglu=False,
            use_liger_rope=False,
        )
        liger_model = Llama(liger_config).to(self.device)
        liger_model.load_state_dict(self.base_model.state_dict())
        liger_model.eval()

        input_ids = torch.randint(0, 1000, (4, 32), device=self.device)

        with torch.no_grad():
            base_out = self.base_model(input_ids)["logits"]
            liger_out = liger_model(input_ids)["logits"]

        torch.testing.assert_close(base_out, liger_out, rtol=1e-5, atol=1e-5)

    def test_swiglu_equivalence(self):
        """Test that Liger SwiGLU produces same output as PyTorch implementation."""
        liger_config = LlamaConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            vocab_size=1000,
            sequence_length=128,
            use_liger_rmsnorm=False,
            use_liger_swiglu=True,
            use_liger_rope=False,
        )
        liger_model = Llama(liger_config).to(self.device)
        liger_model.load_state_dict(self.base_model.state_dict())
        liger_model.eval()

        input_ids = torch.randint(0, 1000, (4, 32), device=self.device)

        with torch.no_grad():
            base_out = self.base_model(input_ids)["logits"]
            liger_out = liger_model(input_ids)["logits"]

        torch.testing.assert_close(base_out, liger_out, rtol=1e-5, atol=1e-5)

    def test_rope_equivalence(self):
        """Test that Liger RoPE produces same output as PyTorch implementation."""
        liger_config = LlamaConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            vocab_size=1000,
            sequence_length=128,
            use_liger_rmsnorm=False,
            use_liger_swiglu=False,
            use_liger_rope=True,
        )
        liger_model = Llama(liger_config).to(self.device)
        liger_model.load_state_dict(self.base_model.state_dict())
        liger_model.eval()

        input_ids = torch.randint(0, 1000, (4, 32), device=self.device)

        with torch.no_grad():
            base_out = self.base_model(input_ids)["logits"]
            liger_out = liger_model(input_ids)["logits"]

        torch.testing.assert_close(base_out, liger_out, rtol=1e-5, atol=1e-5)

    def test_full_liger_equivalence(self):
        """Test all Liger kernels enabled simultaneously."""
        liger_config = LlamaConfig(
            n_layer=2,
            n_head=4,
            n_embd=128,
            vocab_size=1000,
            sequence_length=128,
            use_liger_rmsnorm=True,
            use_liger_swiglu=True,
            use_liger_rope=True,
        )
        liger_model = Llama(liger_config).to(self.device)
        liger_model.load_state_dict(self.base_model.state_dict())
        liger_model.eval()

        input_ids = torch.randint(0, 1000, (4, 32), device=self.device)

        with torch.no_grad():
            base_out = self.base_model(input_ids)["logits"]
            liger_out = liger_model(input_ids)["logits"]

        torch.testing.assert_close(base_out, liger_out, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
