import torch
import pytest

from optimus_dl.modules.model.blocks.attention import RotarySelfAttention
from optimus_dl.modules.model.blocks.rope import precompute_freqs_cis


def get_vanilla_attention_mask(T, causal=True, document_ids=None):
    """Utility to create a manual attention mask for verification."""
    mask = torch.ones(T, T, dtype=torch.bool)
    if causal:
        mask = torch.tril(mask)
    if document_ids is not None:
        # document_ids: (T,)
        doc_mask = document_ids.unsqueeze(0) == document_ids.unsqueeze(1)
        mask = mask & doc_mask
    return mask


class TestAttentionModesEquivalence:
    """Tests to ensure all attention modes (padded, flat, document-masked, varlen) are numerically equivalent."""

    @pytest.fixture
    def setup_data(self):
        torch.manual_seed(42)
        n_embd = 64
        n_head = 4
        head_dim = n_embd // n_head

        # 3 documents of different lengths
        doc_lengths = [5, 8, 3]
        total_tokens = sum(doc_lengths)
        max_len = max(doc_lengths)
        B = len(doc_lengths)

        # Generate random tokens for each document
        docs = [torch.randn(length, n_embd) for length in doc_lengths]

        # 1. Padded Layout (B, max_len, n_embd)
        x_padded = torch.zeros(B, max_len, n_embd)
        for i, doc in enumerate(docs):
            x_padded[i, : doc_lengths[i], :] = doc
        seq_lens = torch.tensor(doc_lengths)

        # 2. Flat Layout (1, total_tokens, n_embd)
        x_flat = torch.cat(docs, dim=0).unsqueeze(0)

        # Metadata for Flat Layout
        document_ids_flat = []
        position_ids_flat = []
        for i, length in enumerate(doc_lengths):
            document_ids_flat.extend([i] * length)
            position_ids_flat.extend(range(length))

        document_ids_flat = torch.tensor(document_ids_flat).unsqueeze(0)
        position_ids_flat = torch.tensor(position_ids_flat).unsqueeze(0)

        # Metadata for VarLen Layout
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
        cu_seqlens[1:] = torch.tensor(doc_lengths).cumsum(0)
        max_seqlen = max(doc_lengths)

        # RoPE frequencies
        freqs_cis = precompute_freqs_cis(head_dim, max(max_len, total_tokens))

        return {
            "n_embd": n_embd,
            "n_head": n_head,
            "docs": docs,
            "doc_lengths": doc_lengths,
            "x_padded": x_padded,
            "seq_lens": seq_lens,
            "x_flat": x_flat,
            "document_ids_flat": document_ids_flat,
            "position_ids_flat": position_ids_flat,
            "cu_seqlens": cu_seqlens,
            "max_seqlen": max_seqlen,
            "freqs_cis": freqs_cis,
        }

    def test_padded_vs_flat_document_masked(self, setup_data, device):
        """Verify that padded layout with seq_lens matches flat layout with document_ids."""
        d = setup_data
        attn = RotarySelfAttention(n_embd=d["n_embd"], n_head=d["n_head"]).to(device)
        attn.eval()

        x_padded = d["x_padded"].to(device)
        freqs_cis = d["freqs_cis"].to(device)
        seq_lens = d["seq_lens"].to(device)
        x_flat = d["x_flat"].to(device)
        doc_ids_flat = d["document_ids_flat"].to(device)
        pos_ids_flat = d["position_ids_flat"].to(device)

        with torch.no_grad():
            # 1. Forward with Padded Layout
            # We need to slice freqs_cis for RoPE inside attention
            out_padded = attn(x_padded, freqs_cis, seq_lens=seq_lens)

            # 2. Forward with Flat Layout + Document Masking
            out_flat = attn(
                x_flat, freqs_cis, document_ids=doc_ids_flat, position_ids=pos_ids_flat
            )

        # Compare document by document
        curr = 0
        for i, length in enumerate(d["doc_lengths"]):
            padded_doc = out_padded[i, :length, :]
            flat_doc = out_flat[0, curr : curr + length, :]

            torch.testing.assert_close(padded_doc, flat_doc, rtol=1e-5, atol=1e-5)
            curr += length

    def test_flat_vs_varlen(self, setup_data, device):
        """Verify that flat layout with document_ids matches varlen layout with cu_seqlens.
        This tests the varlen logic (either the CUDA kernel or the CPU fallback)."""
        d = setup_data

        attn = RotarySelfAttention(n_embd=d["n_embd"], n_head=d["n_head"]).to(device)
        attn.eval()

        x_flat = d["x_flat"].to(device)
        freqs_cis = d["freqs_cis"].to(device)
        doc_ids = d["document_ids_flat"].to(device)
        pos_ids = d["position_ids_flat"].to(device)
        cu_seqlens = d["cu_seqlens"].to(device)

        with torch.no_grad():
            # 1. Flat Layout with Document IDs (Flex/SDPA path)
            out_flat = attn(
                x_flat, freqs_cis, document_ids=doc_ids, position_ids=pos_ids
            )

            # 2. VarLen Layout (Varlen logic path - uses kernel on CUDA or fallback on CPU)
            out_varlen = attn(
                x_flat,
                freqs_cis,
                position_ids=pos_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=d["max_seqlen"],
            )

        torch.testing.assert_close(out_flat, out_varlen, rtol=1e-5, atol=1e-5)

    def test_invalid_inputs(self, setup_data, device):
        """Test that asserts catch invalid combinations of inputs."""
        d = setup_data
        attn = RotarySelfAttention(n_embd=d["n_embd"], n_head=d["n_head"]).to(device)

        # cu_seqlens requires B=1
        bad_x = torch.randn(2, 10, d["n_embd"]).to(device)
        freqs_cis = d["freqs_cis"].to(device)
        cu_seqlens = d["cu_seqlens"].to(device)

        with pytest.raises(AssertionError, match="flat batches"):
            attn(bad_x, freqs_cis, cu_seqlens=cu_seqlens)

        # cu_seqlens must start with 0
        bad_cu = torch.tensor([1, 5, 10]).to(device)
        with pytest.raises(AssertionError, match="start with 0"):
            attn(d["x_flat"].to(device), freqs_cis, cu_seqlens=bad_cu)

        # cu_seqlens[-1] must match T
        bad_cu_end = torch.tensor([0, 5, 9]).to(device)  # T=sum([5, 8, 3]) = 16
        with pytest.raises(AssertionError, match="match sequence length T"):
            attn(d["x_flat"].to(device), freqs_cis, cu_seqlens=bad_cu_end)

    def test_document_boundary_isolation(self, setup_data, device):
        """Verify that document_ids correctly isolate documents within a single sequence."""
        d = setup_data
        attn = RotarySelfAttention(n_embd=d["n_embd"], n_head=d["n_head"]).to(device)
        attn.eval()

        # We'll use a sequence containing 2 documents
        doc1_len = d["doc_lengths"][0]
        doc2_len = d["doc_lengths"][1]
        T = doc1_len + doc2_len

        x = d["x_flat"][:, :T, :].to(device)
        freqs_cis = d["freqs_cis"][:T].to(device)

        # Case A: No document masking (all tokens in one document)
        doc_ids_none = None

        # Case B: Document masking (isolate doc1 and doc2)
        doc_ids_masked = (
            torch.tensor([0] * doc1_len + [1] * doc2_len).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            out_unmasked = attn(x, freqs_cis, document_ids=doc_ids_none)
            out_masked = attn(x, freqs_cis, document_ids=doc_ids_masked)

        # For the first document, both should be identical because there are no PREVIOUS documents to attend to
        torch.testing.assert_close(
            out_unmasked[:, :doc1_len, :], out_masked[:, :doc1_len, :]
        )

        # For the second document, they MUST differ because in unmasked case it attends to doc1
        # (Unless the data is zero, but it's random)
        assert not torch.allclose(
            out_unmasked[:, doc1_len:, :], out_masked[:, doc1_len:, :]
        )

        # Further verification: out_masked for doc2 should match what doc2 would produce if it were ALONE
        x_doc2 = d["docs"][1].unsqueeze(0).to(device)
        freqs_cis_doc2 = d["freqs_cis"][:doc2_len].to(device)
        out_doc2_alone = attn(x_doc2, freqs_cis_doc2)

        torch.testing.assert_close(
            out_masked[:, doc1_len:, :], out_doc2_alone, rtol=1e-5, atol=1e-5
        )

    def test_position_ids_equivalence(self, setup_data, device):
        """Verify that RoPE with position_ids produces expected results."""
        d = setup_data
        attn = RotarySelfAttention(n_embd=d["n_embd"], n_head=d["n_head"]).to(device)
        attn.eval()

        # If we use position_ids=[0, 1, 2...] it should match default RoPE
        x_flat = d["x_flat"].to(device)
        freqs_cis = d["freqs_cis"].to(device)
        T = x_flat.size(1)
        pos_ids = torch.arange(T).unsqueeze(0).to(device)

        with torch.no_grad():
            out_default = attn(x_flat, freqs_cis)
            out_pos_ids = attn(x_flat, freqs_cis, position_ids=pos_ids)

        torch.testing.assert_close(out_default, out_pos_ids)
