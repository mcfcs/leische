"""Channel plumbing regressions that don't need the pretrained encoder."""

import torch

from leische.model import TargetAttention, scatter_items


def test_scatter_items_uneven_counts():
    flat = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    batch_idx = torch.tensor([0, 0, 0, 1, 2, 2])
    out, mask = scatter_items(flat, batch_idx, batch_size=3)
    assert out.shape == (3, 3, 2) and mask.shape == (3, 3)
    assert mask.sum(dim=1).tolist() == [3, 1, 2]


def test_scatter_items_empty_row_masked_out():
    flat = torch.ones(2, 4)
    batch_idx = torch.tensor([1, 1])  # row 0 gets nothing
    out, mask = scatter_items(flat, batch_idx, batch_size=2)
    assert not mask[0].any() and mask[1].sum() == 2


def test_attention_uneven_bank_widths_combine():
    """Regression for the ablation-matrix crash: sarcastic and non-sarcastic
    retrieval banks pad to different widths (fewer than k sarcastic exemplars);
    per-row emptiness must be combined via .any(), never mask | mask."""
    d = 8
    attn_a, attn_b = TargetAttention(d), TargetAttention(d)
    t = torch.randn(2, d)
    a_items, a_mask = scatter_items(torch.randn(3, d), torch.tensor([0, 0, 1]), 2)  # width 2
    b_items, b_mask = scatter_items(torch.randn(5, d), torch.tensor([0, 0, 0, 0, 1]), 2)  # width 4
    assert a_mask.shape[1] != b_mask.shape[1]
    out = torch.cat([attn_a(t, a_items, a_mask), attn_b(t, b_items, b_mask)], dim=-1)
    empty = ~(a_mask.any(dim=-1) | b_mask.any(dim=-1))  # the fixed combination
    assert out.shape == (2, 2 * d) and not empty.any()


def test_attention_all_empty_rows_yield_zeros():
    d = 4
    attn = TargetAttention(d)
    t = torch.randn(3, d)
    items = torch.zeros(3, 1, d)
    mask = torch.zeros(3, 1, dtype=torch.bool)
    out = attn(t, items, mask)
    assert torch.allclose(out, torch.zeros_like(out))
