import torch

from roughness.scheme1.models import (
    CNNEncoder,
    CNNTCNEncoder,
    SegmentSignalRegressor,
    count_trainable_parameters,
    masked_mean_pool,
)


def test_cnn_encoder_emits_32_dimensional_window_embedding_under_parameter_cap():
    encoder = CNNEncoder()
    signal = torch.randn(4, 3, 2048)

    embedding = encoder(signal)

    assert embedding.shape == (4, 32)
    assert count_trainable_parameters(encoder) < 1_000_000
    assert torch.isfinite(embedding).all()


def test_cnn_tcn_encoder_emits_embedding_under_total_parameter_cap():
    encoder = CNNTCNEncoder()
    signal = torch.randn(3, 3, 2048)

    embedding = encoder(signal)

    assert embedding.shape == (3, 32)
    assert count_trainable_parameters(encoder) < 2_000_000
    dilations = [block.dilation for block in encoder.tcn_blocks]
    assert dilations == [1, 2, 4, 8]


def test_masked_mean_pool_ignores_padded_windows():
    embeddings = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    pooled = masked_mean_pool(embeddings, mask)

    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([7.0, 8.0]))


def test_segment_regressor_prediction_is_invariant_to_extra_padding():
    model = SegmentSignalRegressor(CNNEncoder()).eval()
    real = torch.randn(1, 2, 3, 2048)
    padded = torch.zeros(1, 5, 3, 2048)
    padded[:, :2] = real

    with torch.no_grad():
        first = model(real, torch.tensor([[True, True]])).prediction
        second = model(
            padded,
            torch.tensor([[True, True, False, False, False]]),
        ).prediction

    torch.testing.assert_close(first, second)
    assert first.shape == (1,)


def test_segment_regressor_backpropagates_finite_gradients():
    model = SegmentSignalRegressor(CNNTCNEncoder())
    signal = torch.randn(2, 2, 3, 2048)
    mask = torch.ones(2, 2, dtype=torch.bool)

    output = model(signal, mask)
    output.prediction.sum().backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
