import math
from unittest.mock import (
    Mock,
    patch,
)

import torch

from optimus_dl.modules.criterion.cross_entropy import (
    CrossEntropyCriterion,
    CrossEntropyCriterionConfig,
)
from optimus_dl.modules.distributed.fake import FakeCollective
from optimus_dl.modules.metrics.source import StandardProtocols


class TestCrossEntropyCriterionConfig:
    """Tests for CrossEntropyCriterionConfig"""

    def test_default_config(self):
        config = CrossEntropyCriterionConfig()
        assert config.label_smoothing == 0.0

    def test_custom_config(self):
        config = CrossEntropyCriterionConfig(label_smoothing=0.1)
        assert config.label_smoothing == 0.1

    def test_config_inheritance(self):
        """Test that config inherits from RegistryConfig"""
        from optimus_dl.core.registry import RegistryConfigStrict

        config = CrossEntropyCriterionConfig()
        assert isinstance(config, RegistryConfigStrict)


class TestCrossEntropyCriterion:
    """Tests for CrossEntropyCriterion"""

    def test_init(self):
        config = CrossEntropyCriterionConfig(label_smoothing=0.1)
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        assert criterion.cfg == config
        assert criterion.cfg.label_smoothing == 0.1

    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_summed")
    def test_call_basic(self, mock_log_summed, mock_log_averaged):
        """Test basic criterion call functionality"""
        config = CrossEntropyCriterionConfig(label_smoothing=0.0)
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create mock model and batch
        mock_model = Mock()
        batch_size, seq_len, vocab_size = 2, 10, 100

        # Mock model output
        logits = torch.randn(
            batch_size, seq_len - 1, vocab_size
        )  # seq_len - 1 due to shifting
        mock_model.return_value = {"logits": logits}

        # Create batch with input_ids
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch = {"input_ids": input_ids.clone()}

        # Call criterion requesting logits
        loss, exposed = criterion(
            mock_model, batch, requested_protocols={StandardProtocols.LOGITS}
        )

        # Check that model was called with shifted input_ids
        mock_model.assert_called_once()
        call_args = mock_model.call_args
        assert torch.equal(call_args[1]["input_ids"], input_ids[:, :-1])

        # Check that loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        # Check exposed data
        assert StandardProtocols.LOGITS in exposed
        assert torch.equal(exposed[StandardProtocols.LOGITS], logits)

        # Check that metrics were logged
        assert mock_log_averaged.call_count >= 2  # accuracy, loss, perplexity
        assert mock_log_summed.call_count >= 2  # batch_tokens, total_tokens

    def test_input_target_shifting(self):
        """Test that input_ids are properly shifted for next-token prediction"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create mock model
        mock_model = Mock()
        batch_size, seq_len, vocab_size = 1, 5, 10
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)
        mock_model.return_value = {"logits": logits}

        # Input sequence: [1, 2, 3, 4, 5]
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        batch = {"input_ids": input_ids.clone()}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            criterion(mock_model, batch)

            # Model should receive [1, 2, 3, 4] (all but last token)
            call_args = mock_model.call_args
            expected_input = torch.tensor([[1, 2, 3, 4]])
            assert torch.equal(call_args[1]["input_ids"], expected_input)

    def test_cross_entropy_computation(self):
        """Test cross-entropy loss computation with deterministic predictions."""
        config = CrossEntropyCriterionConfig(label_smoothing=0.0)
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create deterministic test case
        batch_size, seq_len, vocab_size = 1, 3, 5

        # Create logits that predict specific tokens
        logits = torch.zeros(batch_size, seq_len - 1, vocab_size)
        logits[0, 0, 1] = 10.0  # Strongly predict token 1 at position 0
        logits[0, 1, 2] = 10.0  # Strongly predict token 2 at position 1

        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        # Targets should be [1, 2] (tokens 1 and 2)
        input_ids = torch.tensor([[0, 1, 2]])  # Input: [0, 1, 2], targets: [1, 2]
        batch = {"input_ids": input_ids.clone()}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            loss, exposed = criterion(mock_model, batch)

            # Loss should be low since predictions match targets
            assert loss.item() < 1.0

    def test_label_smoothing(self):
        """Test that label smoothing affects loss computation (regularization technique)."""
        config_no_smooth = CrossEntropyCriterionConfig(label_smoothing=0.0)
        config_smooth = CrossEntropyCriterionConfig(label_smoothing=0.1)

        criterion_no_smooth = CrossEntropyCriterion(
            config_no_smooth, collective=FakeCollective(0, 1)
        )
        criterion_smooth = CrossEntropyCriterion(
            config_smooth, collective=FakeCollective(0, 1)
        )

        # Same setup for both
        batch_size, seq_len, vocab_size = 1, 3, 10
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)

        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch_no_smooth = {"input_ids": input_ids.clone()}
        batch_smooth = {"input_ids": input_ids.clone()}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            loss_no_smooth, _ = criterion_no_smooth(mock_model, batch_no_smooth)
            loss_smooth, _ = criterion_smooth(mock_model, batch_smooth)

            # Losses should be different due to label smoothing
            assert not torch.allclose(loss_no_smooth, loss_smooth)

    def test_accuracy_metric(self):
        """Test accuracy computation"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Perfect predictions
        batch_size, seq_len, vocab_size = 2, 3, 5
        logits = torch.zeros(batch_size, seq_len, vocab_size)
        targets = torch.tensor([[1, 2, 3], [0, 4, 1]])

        # Set logits to strongly predict the correct targets
        for b in range(batch_size):
            for t in range(seq_len):
                logits[b, t, targets[b, t]] = 10.0

        predictions = criterion.gather_predictions(logits)
        accuracy = criterion.accuracy_metric(predictions, targets)
        assert accuracy == 1.0  # Perfect accuracy

        # Random predictions
        random_logits = torch.randn(batch_size, seq_len, vocab_size)
        random_predictions = criterion.gather_predictions(random_logits)
        accuracy_random = criterion.accuracy_metric(random_predictions, targets)
        assert 0.0 <= accuracy_random <= 1.0

    def test_accuracy_metric_partial_correct(self):
        """Test accuracy with partially correct predictions"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # 2 out of 4 predictions correct
        logits = torch.zeros(1, 4, 3)
        targets = torch.tensor([[0, 1, 2, 1]])

        # Make first 2 predictions correct, last 2 incorrect
        logits[0, 0, 0] = 10.0  # Correct: predict 0, target is 0
        logits[0, 1, 1] = 10.0  # Correct: predict 1, target is 1
        logits[0, 2, 0] = 10.0  # Incorrect: predict 0, target is 2
        logits[0, 3, 0] = 10.0  # Incorrect: predict 0, target is 1

        predictions = criterion.gather_predictions(logits)
        accuracy = criterion.accuracy_metric(predictions, targets)
        assert accuracy == 0.5  # 2/4 = 0.5

    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged_exponent")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_summed")
    def test_metric_logging(
        self, mock_log_summed, mock_log_averaged, mock_log_averaged_exponent
    ):
        """Test that all metrics are logged correctly"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Setup
        batch_size, seq_len, vocab_size = 2, 5, 100
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)
        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch = {"input_ids": input_ids.clone()}

        criterion(mock_model, batch)

        # Check log_averaged calls
        averaged_calls = [call[0][0] for call in mock_log_averaged.call_args_list]
        assert "accuracy" in averaged_calls
        assert "loss" in averaged_calls

        # Check log_averaged_exponent calls
        exponent_calls = [
            call[0][0] for call in mock_log_averaged_exponent.call_args_list
        ]
        assert "perplexity" in exponent_calls

        # Check log_summed calls
        summed_calls = [call[0][0] for call in mock_log_summed.call_args_list]
        assert "batch_tokens" in summed_calls
        assert "total_tokens" in summed_calls

    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_summed")
    def test_token_counting(self, mock_log_summed, mock_log_averaged):
        """Test that token counts are computed correctly"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        batch_size, seq_len, vocab_size = 3, 10, 50
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)
        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch = {"input_ids": input_ids.clone()}

        criterion(mock_model, batch)

        # Find token count calls
        batch_tokens_call = None
        total_tokens_call = None

        for call in mock_log_summed.call_args_list:
            if call[0][0] == "batch_tokens":
                batch_tokens_call = call
            elif call[0][0] == "total_tokens":
                total_tokens_call = call

        assert batch_tokens_call is not None
        assert total_tokens_call is not None

        # Token count should be batch_size * (seq_len - 1)
        expected_tokens = batch_size * (seq_len - 1)

        # Extract the lambda function and call it to get the token count
        batch_tokens_value = batch_tokens_call[0][
            1
        ]()  # Second positional argument is the lambda
        total_tokens_value = total_tokens_call[0][1]()

        assert batch_tokens_value == expected_tokens
        assert total_tokens_value == expected_tokens

    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged_exponent")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_averaged")
    @patch("optimus_dl.modules.criterion.cross_entropy.log_summed")
    def test_perplexity_computation(
        self, mock_log_summed, mock_log_averaged, mock_log_averaged_exponent
    ):
        """Test that perplexity is computed correctly"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create a case with known loss
        batch_size, seq_len, vocab_size = 1, 3, 10
        logits = torch.zeros(batch_size, seq_len - 1, vocab_size)

        # Set up for specific loss value
        torch.tensor([[5, 7]])  # Target tokens

        # Logits that will produce a specific cross-entropy loss
        logits[0, 0, 5] = 1.0  # Some prediction strength for token 5
        logits[0, 1, 7] = 1.0  # Some prediction strength for token 7

        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.tensor([[0, 5, 7]])  # Input IDs where targets are [5, 7]
        batch = {"input_ids": input_ids.clone()}

        loss, _ = criterion(mock_model, batch)

        # Find perplexity logging call
        perplexity_call = None
        for call in mock_log_averaged_exponent.call_args_list:
            if call[0][0] == "perplexity":
                perplexity_call = call
                break

        assert perplexity_call is not None

        # Perplexity should be exp(loss)
        logged_perplexity = perplexity_call[1]["value"]()
        logged_perplexity = math.exp(logged_perplexity)
        expected_perplexity = torch.exp(loss).item()

        assert abs(logged_perplexity - expected_perplexity) < 1e-6

    def test_batch_modification(self):
        """Test that the original batch is modified correctly"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        batch_size, seq_len, vocab_size = 1, 5, 10
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)
        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        input_ids.clone()
        batch = {"input_ids": input_ids, "other_key": "value"}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            criterion(mock_model, batch)

            # Batch should be modified: input_ids should be shifted
            # batch.pop("input_ids") removes input_ids and then it's re-added as shifted version
            # So the batch should contain the shifted input_ids
            assert "other_key" in batch
            assert batch["other_key"] == "value"

    def test_different_batch_sizes_and_sequence_lengths(self):
        """Test criterion with various batch sizes and sequence lengths"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        test_cases = [
            (1, 5, 100),  # Single sample, short sequence
            (4, 10, 50),  # Small batch, medium sequence
            (8, 20, 1000),  # Larger batch, longer sequence
        ]

        for batch_size, seq_len, vocab_size in test_cases:
            logits = torch.randn(batch_size, seq_len - 1, vocab_size)
            mock_model = Mock()
            mock_model.return_value = {"logits": logits}

            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            batch = {"input_ids": input_ids.clone()}

            with (
                patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
                patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
            ):

                loss, exposed = criterion(
                    mock_model, batch, requested_protocols={StandardProtocols.LOGITS}
                )

                assert isinstance(loss, torch.Tensor)
                assert loss.dim() == 0  # Scalar loss
                assert torch.isfinite(loss)
                assert StandardProtocols.LOGITS in exposed

    def test_gradient_flow(self):
        """Test that gradients flow through the loss"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create model with parameters
        class SimpleModel(torch.nn.Module):
            def __init__(self, vocab_size, embed_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, embed_size)
                self.lm_head = torch.nn.Linear(embed_size, vocab_size)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                logits = self.lm_head(x)
                return {"logits": logits}

        vocab_size, embed_size = 50, 32
        model = SimpleModel(vocab_size, embed_size)

        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        batch = {"input_ids": input_ids}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            loss, _ = criterion(model, batch)
            loss.backward()

            # Check that gradients exist
            assert model.embedding.weight.grad is not None
            assert model.lm_head.weight.grad is not None
            assert model.lm_head.bias.grad is not None

    def test_no_grad_accuracy_metric(self):
        """Test that accuracy_metric is computed with no_grad"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        # Create tensors that require grad
        logits = torch.randn(2, 5, 10, requires_grad=True)
        targets = torch.randint(0, 10, (2, 5))

        # This should not raise an error despite logits requiring grad
        predictions = criterion.gather_predictions(logits)
        accuracy = criterion.accuracy_metric(predictions, targets)

        assert isinstance(accuracy, float)
        assert 0.0 <= accuracy <= 1.0

    def test_edge_case_single_token(self):
        """Test with minimal sequence length (2 tokens -> 1 target)"""
        config = CrossEntropyCriterionConfig()
        criterion = CrossEntropyCriterion(config, collective=FakeCollective(0, 1))

        batch_size, seq_len, vocab_size = 1, 2, 10
        logits = torch.randn(batch_size, seq_len - 1, vocab_size)  # Shape: (1, 1, 10)
        mock_model = Mock()
        mock_model.return_value = {"logits": logits}

        input_ids = torch.tensor([[3, 7]])  # Input: [3], Target: [7]
        batch = {"input_ids": input_ids.clone()}

        with (
            patch("optimus_dl.modules.criterion.cross_entropy.log_averaged"),
            patch("optimus_dl.modules.criterion.cross_entropy.log_summed"),
        ):

            loss, exposed = criterion(
                mock_model, batch, requested_protocols={StandardProtocols.LOGITS}
            )

            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0
            assert torch.isfinite(loss)
            assert StandardProtocols.LOGITS in exposed
