"""Integration tests for data pipeline components."""

from unittest.mock import MagicMock, Mock, patch

from optimus_dl.modules.data.transforms.tokenize import (
    TokenizeTransform,
    TokenizeTransformConfig,
)


class TestDataPipelineIntegration:
    """Test realistic data pipeline usage patterns."""

    def test_tokenization_pipeline(self):
        """Test tokenization in realistic scenario."""
        # Sample text data similar to what would come from a dataset
        sample_texts = [
            {"text": "This is a short sentence."},
            {"text": "This is a much longer sentence that contains more tokens."},
            {"text": "Short text."},
            {"text": "Another medium length sentence for testing purposes."},
        ]

        # Setup tokenization with mocked tokenizer
        with patch(
            "optimus_dl.modules.data.transforms.tokenize.build_tokenizer"
        ) as mock_build_tokenizer:
            # Mock tokenizer behavior
            mock_tokenizer = Mock()

            def mock_encode(text):
                # Simple tokenization: split words and add IDs
                words = text.split()
                return list(range(len(words)))

            def mock_decode(ids):
                return " ".join(["word"] * len(ids))

            mock_tokenizer.encode.side_effect = mock_encode
            mock_tokenizer.decode.side_effect = mock_decode
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.bos_token_id = 1
            mock_build_tokenizer.return_value = mock_tokenizer

            # Create tokenization transform
            tokenizer_config = MagicMock()
            tokenize_config = TokenizeTransformConfig(tokenizer_config=tokenizer_config)
            tokenizer_transform = TokenizeTransform(tokenize_config)

            # Test the mapping function directly
            for sample in sample_texts:
                result = tokenizer_transform._map(sample)

                # Verify tokenization worked
                assert "input_ids" in result
                assert isinstance(result["input_ids"], list)
                assert len(result["input_ids"]) > 0

    def test_transform_error_handling(self):
        """Test that transforms handle errors gracefully."""
        # Test with problematic data
        problematic_samples = [
            {"text": ""},  # Empty text
            {"not_text": "missing text field"},  # Wrong field name
            {"text": "Normal text"},  # Valid sample
        ]

        with patch(
            "optimus_dl.modules.data.transforms.tokenize.build_tokenizer"
        ) as mock_build_tokenizer:
            mock_tokenizer = Mock()

            def mock_encode(text):
                if not text:
                    return []
                return [1, 2, 3]

            mock_tokenizer.encode.side_effect = mock_encode
            mock_build_tokenizer.return_value = mock_tokenizer

            tokenizer_config = MagicMock()
            tokenize_config = TokenizeTransformConfig(tokenizer_config=tokenizer_config)
            tokenizer_transform = TokenizeTransform(tokenize_config)

            # Process with error handling
            successful_results = []
            errors = []

            for sample in problematic_samples:
                try:
                    result = tokenizer_transform._map(sample)
                    successful_results.append(result)
                except (ValueError, KeyError) as e:
                    errors.append((sample, str(e)))

            # Should have some successful results and some errors
            assert len(successful_results) >= 1
            assert len(errors) >= 1

    def test_batch_tokenization_pattern(self):
        """Test processing batches of samples through tokenization."""
        batch_size = 4

        # Generate sample data
        sample_data = []
        for i in range(12):  # 3 batches worth
            sample_data.append(
                {
                    "text": f"Sample text number {i} with varying lengths "
                    + "word " * (i % 3),
                }
            )

        # Mock tokenizer
        with patch(
            "optimus_dl.modules.data.transforms.tokenize.build_tokenizer"
        ) as mock_build_tokenizer:
            mock_tokenizer = Mock()

            def mock_encode(text):
                word_count = len(text.split())
                return list(range(word_count))

            mock_tokenizer.encode.side_effect = mock_encode
            mock_tokenizer.decode.return_value = "decoded"
            mock_build_tokenizer.return_value = mock_tokenizer

            # Setup transform
            tokenizer_config = MagicMock()
            tokenize_config = TokenizeTransformConfig(tokenizer_config=tokenizer_config)
            tokenizer_transform = TokenizeTransform(tokenize_config)

            # Process data in batches
            processed_batches = []
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data[i : i + batch_size]

                # Process batch
                tokenized_batch = []
                for sample in batch:
                    tokenized = tokenizer_transform._map(sample)
                    tokenized_batch.append(tokenized)

                processed_batches.append(tokenized_batch)

            # Verify batch processing
            assert len(processed_batches) == 3  # 12 samples / 4 per batch

            # Check that all samples have consistent structure
            all_samples = [sample for batch in processed_batches for sample in batch]
            assert all("input_ids" in sample for sample in all_samples)
            assert all(isinstance(sample["input_ids"], list) for sample in all_samples)
