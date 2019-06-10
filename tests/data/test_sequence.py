
import math

import pytest
import numpy as np
import numpy.random as npr

from custom_envs.utils.utils_common import shuffle
from custom_envs.data.sequence import InMemorySequence


SEQUENCECLASSES = [InMemorySequence]


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_len_even_batches(sequence_class):
    features = npr.rand(256, 8)
    labels = npr.rand(256, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        assert len(sequence) == (256 / size)


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_len_uneven_batches(sequence_class):
    features = npr.rand(255, 8)
    labels = npr.rand(255, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        assert len(sequence) == math.ceil(255 / size)


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_index_shape_batch_even(sequence_class):
    features = npr.rand(256, 8)
    labels = npr.rand(256, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        assert sequence[0].features.shape == (size, 8)
        assert sequence[0].labels.shape == (size, 1)


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_index_shape_batch_uneven(sequence_class):
    features = npr.rand(255, 8)
    labels = npr.rand(255, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        assert sequence[0].features.shape == (size, 8)
        assert sequence[0].labels.shape == (size, 1)


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_iteration_batch_even(sequence_class):
    features = npr.rand(256, 8)
    labels = npr.rand(256, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        summation = 0
        for j in range(len(sequence)):
            summation += len(sequence[j].features)
        assert summation == 256


@pytest.mark.parametrize("sequence_class", SEQUENCECLASSES)
def test_iteration_batch_uneven(sequence_class):
    features = npr.rand(255, 8)
    labels = npr.rand(255, 1)

    for i in range(8):
        size = 2**i
        sequence = sequence_class(features, labels, size)
        summation = 0
        for j in range(len(sequence)):
            summation += len(sequence[j].features)
        assert summation == 255
