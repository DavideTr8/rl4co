import pytest
import torch

from rl4co.envs.scheduling.scratch import (
    batched_last_nonzero_indices,
    end_time_lb,
    last_nonzero_indices,
)


@pytest.fixture
def num_jobs():
    return 5


@pytest.fixture
def num_machines():
    return 3


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def starting_times(num_jobs, num_machines):
    st = torch.zeros(num_jobs, num_machines)
    st[0, 0] = 1
    st[1, 0] = 2
    st[1, 1] = 2
    st[2, 1] = 1
    st[2, 1] = 2
    st[2, 2] = 3
    st[3, 0] = 3
    st[3, 1] = 4
    return st


@pytest.fixture
def batched_starting_times(starting_times, batch_size):
    st = starting_times.unsqueeze(0).repeat(batch_size, 1, 1)
    st[1, 3, 1] = 0
    return st


@pytest.fixture
def batched_durations(num_jobs, num_machines, batch_size):
    return torch.ones(batch_size, num_jobs, num_machines)


def test_last_nonzero_indices(starting_times):
    x_idxs, y_idxs = last_nonzero_indices(starting_times)
    assert x_idxs.tolist() == [0, 1, 2, 3]
    assert y_idxs.tolist() == [0, 1, 2, 1]


def test_batched_last_nonzero_indices(batched_starting_times):
    b_idxs, x_idxs, y_idxs = batched_last_nonzero_indices(batched_starting_times)
    assert b_idxs.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert x_idxs.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert y_idxs.tolist() == [0, 1, 2, 1, 0, 1, 2, 0]


def test_end_time_lb(batched_starting_times, batched_durations):
    lb = end_time_lb(batched_starting_times, batched_durations)
    print(lb)
