from typing import Optional, Tuple

import numpy as np
import torch

from tensordict.tensordict import TensorDict

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.envs.scheduling.jssp import configs, permissibleLeftShift


def last_nonzero_indices(
    starting_times: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the last non-zero indices of the given 2D tensor along the columns (dim=2).

    Args:
        starting_times (torch.Tensor): 2D array with jobs starting times to find the last non-zero indices of.
            shape: (num_jobs, num_machines)
    Returns:
        Tuple of tensors containing the last non-zero indices of the given array along the given axis.
    """
    invalid_val = -1
    dim = 1
    mask = (starting_times != 0).to(dtype=torch.int32)
    val = starting_times.shape[dim] - torch.flip(mask, dims=[dim]).argmax(dim=dim) - 1
    yAxis = torch.where(mask.any(dim=dim), val, invalid_val)
    xAxis = torch.arange(starting_times.shape[0], dtype=torch.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet


def batched_last_nonzero_indices(starting_times: torch.Tensor) -> torch.Tensor:
    stack_b = []
    stack_x = []
    stack_y = []
    for i in range(starting_times.shape[0]):
        x_idx, y_idx = last_nonzero_indices(starting_times[i])
        stack_x.append(x_idx)
        stack_y.append(y_idx)
        stack_b.append(torch.tensor([i] * x_idx.shape[0]))
    return torch.stack(stack_b), torch.stack(stack_x), torch.stack(stack_y)


def end_time_lb(starting_times: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    Calculate the lower bound of the end time of each job.

    Args:
        starting_times (torch.Tensor): batched 2D array containing the start time of each job.
            shape: (batch_size, num_jobs, num_machines)
        durations (torch.Tensor): batched 2D array containing the duration of each job.
            shape: (batch_size, num_jobs, num_machines)
    Returns:
        Tensor containing the lower bound of the end time of each job.
    """
    b, x, y = batched_last_nonzero_indices(starting_times)
    durations[starting_times != 0] = 0
    durations[b, x, y] = starting_times[b, x, y]
    ret = starting_times + torch.where(
        starting_times != 0, 0, torch.cumsum(durations, axis=2)
    )
    return ret


def get_action_nbghs(
    actions: torch.Tensor, op_ids_on_mchs: np.ndarray
) -> Tuple[int, int]:
    """Get the predecessor and successor of the given action in the given schedule.

    Args:
        actions (torch.Tensor): The action to get the predecessor and successor of.
        op_ids_on_mchs (np.ndarray): 2D array containing the operation IDs on each machine.
    Returns:
        Tuple of ints containing the predecessor and successor of the given action.
    """
    torch.arange(actions.shape[0], device=actions.device)
    row, col = torch.nonzero(op_ids_on_mchs == actions)
    col_left = max(0, col.item() - 1)
    col_right = min(op_ids_on_mchs.shape[-1] - 1, col.item() + 1)

    precd = op_ids_on_mchs[row, col_left].item()
    succd = (
        actions
        if op_ids_on_mchs[row, col_right].item() < 0
        else op_ids_on_mchs[row, col_right].item()
    )

    return precd, succd


class JSSP(RL4COEnvBase):

    """Job Shop Scheduling Problem (JSSP) environment.
    As per the definition given in https://arxiv.org/pdf/2010.12367.pdf.
    The goal is to schedule a set of jobs on a set of machines such that the makespan is minimized.
    In this variation, the number of operations per job is equal to the number of machines.

    Args:

    Note:
        -
    """

    name = "jssp"

    def __init__(self, num_jobs, num_machines, **kwargs):
        super().__init__(**kwargs)

        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_tasks = self.num_jobs * self.num_machines
        # the task id for first column
        self.first_col = torch.arange(
            start=0,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )
        # the task id for last column
        self.last_col = torch.arange(
            start=num_machines - 1,
            end=self.num_tasks,
            step=num_machines,
            dtype=torch.int64,
            device=self.device,
        )

        # initialize zero matrices for memory allocation
        self.machines = torch.zeros(
            (*self.batch_size, self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )
        self.durations = torch.zeros(
            (*self.batch_size, self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )
        self.durations_cp = torch.zeros(
            (*self.batch_size, self.num_jobs, self.num_machines),
            dtype=torch.float32,
            device=self.device,
        )

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        """Reset the environment."""
        batch_size = self.batch_size if batch_size is None else batch_size
        assert len(batch_size) == 1, "Only support one dimensional batch size."
        # if td is None:
        #     td = generate_data(batch_size=batch_size)

        self.machines = td["machines"]
        assert self.machines.shape == (
            *batch_size,
            self.num_jobs,
            self.num_machines,
        ), "Provided data shape error"
        self.durations = td["durations"]
        self.durations_cp = torch.copy(self.durations)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = torch.diag_embed(torch.ones(self.num_tasks - 1), offset=1)
        conj_nei_low_stream = torch.diag_embed(torch.ones(self.num_tasks + 1), offset=-1)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = torch.eye(self.num_tasks, dtype=torch.float32, device=self.device)
        adjacency = self_as_nei + conj_nei_up_stream  # TODO check conj_nei_low_stream
        self.adjacency = adjacency.usqueeze(0).repeat(*batch_size, 1, 1)

        # initialize features
        self.LBs = torch.cumsum(self.durations, dim=2)
        self.initQuality = (
            torch.amax(self.LBs, dim=(1, 2))
            if not configs.init_quality_flag
            else torch.zeros(*batch_size)
        )
        self.max_endTime = torch.copy(self.initQuality)
        self.finished_mark = torch.zeros_like(self.machines)

        features = torch.concatenate(
            [
                self.LBs.reshape(*batch_size, self.num_tasks, 1)
                / configs.et_normalize_coef,
                self.finished_mark.reshape(batch_size, self.num_tasks, 1),
            ],
            dim=2,
        )

        # initialize feasible omega
        self.feasible_actions = self.first_col.to(dtype=torch.int64).repeat(
            *batch_size, 1
        )

        # initialize mask
        self.mask = torch.zeros(
            size=(
                batch_size,
                self.num_jobs,
            ),
            dtype=torch.bool,
        )

        # start time of operations on machines
        self.mchsStartTimes = (
            torch.ones_like(self.durations.transpose(1, 2), dtype=torch.int32)
            * -configs.high
        )
        # Ops ID on machines
        self.operations_on_machines = -self.num_jobs * np.ones_like(
            self.durations.transpose(1, 2), dtype=torch.int32
        )

        self.starting_times = torch.zeros_like(self.durations, dtype=torch.float32)

        tensordict = TensorDict(
            {
                "adjacency": self.adjacency,
                "features": features,
                "feasible_actions": self.feasible_actions,
                "mask": self.mask,
            },
            batch_size=batch_size,
        )
        return tensordict

    def done(self):
        if len(self.partial_sol_sequeence) == self.num_tasks:
            return True
        return False

    def step(self, action):
        # action is an int [0, self.num_tasks]
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:
            # UPDATE BASIC INFO:
            row = action // self.num_machines
            col = action % self.num_machines
            batch_idx = torch.arange(
                self.batch_size, dtype=torch.int64, device=self.device
            )
            self.finished_mark[batch_idx, row, col] = 1
            dur_a = self.durations[batch_idx, row, col]
            self.partial_sol_sequeence = torch.concatenate(
                [self.partial_sol_sequeence, action]
            )

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = permissibleLeftShift(
                a=action,
                durMat=self.durations,
                mchMat=self.m,
                mchsStartTimes=self.mchsStartTimes,
                operations_on_machines=self.operations_on_machines,
            )
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.feasible_actions[batch_idx, row] += 1
            else:
                self.mask[batch_idx, row] = 1

            self.starting_times[batch_idx, row, col] = startTime_a + dur_a

            self.LBs = end_time_lb(self.starting_times, self.durations_cp)

            # adj matrix
            precd, succd = get_action_nbghs(action, self.operations_on_machines)
            self.adjacency[action] = 0
            self.adjacency[action, action] = 1
            if action not in self.first_col:
                self.adjacency[action, action - 1] = 1
            self.adjacency[action, precd] = 1
            self.adjacency[succd, action] = 1
            if (
                flag and precd != action and succd != action
            ):  # Remove the old arc when a new operation inserts between two operations
                self.adjacency[succd, precd] = 0

        # prepare for return
        features = np.concatenate(
            (
                self.LBs.reshape(-1, 1),  # /configs.et_normalize_coef,
                self.finished_mark.reshape(-1, 1),
            ),
            axis=1,
        )
        reward = -(self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = 1  # configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return (
            self.adjacency,
            features,
            reward,
            self.done(),
            self.feasible_actions,
            self.mask,
        )


if __name__ == "__main__":
    dur = torch.tensor([[1, 2], [3, 4]])
    temp1 = torch.zeros_like(dur)
    temp2 = torch.zeros_like(dur)
    temp3 = torch.ones_like(dur)

    temp1[0, 0] = 1
    temp1[1, 0] = 3
    temp1[1, 1] = 5

    temp2[0, 1] = 1
    temp2[1, 0] = 3
    print(temp1)

    temp = torch.stack([temp1, temp2, temp3])
    print(last_nonzero_indices(temp2))
    print(batched_last_nonzero_indices(temp))

    end_time_lb(temp, torch.stack([dur, dur, dur]))
