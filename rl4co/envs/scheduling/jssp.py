from typing import Tuple

import numpy as np
import torch


class Configs:
    pass


configs = Configs()


def permissible_left_shift(
    actions,
    durations,
    machines,
    machines_start_times,
    operations_on_machines,
):
    batch_idxs = torch.arange(machines.shape[0])
    ops_ready_time, machines_ready_time = job_machines_ready_time(
        actions,
        machines,
        durations,
        machines_start_times,
        operations_on_machines,
    )
    ops_duration = durations.flatten(1)[batch_idxs, actions]
    machines_selected = machines.flatten(1)[batch_idxs, actions] - 1
    start_times_for_selected_machines = machines_start_times[
        batch_idxs, machines_selected
    ]
    ops_for_selected_machines = operations_on_machines[batch_idxs, machines_selected]
    flag = False
    possible_pos = torch.nonzero(
        ops_ready_time < start_times_for_selected_machines, as_tuple=True
    )
    # print('possiblePos:', possiblePos)
    # TODO check if calLegalPos is correct also when possiblePos is empty for some batch idx
    if len(possible_pos) == 0:
        startTime_a = putInTheEnd(
            actions,
            ops_ready_time,
            machines_ready_time,
            start_times_for_selected_machines,
            ops_for_selected_machines,
        )
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(
            ops_duration,
            ops_ready_time,
            durations,
            possible_pos,
            start_times_for_selected_machines,
            ops_for_selected_machines,
        )
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(
                actions,
                ops_ready_time,
                machines_ready_time,
                start_times_for_selected_machines,
                ops_for_selected_machines,
            )
        else:
            flag = True
            startTime_a = putInBetween(
                actions,
                idxLegalPos,
                legalPos,
                endTimesForPossiblePos,
                start_times_for_selected_machines,
                ops_for_selected_machines,
            )
    return startTime_a, flag


def putInTheEnd(
    actions,
    ops_ready_time,
    machines_ready_time,
    start_times_for_selected_machines,
    opsIDsForMchOfa,
):
    # index = first position of -config.high in start_times_for_selected_machines
    # print('Yes!OK!')
    index = np.where(start_times_for_selected_machines == -configs.high)[0][0]
    startTime_a = max(ops_ready_time, machines_ready_time)
    start_times_for_selected_machines[index] = startTime_a
    opsIDsForMchOfa[index] = actions
    return startTime_a


def calLegalPos(
    dur_a,
    ops_ready_time,
    durations,
    possiblePos,
    start_times_for_selected_machines,
    opsIDsForMchOfa,
):
    startTimesOfPossiblePos = start_times_for_selected_machines[possiblePos]
    durOfPossiblePos = np.take(durations, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(
        ops_ready_time,
        start_times_for_selected_machines[possiblePos[0] - 1]
        + np.take(durations, [opsIDsForMchOfa[possiblePos[0] - 1]]),
    )
    endTimesForPossiblePos = np.append(
        startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos)
    )[
        :-1
    ]  # end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(
    actions,
    idxLegalPos,
    legalPos,
    endTimesForPossiblePos,
    start_times_for_selected_machines,
    opsIDsForMchOfa,
):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    start_times_for_selected_machines[:] = np.insert(
        start_times_for_selected_machines, earlstPos, startTime_a
    )[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, actions)[:-1]
    return startTime_a


def job_machines_ready_time(
    actions: torch.Tensor,
    machines: torch.Tensor,
    durations: torch.Tensor,
    machines_start_times: torch.Tensor,
    operations_on_machines: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the ready time on the job and machine for the selected operations (=actions).

    Args:
        actions (torch.Tensor): actions taken by the agent
            shape (batch_size, 1)
        machines (torch.Tensor): matrices with the indexes of the machines for each task.
            For example, if machines[0, 1, 2] = 3, it means that
            the task 2 of the job 1 is executed on the machine 3 in batch 0.
            shape (batch_size, num_jobs, num_machines)
        durations (torch.Tensor): matrices with the duration of each task.
            For example, if durations[0, 1, 2] = 3, it means that
            the task 2 of the job 1 takes 3 time units in batch 0.
            shape (batch_size, num_jobs, num_machines)
        machines_start_times (torch.Tensor): matrices with the starting time of each task.
            For example, if machines_start_times[0, 1, 2] = 3, it means that
            the task 2 of the job 1 starts at time 3 in batch 0.
            shape (batch_size, num_jobs, num_machines)
        operations_on_machines (torch.Tensor): matrices with the indexes of the operations executed on each machine.
            For example, if operations_on_machines[0, 1, 2] = 3, it means that
            the machine of index 1 executes the operation 3 at position 2 in batch 0.
            shape (batch_size, num_machines, num_jobs)

    Returns:
        ops_ready_time (torch.Tensor): ready time of the selected operations
            shape (batch_size, 1)
        machines_ready_time (torch.Tensor): ready time of the machines
            shape (batch_size, 1)
    """
    # flatten the actions to use them as indexes
    actions = actions.flatten(0)
    batch_idxs = torch.arange(machines.shape[0])
    num_machines = machines.shape[2]
    rows = actions // num_machines
    cols = actions % num_machines
    selected_machines = machines[batch_idxs, rows, cols] - 1

    ## Compute the ready time of the selected operation
    # compute the previous op in the job, use dummy value -1 if no previous op
    previous_op_in_job = actions - 1 if actions % num_machines != 0 else -1
    # extract the duration of the previous op in the job
    duration_previous_op_in_job = durations.flatten(1)[batch_idxs, previous_op_in_job]
    # give duration 0 if previous_op_in_job is -1
    duration_previous_op_in_job[(previous_op_in_job == -1).flatten()] = 0
    # extract the machine of the previous op in the job
    machine_previous_op_in_job = (
        machines.flatten(1)[batch_idxs, previous_op_in_job] - 1
    )  # -1 since machines start from 1
    # dummy machine value -1 if no previous op
    machine_previous_op_in_job[(previous_op_in_job == -1).flatten()] = -1

    # compute the ready time of the operations
    ops_ready_time = (
        machines_start_times[batch_idxs, machine_previous_op_in_job][
            (operations_on_machines == previous_op_in_job.view(-1, 1, 1)).nonzero(
                as_tuple=True
            )
        ]
        + duration_previous_op_in_job
    )
    ops_ready_time[(previous_op_in_job == -1).flatten()] = 0

    ## Compute the ready time of the machines
    # Create a mask for non negative elements
    nonnegative_mask = (operations_on_machines[:, machines, :] >= 0).nonzero()

    # Initialize result tensor with -1
    previous_op_on_machine_idx = torch.full(
        (operations_on_machines.size(0),), -1, dtype=torch.long
    )

    # compute the index of the last non zero element for each row
    previous_op_on_machine_idx[nonnegative_mask[:, 0]] = nonnegative_mask[:, 1]
    # extract the value in each row
    previous_op_on_machine = operations_on_machines[
        batch_idxs, selected_machines, previous_op_on_machine_idx
    ]

    # extract the duration of the previous op in the machine
    duration_previous_op_in_machine = durations.flatten(1)[
        batch_idxs, previous_op_on_machine
    ]
    duration_previous_op_in_machine[(previous_op_on_machine < 0).flatten()] = 0
    # compute the ready time of the machines
    machines_ready_time = (
        machines_start_times[batch_idxs, selected_machines][
            (operations_on_machines == previous_op_on_machine.view(-1, 1, 1)).nonzero(
                as_tuple=True
            )
        ]
        + duration_previous_op_in_job
    )
    machines_ready_time[(previous_op_on_machine < 0).flatten()] = 0
    return ops_ready_time, machines_ready_time


# if __name__ == "__main__":
#     import time
#
#     from JSSP_Env import SJSSP
#     from uniform_instance_gen import uni_instance_gen
#
#     n_j = 3
#     n_m = 3
#     low = 1
#     high = 99
#     SEED = 10
#     np.random.seed(SEED)
#     env = SJSSP(n_j=n_j, n_m=n_m)
#
#     """arr = np.ones(3)
#     idces = np.where(arr == -1)
#     print(len(idces[0]))"""
#
#     # rollout env random action
#     t1 = time.time()
#     data = uni_instance_gen(n_j=n_j, n_m=n_m, low=low, high=high)
#     print("Dur")
#     print(data[0])
#     print("Mach")
#     print(data[-1])
#     print()
#
#     # start time of operations on machines
#     machines_start_times = -configs.high * np.ones_like(
#         data[0].transpose(), dtype=np.int32
#     )
#     # Ops ID on machines
#     operations_on_machines = -n_j * np.ones_like(data[0].transpose(), dtype=np.int32)
#
#     # random rollout to test
#     # count = 0
#     _, _, omega, mask = env.reset(data)
#     rewards = []
#     flags = []
#     # ts = []
#     while True:
#         action = np.random.choice(omega[np.where(mask == 0)])
#         print(action)
#         machines_actions = np.take(data[-1], action) - 1
#         # print(machines_actions)
#         # print('action:', action)
#         # t3 = time.time()
#         adj, _, reward, done, omega, mask = env.step(action)
#         # t4 = time.time()
#         # ts.append(t4 - t3)
#         # ops_ready_time, machines_ready_time = job_machines_ready_time(actions=action, machines=data[-1], durations=data[0], machines_start_times=machines_start_times, operations_on_machines=operations_on_machines)
#         # print('machines_ready_time:', machines_ready_time)
#         startTime_a, flag = permissible_left_shift(
#             actions=action,
#             durations=data[0].astype(np.single),
#             machines=data[-1],
#             machines_start_times=machines_start_times,
#             operations_on_machines=operations_on_machines,
#         )
#         flags.append(flag)
#         # print('startTime_a:', startTime_a)
#         # print('machines_start_times\n', machines_start_times)
#         # print('NOOOOOOOOOOOOO' if not np.array_equal(env.machines_start_times, machines_start_times) else '\n')
#         print("operations_on_machines\n", operations_on_machines)
#         # print('LBs\n', env.LBs)
#         rewards.append(reward)
#         # print('ET after action:\n', env.LBs)
#         print()
#         if env.done():
#             break
#     t2 = time.time()
#     print(t2 - t1)
#     # print(sum(ts))
#     # print(np.sum(operations_on_machines // n_m, axis=1))
#     # print(np.where(machines_start_times == machines_start_times.max()))
#     # print(operations_on_machines[np.where(machines_start_times == machines_start_times.max())])
#     print(
#         machines_start_times.max()
#         + np.take(
#             data[0],
#             operations_on_machines[
#                 np.where(machines_start_times == machines_start_times.max())
#             ],
#         )
#     )
#     # np.save('sol', operations_on_machines // n_m)
#     # np.save('jobSequence', operations_on_machines)
#     # np.save('testData', data)
#     # print(machines_start_times)
#     durAlongMchs = np.take(data[0], operations_on_machines)
#     mchsEndTimes = machines_start_times + durAlongMchs
#     print(machines_start_times)
#     print(mchsEndTimes)
#     print()
#     print(env.operations_on_machines)
#     print(env.adj)
#     # print(sum(flags))
#     # data = np.load('data.npy')
#
#     # print(len(np.where(np.array(rewards) == 0)[0]))
#     # print(rewards)
