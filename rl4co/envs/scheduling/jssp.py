import numpy as np


class Configs:
    pass


configs = Configs()


def permissibleLeftShift(
    actions,
    duration_matrices,
    machines_matrices,
    machines_starting_times,
    operations_on_machines,
):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(
        actions,
        machines_matrices,
        duration_matrices,
        machines_starting_times,
        operations_on_machines,
    )
    dur_a = np.take(duration_matrices, actions)
    machines_actions = np.take(machines_matrices, actions) - 1
    startTimesForMchOfa = machines_starting_times[machines_actions]
    opsIDsForMchOfa = operations_on_machines[machines_actions]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(
            actions, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa
        )
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(
            dur_a,
            jobRdyTime_a,
            duration_matrices,
            possiblePos,
            startTimesForMchOfa,
            opsIDsForMchOfa,
        )
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(
                actions, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa
            )
        else:
            flag = True
            startTime_a = putInBetween(
                actions,
                idxLegalPos,
                legalPos,
                endTimesForPossiblePos,
                startTimesForMchOfa,
                opsIDsForMchOfa,
            )
    return startTime_a, flag


def putInTheEnd(
    actions, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa
):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -configs.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = actions
    return startTime_a


def calLegalPos(
    dur_a,
    jobRdyTime_a,
    duration_matrices,
    possiblePos,
    startTimesForMchOfa,
    opsIDsForMchOfa,
):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(duration_matrices, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(
        jobRdyTime_a,
        startTimesForMchOfa[possiblePos[0] - 1]
        + np.take(duration_matrices, [opsIDsForMchOfa[possiblePos[0] - 1]]),
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
    startTimesForMchOfa,
    opsIDsForMchOfa,
):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, actions)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(
    actions,
    machines_matrices,
    duration_matrices,
    machines_starting_times,
    operations_on_machines,
):
    rows = actions // machines_matrices.shape[2]
    cols = actions % machines_matrices.shape[2]
    machines_actions = machines_matrices[:, rows, cols] - 1
    # cal jobRdyTime_a
    jobPredecessor = actions - 1 if actions % machines_matrices.shape[2] != 0 else None
    if jobPredecessor is not None:
        durJobPredecessor = np.take(duration_matrices, jobPredecessor)
        mchJobPredecessor = np.take(machines_matrices, jobPredecessor) - 1
        jobRdyTime_a = (
            machines_starting_times[mchJobPredecessor][
                np.where(operations_on_machines[mchJobPredecessor] == jobPredecessor)
            ]
            + durJobPredecessor
        ).item()
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a
    mchPredecessor = (
        operations_on_machines[machines_actions][
            np.where(operations_on_machines[machines_actions] >= 0)
        ][-1]
        if len(np.where(operations_on_machines[machines_actions] >= 0)[0]) != 0
        else None
    )
    if mchPredecessor is not None:
        durMchPredecessor = np.take(duration_matrices, mchPredecessor)
        mchRdyTime_a = (
            machines_starting_times[machines_actions][
                np.where(machines_starting_times[machines_actions] >= 0)
            ][-1]
            + durMchPredecessor
        ).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a


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
#     machines_starting_times = -configs.high * np.ones_like(
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
#         # jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(actions=action, machines_matrices=data[-1], duration_matrices=data[0], machines_starting_times=machines_starting_times, operations_on_machines=operations_on_machines)
#         # print('mchRdyTime_a:', mchRdyTime_a)
#         startTime_a, flag = permissibleLeftShift(
#             actions=action,
#             duration_matrices=data[0].astype(np.single),
#             machines_matrices=data[-1],
#             machines_starting_times=machines_starting_times,
#             operations_on_machines=operations_on_machines,
#         )
#         flags.append(flag)
#         # print('startTime_a:', startTime_a)
#         # print('machines_starting_times\n', machines_starting_times)
#         # print('NOOOOOOOOOOOOO' if not np.array_equal(env.machines_starting_times, machines_starting_times) else '\n')
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
#     # print(np.where(machines_starting_times == machines_starting_times.max()))
#     # print(operations_on_machines[np.where(machines_starting_times == machines_starting_times.max())])
#     print(
#         machines_starting_times.max()
#         + np.take(
#             data[0],
#             operations_on_machines[
#                 np.where(machines_starting_times == machines_starting_times.max())
#             ],
#         )
#     )
#     # np.save('sol', operations_on_machines // n_m)
#     # np.save('jobSequence', operations_on_machines)
#     # np.save('testData', data)
#     # print(machines_starting_times)
#     durAlongMchs = np.take(data[0], operations_on_machines)
#     mchsEndTimes = machines_starting_times + durAlongMchs
#     print(machines_starting_times)
#     print(mchsEndTimes)
#     print()
#     print(env.operations_on_machines)
#     print(env.adj)
#     # print(sum(flags))
#     # data = np.load('data.npy')
#
#     # print(len(np.where(np.array(rewards) == 0)[0]))
#     # print(rewards)
