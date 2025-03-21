import torch
import numpy as np
from Params import WirelessNetwork
from collections import deque
from torch.distributions import Categorical
import torch.optim as optim
from models import SimpleTransformerEncoder, Policy
import torch.nn.functional as F
import random
import pickle
import copy
from utlis import sample_index

torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# set net work

task = f"train_rbg"
name = f"RL_Test"
RBGs = 17
UEs = 100
Total_TTI = 1500
MU_B_VIP = 70000
SIMGA_B_VIP = 1150
MU_B = 25000
SIMGA_B = 1100
SP_sigma = 6
alpha_1 = 0.8
alpha_2 = 0.2
SP_N = [random.gauss(0, SP_sigma) for _ in range(UEs)]
Env_rl = WirelessNetwork(N=UEs, M=RBGs, mu_B_vip=MU_B_VIP, sigma_B_vip=SIMGA_B_VIP, mu_B=MU_B, sigma_B=SIMGA_B,
                         SP_N=SP_N)
# Env_rand=WirelessNetwork(N=UEs,M=RBGs,mu_B_vip=MU_B_VIP,sigma_B_vip=SIMGA_B_VIP,mu_B=MU_B,sigma_B=SIMGA_B,SP_N=SP_N)
VIP_list = Env_rl.Vip_index  # Note:ensure they have same VIP UEs

lr = 1e-2
hidden_dim = 64
nT = 8
input_dim = 2 * nT + 1 + 3 + 1 + RBGs+1
output_dim = RBGs
fixed_size = False
if fixed_size:
    policy = Policy(num_inputs=input_dim * 10, num_outputs=output_dim * 10, hidden_size=hidden_dim).to(device)
else:
    policy = SimpleTransformerEncoder(input_dim, output_dim=output_dim, hidden_dim=hidden_dim).to(device)

# optimizer = optim.Adam(policy.parameters(), lr=lr)
optimizer = optim.Adam(policy.parameters(), lr=lr)

eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = np.array(state)

    state = torch.from_numpy(state).float()

    probs = policy(state)

    X_flat=Env_rl.last_X.flatten()
    none=np.where(X_flat>=3)[0]



    # 将tensor展平为一维
    if fixed_size != True:
        flat_tensor = probs.view(-1)
        # 对展平后的tensor计算softmax，得到概率分布
        probs = F.softmax(flat_tensor, dim=0)

    # if len(none)>0:
    #     new_probs=probs.clone()
    #     new_probs[none]=torch.tensor(0).float()
    #     sum_other = new_probs.sum()
    #     probs=new_probs/sum_other

    m = Categorical(probs)
    # top1 = torch.argmax(probs)

    action = m.sample()

    policy.saved_log_probs.append(m.log_prob(action))


    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + 0.95 * R
        returns.appendleft(R)
    returns = torch.tensor(returns)

    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)


    policy_loss = sum(policy_loss)

    optimizer.zero_grad()
    policy_loss.backward()
    print("loss:------------------------", policy_loss)
    optimizer.step()


    del policy.rewards[:]
    del policy.saved_log_probs[:]


B_que_rl = []
sinrs = []
nums = []
for i in range(UEs):
    q1 = []

    if i in VIP_list:

        for t in range(Total_TTI):
            x = random.gauss(MU_B_VIP, SIMGA_B_VIP)
            q1.append(x)

    else:
        for t in range(Total_TTI):
            x = random.gauss(MU_B, SIMGA_B)
            q1.append(x)

    B_que_rl.append(q1)
Env_rl.get_data_list(B_list=B_que_rl)
# Env_rand.get_data_list(B_list=B_que_rand)
Env_rl.reset_B()

if __name__ == '__main__':

    all_data = []
    pre_data = []
    timelog = 50
    Env_rl.timesteps = timelog

    delta_rl = []
    H_change = 80
    # suc_rl = []
    # base_rate = []
    # rl_rate = []

    running_reward = 10
    pl = 100

    # get H
    H = Env_rl.H
    V = Env_rl.V

    # get state.
    Env_rl.init_P()
    state = Env_rl.getState()
    if fixed_size:
        state = state.flatten().copy()

    for t in range(timelog * Total_TTI):  # Don't infinite loop while learning
        # get action

        active_UE_base = Env_rl.active_UE.copy()
        action = select_action(state)
        RBG_index = action // len(Env_rl.active_UE)
        UE_index = action - RBG_index * len(Env_rl.active_UE)
        # step env
        state, reward, done, info = Env_rl.step((UE_index, RBG_index))
        if fixed_size:
            state = state.flatten().copy()

        policy.rewards.append(reward)
        # print(Env_rl.last_P)
        if done:
            finish_episode()

            if (t + 1) == timelog * 2:
                pass
                # print(t)
                # quit()
            finnal_Power = Env_rl.last_P.copy()
            actual_rate, data = Env_rl.update(finnal_Power)
            record = [Env_rl.VIP_times_record.copy(), Env_rl.VIP_rate_record.copy(), active_UE_base,
                      actual_rate.copy()]

            data.append(record)
            all_data.append(data)
            with open(f"./{name}.pkl", "wb") as file:
                pickle.dump(all_data, file)
            pre_data.append(Env_rl.preference)
            with open(f"./preference_{name}.pkl", "wb") as file:
                pickle.dump(pre_data, file)
            Env_rl.init_P()
            state = Env_rl.getState()
            if fixed_size:
                state = state.flatten().copy()
