import numpy
import argparse
from M2M_3   import  M2M_baseline_abla
import numpy as np
from sinr_new import RRMSimulator
import os
import pickle
from utlis import *
from Params import WirelessNetwork
import copy
from sinr_new import RRMSimulator
from error_event import Mcs_Sinr_p
import time as tm
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-q', '--Q', type=int, default=20000)
parser.add_argument('-p1', '--prob1', type=float, default=0.5)
parser.add_argument('-p2', '--prob2', type=float, default=0.4)  # radius
parser.add_argument('-p3', '--prob3', type=float, default=0.4)  # mode
parser.add_argument('-s', '--safe', type=int, default=1)  # mode
parser.add_argument('-e', '--seed', type=int, default=43)  # mode
parser.add_argument('-f', '--phi', type=float, default=8e-12)
args = parser.parse_args()

seed = args.seed

np.random.seed(seed)
bit_error = True


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalization_1(data):
    _range = np.sum(data)
    return (data) / _range


VIP_TARGET=58000
name=f"Baseline_Test_{VIP_TARGET}_ev"
RBGs = 17
UEs = 100
Total_TTI = 1000
MU_B_VIP=150000
SIMGA_B_VIP=1000
MU_B=80000
SIMGA_B=1000
SP_sigma=6
alpha_1=0.8
alpha_2=0.2
SP_N = [random.gauss(0, SP_sigma) for _ in range(UEs)]
Env_base=WirelessNetwork(N=UEs,M=RBGs,mu_B_vip=MU_B_VIP,sigma_B_vip=SIMGA_B_VIP,mu_B=MU_B,sigma_B=SIMGA_B,SP_N=SP_N)
# Env_rand=WirelessNetwork(N=UEs,M=RBGs,mu_B_vip=MU_B_VIP,sigma_B_vip=SIMGA_B_VIP,mu_B=MU_B,sigma_B=SIMGA_B,SP_N=SP_N)
VIP_list=Env_base.Vip_index #Note:ensure they have same VIP UEs

if __name__ == "__main__":
    B_que_base = []
    B_que_rand = []

    sinrs = []
    nums = []
    for i in range(UEs):
        q1 = []
        q2 = []
        if i in VIP_list:

            for t in range(Total_TTI):
                x = random.gauss(MU_B_VIP, SIMGA_B_VIP)
                q1.append(x)
                q2.append(x)
        else:
            for t in range(Total_TTI):
                x = random.gauss(MU_B, SIMGA_B)
                q1.append(x)
                q2.append(x)
        B_que_base.append(q1)
        B_que_rand.append(q2)

    Env_base.get_data_list(B_list=B_que_base)
    # Env_rand.get_data_list(B_list=B_que_rand)
    Env_base.reset_B()
    # Env_rand.reset_B()
    # Env_base.refresh_data()


    all_data=[]

    for time in range(Total_TTI):
        print("---------------------------------")
        UE_num = len(Env_base.active_UE)
        list_ue=Env_base.active_UE
        X = np.random.binomial(n=1, p=0.1, size=(UE_num, 17))
        P = normalize_columns(X)

        #To get a Power allocation!
        #B_base
        active_UE_base=Env_base.active_UE.copy()
        all_B_base=Env_base.B_N
        # print(Env_base.active_UE)
        # print(Env_base.B_N[Env_base.active_UE])

        #H, V, all_HV_2
        H=Env_base.H

        V=Env_base.V
        Phi=Env_base.Phi
        x=0
        active_vip=Env_base.active_vip


        GBR_info =(Env_base.VIP_rate_record,Env_base.VIP_times_record,active_vip,Env_base.Vip_index,VIP_TARGET)
        P,ave_sinr,ave_mcs,_,ave_count,_ = M2M_baseline_abla(B=all_B_base, active_UE=active_UE_base,
                                                                                      channel=(H, V),
                                                                                      epf=Env_base.epf_param,beta=Env_base.all_beta,GBR=GBR_info)
        print("功率",P)
        dec=[ave_mcs,ave_count]
        print("request:", Env_base.B_N[Env_base.Vip_index])
        actual_rate,data=Env_base.update(P,dec=dec)


        record=[Env_base.VIP_times_record.copy(),Env_base.VIP_rate_record.copy(),active_UE_base,actual_rate.copy()]
        # print(Env_base.VIP_times_record)


        data.append(record)
        all_data.append(data)
        end=tm.time()

        with open(f"./{name}_Phi.pkl", "wb") as file:
            pickle.dump(all_data, file)

        Env_base.epf_param[active_UE_base] =alpha_1*Env_base.epf_param[active_UE_base]+ alpha_2*actual_rate.copy()

        # if time==10:
        #     quit()


    quit()


