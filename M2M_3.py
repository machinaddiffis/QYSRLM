import numpy as np
import random
random.seed(0)
from utlis import compute_MCS,getBaselineMcsSinr,getSendRate,compute_sinr_loop

def mcs_decrease(ref_mcs,de):

    ref_mcs = ref_mcs + de

    for i in range(ref_mcs.shape[1]):
        if ref_mcs[0, i] < -1:
            ref_mcs[0, i] = -1
        # if ref_mcs[0, i] == -1:
        #     ref_mcs[0, i] = 0
        if ref_mcs[0, i] > 28:
            ref_mcs[0, i] = 28
    return ref_mcs



def M2M_baseline_abla(B=None,channel=None,active_UE=None,de=0,epf=None,beta=None,GBR=None):
    """
    :param Q: 用户需求，N
    :param H: 信道矩阵 N x M x nT
    :param Phi_now: 实时干扰矩阵 N x M
    :param Phi_past: 延时干扰矩阵 N x M
    :return:X: N x M 分配决策, P: N x M 功率决策, MCS_avg 平均MCS
    """
    H,V = channel
    User_num = len(active_UE)
    RBG_num = H.shape[1]


    X = np.zeros((User_num,RBG_num))
    P = np.zeros((User_num, RBG_num))

    # Q=B[active_UE]
    # init_reside=1/(Q)
    # new_epf=epf.copy()#已经发送量
    # new_reside=init_reside.copy() #1/剩余量
    #
    #
    #
    # if epf is not None:
    #     scores = np.multiply(new_reside, 1 / (new_epf[active_UE] + 1))
    #     # 得到逆序
    #     oder = np.argsort(-scores)
    #
    # random.shuffle(oder)
    RBG_Rate_avg=0


    ##EPF calculate
    Q = B[active_UE]

    ##### ---整个这一段，都在计算一个用户的顺序---
    upterm =1
    downterm=(1+epf.copy())[active_UE]
    EPF=upterm/downterm
    if GBR is not None:
        VIP_rate_record,VIP_times_record,active_vip,Vip_index,vip_target=GBR
    ##weights calculate
    delta_N=np.ones(User_num)
    if active_vip is not None:
        for i in range(len(active_vip)):
            index=active_vip[i]
            real_index=np.where(Vip_index==index)[0]
            temp=VIP_rate_record[real_index]/VIP_times_record[real_index]
            if temp==0:
                delta=1.5
            else:
                delta=vip_target/temp
            k=np.where(active_UE==index)[0]
            delta_N[k]=delta
        print("delta序列！！！！！",delta_N)
    final_scores=delta_N*EPF
    order = np.argsort(-final_scores)
    ##### ---整个这一段，都在计算一个用户的顺序---



    for j in range(RBG_num):#对RB进行遍历

        for i in range(User_num): #对用户进行遍历
            User_ID = order[i]
            # User_ID = i
            outer_User_ID=active_UE[User_ID]
            # print(Q,Q[User_ID],i,outer_User_ID,User_ID)

            if Q[User_ID]==0:#检测用户是否有需求
                continue

            #贪心分配
            X[User_ID,j]=1 #新UE

            RBG_User_num=sum(X[:,j])

            P[:,j]=X[:,j]/RBG_User_num #平均功率

            #计算当前RBG内的所有用户传输速率： input X[：,j],P[:,j]，H[User_ID,:,:], Phi[,:], output M-d vector
            SINR, MCS = compute_sinr_loop(X, H[active_UE], V[active_UE])


            ave_mcs, ave_count, ave_sinr = getBaselineMcsSinr(MCS, SINR)


            # print(ave_mcs)
            if de!=0:
                ave_mcs=mcs_decrease(ave_mcs,de)
            rate = getSendRate(ave_mcs, ave_count)

            #计算所有用户实际传输数据量：
            actual_rate = np.minimum(Q, rate)


            if sum(actual_rate) < RBG_Rate_avg:#速率下降，还原
                X[User_ID, j] = 0
                RBG_User_num = sum(X[:, j])
                P[:, j] = X[:, j] / RBG_User_num  # 平均功率
                continue
            else:
                RBG_Rate_avg=sum(actual_rate)

    SINR, MCS = compute_sinr_loop(X, H[active_UE], V[active_UE])
    ave_mcs, ave_count, ave_sinr = getBaselineMcsSinr(MCS, SINR)

    if de != 0:
        ave_mcs = mcs_decrease(ave_mcs, de)

    rate = getSendRate(ave_mcs, ave_count)

    # 计算所有用户实际传输数据量：
    actual_rate = np.minimum(Q, rate)

    # actual_rate = np.minimum(Q, 100)



    return P,ave_sinr,ave_mcs,actual_rate,ave_count,rate



def M2M_baseline_co(B=None,channel=None,active_UE=None,de=0,epf=None,beta=None,GBR=None):
    """
    :param Q: 用户需求，N
    :param H: 信道矩阵 N x M x nT
    :param Phi_now: 实时干扰矩阵 N x M
    :param Phi_past: 延时干扰矩阵 N x M
    :return:X: N x M 分配决策, P: N x M 功率决策, MCS_avg 平均MCS
    """
    H,V = channel
    User_num = len(active_UE)
    RBG_num = H.shape[1]


    X = np.zeros((User_num,RBG_num))
    P = np.zeros((User_num, RBG_num))

    # Q=B[active_UE]
    # init_reside=1/(Q)
    # new_epf=epf.copy()#已经发送量
    # new_reside=init_reside.copy() #1/剩余量
    #
    #
    #
    # if epf is not None:
    #     scores = np.multiply(new_reside, 1 / (new_epf[active_UE] + 1))
    #     # 得到逆序
    #     oder = np.argsort(-scores)
    #
    # random.shuffle(oder)
    RBG_Rate_avg=0


    ##EPF calculate
    Q = B[active_UE]

    upterm = (np.array(beta))[active_UE]
    # upterm =1
    downterm=(1+epf.copy())[active_UE]
    EPF=upterm/downterm
    if GBR is not None:
        VIP_rate_record,VIP_times_record,active_vip,Vip_index,vip_target=GBR
    ##weights calculate
    delta_N=np.ones(User_num)
    if active_vip is not None:
        for i in range(len(active_vip)):

            index=active_vip[i]
            real_index=np.where(Vip_index==index)[0]
            temp=VIP_rate_record[real_index]/VIP_times_record[real_index]
            if temp==0:
                delta=1.5
            else:
                delta=vip_target/temp
            k=np.where(active_UE==index)[0]
            delta_N[k]=delta
        print("delta序列！！！！！",delta_N)

    final_scores=delta_N*EPF

    order = np.argsort(-final_scores)


    for j in range(RBG_num):#对RB进行遍历

        for i in range(User_num): #对用户进行遍历
            User_ID = order[i]
            # User_ID = i
            outer_User_ID=active_UE[User_ID]
            # print(Q,Q[User_ID],i,outer_User_ID,User_ID)

            if Q[User_ID]==0:#检测用户是否有需求
                continue

            #贪心分配
            X[User_ID,j]=1 #新UE

            RBG_User_num=sum(X[:,j])

            P[:,j]=X[:,j]/RBG_User_num #平均功率

            #计算当前RBG内的所有用户传输速率： input X[：,j],P[:,j]，H[User_ID,:,:], Phi[,:], output M-d vector
            SINR, MCS = compute_sinr_loop(X, H[active_UE], V[active_UE])


            ave_mcs, ave_count, ave_sinr = getBaselineMcsSinr(MCS, SINR)


            # print(ave_mcs)
            if de!=0:
                ave_mcs=mcs_decrease(ave_mcs,de)
            rate = getSendRate(ave_mcs, ave_count)

            #计算所有用户实际传输数据量：
            actual_rate = np.minimum(Q, rate)


            if sum(actual_rate) < RBG_Rate_avg:#速率下降，还原
                X[User_ID, j] = 0
                RBG_User_num = sum(X[:, j])
                P[:, j] = X[:, j] / RBG_User_num  # 平均功率
                continue
            else:
                RBG_Rate_avg=sum(actual_rate)

    SINR, MCS = compute_sinr_loop(X, H[active_UE], V[active_UE])
    ave_mcs, ave_count, ave_sinr = getBaselineMcsSinr(MCS, SINR)

    if de != 0:
        ave_mcs = mcs_decrease(ave_mcs, de)

    rate = getSendRate(ave_mcs, ave_count)

    # 计算所有用户实际传输数据量：
    actual_rate = np.minimum(Q, rate)

    # actual_rate = np.minimum(Q, 100)



    return P,ave_sinr,ave_mcs,actual_rate,ave_count,rate