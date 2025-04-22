import random
import math

import numpy as np

from utlis import *
import copy
from sinr_new import RRMSimulator
random.seed(0)

class WirelessNetwork:
    def __init__(self, K=1, M=17, Tx=8, N=100, Rx=1, N_vip=10, dmin=10, dmax=1000,
                 d_0=1, PL_0=0, alpha=2, SP_sigma=6, N_max=8, gamma=None,
                 mu_B_vip=100000, sigma_B_vip=150, mu_B=80000, sigma_B=100,SP_N=None):
        self.K = K
        self.M = M
        self.Tx = Tx
        self.N = N
        self.Rx = Rx
        self.N_vip = N_vip
        self.dmin = dmin
        self.dmax = dmax
        self.d_0 = d_0
        self.PL_0 = PL_0
        self.alpha = alpha
        self.SP_sigma = SP_sigma
        self.N_max = N_max
        self.gamma = gamma if gamma is not None else (N_max / N)*2

        self.mu_B_vip = mu_B_vip
        self.sigma_B_vip = sigma_B_vip
        self.mu_B = mu_B
        self.sigma_B = sigma_B
        self.sigma_int=2e-2

        self.preference=generate_random_vector()

        self.GNS=1


        self.error = Mcs_Sinr_p()
        self.error.load_table()
        self.error.create_inte()


        #Datelist from outside
        self.all_B_request=None

        self.SP_N=SP_N

        self.reset()

        self.slot=0
        self.inner_slot=None
        self.timesteps=20

        self.epf_param=np.zeros(self.N)

        self.state_state=0



    def reset(self):
        if self.SP_N is not None:
            pass
        else:
            self.SP_N = [random.gauss(0, self.SP_sigma) for _ in range(self.N)]  # set random seed
        self.t=0
        self.B_N =np.array( [0] * self.N)
        self.Vip_index = [i for i in range(self.N_vip)]
        # self.VIP_times_record = np.array([0]*len(self.Vip_index))#list 0 for recording times, list 1 for recording RateSum
        # self.VIP_rate_record = np.array([0.]*len(self.Vip_index))
        self.VIP_times_record = np.array([0] * self.N)  # list 0 for recording times, list 1 for recording RateSum
        self.VIP_rate_record = np.array([0.] * self.N)
        self.UE_point = random_points(self.dmin, self.dmax, self.N)
        self.edge_index=[95,96,97,98,99]

        self.all_beta = beta_N(self.SP_N, self.UE_point, self.PL_0, self.d_0, self.alpha,VIP_TUNE=self.Vip_index)
        self.active_UE = sample_vector(self.gamma, self.N)
        intern=[item for item in self.active_UE if item not in self.edge_index]
        intern = [item for item in intern if item not in self.Vip_index]

        vip_option = random.choice(self.Vip_index)
        edge_option = random.choice(self.edge_index)
        if len(intern)+2 - self.N_max > 0:
            num = len(intern)+2 - self.N_max
            self.active_UE = remove_random_elements(intern, num)
        self.active_UE.extend([vip_option])
        self.active_UE.extend([edge_option])
        self.active_vip=common_elements(self.active_UE, self.Vip_index)


        # if len(self.active_UE)+2 - self.N_max > 0:
        #     num = len(self.active_UE)+2 - self.N_max
        #     self.active_UE = remove_random_elements(self.active_UE, num)
        #
        # self.active_vip=common_elements(self.active_UE, self.Vip_index)




        self.VIP_times_record[self.active_UE]+=1
        if self.active_vip==None:
            self.active_normal=self.active_UE
        else:
            # for n in self.active_vip:
                # self.B_N[n] = random.gauss(self.mu_B_vip, self.sigma_B_vip)##1
            self.active_normal = remove_elements(self.active_UE, self.active_vip)
            # vip_add_info = [self.Vip_index.index(item) if item in self.Vip_index else -1 for item in self.active_vip]
            #
            # self.VIP_times_record[vip_add_info] += 1
            # print(self.VIP_times_record)

        #sample B
        # for n in self.active_normal:
        #     self.B_N[n]=random.gauss(self.mu_B, self.sigma_B)##2

        #calculate H
        Phi=[]
        H=[]
        for n in range(self.N):
            # G_n=generate_rayleigh_matrix(self.M,self.Tx,s=self.GNS)
            # H_n=G_n*np.sqrt(self.all_beta[n])
            # # Phi_n=generate_rayleigh_matrix(self.M,1,s=self.sigma_int).squeeze(-1)
            # # H.append(H_n)
            # # Phi.append(Phi_n)
            G_n = generate_rayleigh_matrix(1, self.Tx, s=self.GNS)
            G_n=np.array([G_n]*self.M).squeeze(axis=1)

            H_n = G_n * np.sqrt(self.all_beta[n])
            Phi_n = generate_rayleigh_matrix(self.M, 1, s=self.sigma_int).squeeze(-1)
            H.append(H_n)
            Phi.append(Phi_n)


        self.Phi=np.array(Phi)
        # print(self.Phi[0])

        self.H = np.array(H)

        self.V = compute_zf_precoder(self.H)

        self.slot=0

    def reset_H(self):
        H = []
        for n in range(self.N):
            G_n = generate_rayleigh_matrix(self.M, self.Tx,s=self.GNS)
            H_n = G_n * np.sqrt(self.all_beta[n])
            H.append(H_n)
        self.H = np.array(H)
        self.V = compute_zf_precoder(self.H)

    def reset_Phi(self):
        Phi = []
        for n in range(self.N):
            Phi_n = generate_rayleigh_matrix(self.M, 1, s=self.sigma_int).squeeze(-1)
            Phi.append(Phi_n)
        self.Phi = np.array(Phi)

    def reset_B(self):

        if self.active_vip == None:
            self.active_normal = self.active_UE
        else:
            for n in self.active_vip:
                self.B_N[n]= self.all_B_request[n].pop(0)##1 VIP
            self.active_normal = remove_elements(self.active_UE, self.active_vip)
        for n in self.active_normal:
            self.B_N[n]=self.all_B_request[n].pop(0)##2 Nomarl


    def get_data_list(self,B_list):
        self.all_B_request=B_list

    def refresh_data(self,delta):
        if self.all_B_request is None:
            print("-----Date is not correct!-----")
            quit()
        else:
            for index in delta:
                if index in self.Vip_index:
                    self.B_N[index] = self.all_B_request[index].pop(0)  ##
                else:
                    self.B_N[index] = self.all_B_request[index].pop(0)  ##
            pass
        pass

    def init_P(self):
        self.last_X=np.zeros((len(self.active_UE),self.M))
        self.last_P=np.zeros((len(self.active_UE),self.M))
        self.last_mcs=0

    def step(self,action):
        self.act_H = self.H[self.active_UE]
        self.act_V = self.V[self.active_UE]
        self.act_Phi = self.Phi[self.active_UE]
        print("UE索引:",self.active_UE)

        # print(action)
        # action = random.randint(0, 169)
        # RBG_index = action // 10
        # UE_index = action - RBG_index * 10
        # print(action)
        UE_index, RBG_index=action # It's the Inner index.
        print("X变化:",1,"位置：",(UE_index, RBG_index))
        self.last_X[UE_index, RBG_index] += 1

        mask = self.last_X[:, RBG_index] > 0
        mask = (1 - mask) * 1e18

        Q=self.last_P.copy()
        self.last_P[:, RBG_index] = self.softmax(self.last_X[:, RBG_index] - mask)
        dif=self.last_P-Q
        Res = get_nonzero_elements_with_indices(dif)
        print("P变化和位置:",Res,)
        #compute MCS and Sinr

        ##mask tiny
        # none = np.where((0 < self.last_P) & (self.last_P <= 0.05))[0]
        # row_indices = np.array(np.where((0 < self.last_P) & (self.last_P <= 0.1))[0])
        # col_indices = np.array(np.where((0 < self.last_P) & (self.last_P <= 0.1))[1])
        #
        # if len(none) > 0:
        #
        #     self.last_P[row_indices,col_indices]=0
        #     col_sum=np.sum(self.last_P,axis=0)
        #     col_sum = np.where(col_sum == 0, 1, col_sum)
        #
        #     self.last_P=self.last_P/col_sum

        print("上次mcs:", self.last_mcs)
        SINR, MCS = compute_sinr_loop(self.last_P, self.act_H, self.act_V)

        ave_mcs, ave_count, ave_sinr = getBaselineMcsSinr(MCS, SINR)
        print("当前mcs:",ave_mcs)
        ave_sinr[np.where(ave_sinr > 28)] = 28  # fix
        self.last_mcs=ave_mcs

        rate = getSendRate(ave_mcs, ave_count)
        now_rate = np.minimum(self.B_N[self.active_UE], rate)
        self.cur_B=np.expand_dims(self.scale_normalize_vector(self.B_N[self.active_UE]-now_rate), axis=1)

        reward_Rate=np.sum(now_rate)-self.last_rate
        self.last_rate=np.sum(now_rate)

        print("上一次:",self.last_satis)
        now_satis=np.dot(now_rate, self.vip_vector)
        print("现在:", now_satis)
        reward_satis=(now_satis-self.last_satis)
        self.last_satis=now_satis

        now_edge=np.dot(now_rate, self.edge_vector)
        reward_edge = (now_edge - self.last_edge)
        self.last_edge=now_edge

        # REWARD=np.dot(self.N_w,[reward_Rate,reward_satis,reward_edge])

        # REWARD = reward_Rate
        REWARD=reward_Rate+reward_satis+reward_edge
        # REWARD=reward_satis
        REWARD=REWARD/10000

        if reward_Rate==0:
            self.state_state+=1
        else:
            self.state_state =0


        # if REWARD==0:
        #     REWARD = -1000

        print("--------","TTI:",self.slot," |inner time:",self.inner_slot,"--------")
        print("Rate:",reward_Rate," |Satisfaction:",reward_satis," |Edge:",reward_edge," |all:",REWARD)
        print("LENS:",len(self.active_UE)," | VIP:",np.sum(self.vip_vector)," | Edge:",np.sum(self.edge_vector))

        self.inner_slot+=1
        self.state = np.hstack((
            self.normalize_vector(self.state_H),
            self.scale_normalize_vector(self.cur_B),
            self.scale_normalize_vector(self.state_B),
            self.W,
            self.Class,
            self.last_X
        ))

        info=None
        if self.inner_slot == self.timesteps:
            done = True
            objs = [reward_Rate,reward_satis,reward_edge,REWARD]
            info = objs
        else:
            done = False



        return self.state,REWARD,done,info
    def getState(self):
        state_H = self.H[self.active_UE][:,0,:]
        real_part = np.real(state_H)
        imaginary_part = np.imag(state_H)
        self.state_H=np.hstack((real_part, imaginary_part))

        num=len(self.active_UE)
        self.state_B=np.expand_dims(self.B_N[self.active_UE],axis=1)

        self.W=np.array([self.preference]*num)
        self.Class=np.zeros((len(self.active_UE)))
        for i in range(len(self.Class)):
            if self.active_UE[i] in self.Vip_index:
                self.Class[i]=1
            if self.active_UE[i] in self.edge_index:
                self.Class[i]=-1
        self.Class = np.expand_dims(self.Class, axis=1)
        self.cur_B=self.normalize_vector(self.state_B).copy()
        self.state=np.hstack((
            self.normalize_vector(self.state_H),
            self.scale_normalize_vector(self.state_B),
            self.W,
            self.Class,
            self.last_X,
            self.cur_B
                              ))

        self.last_rate =0
        self.last_edge=0
        self.last_satis=0
        self.inner_slot = 0
        self.state_state = 0

        self.vip_vector = np.zeros(len(self.active_UE))
        VIP_inner = np.where(self.Class.squeeze(1) == 1)
        self.vip_vector[VIP_inner] = 1

        self.edge_vector = np.zeros(len(self.active_UE))
        edge_inner = np.where(self.Class.squeeze(1) == -1)
        self.edge_vector[edge_inner] = 1

        #W normalization, Cause  somtimes, no vip,no edge UEs.
        self.N_w=self.preference.copy()
        if np.sum(self.vip_vector)==0:
            self.N_w[1]=0
        if np.sum(self.edge_vector)==0:
            self.N_w[2]=0
        self.N_w=self.N_w/np.sum(self.N_w)

        return self.state

    def update(self,P=None,dec=None):
        vip_finish=False
        edge_finish=False
        vip_choice=None
        edge_choice=None

        print("待处理用户:",self.active_UE)
        self.act_H = self.H[self.active_UE]
        self.act_V = self.V[self.active_UE]
        self.act_Phi =self.Phi[self.active_UE]

        if dec is not  None:
            AVE_MCS,AVE_Count=dec

        # SINR,MCS=compute_sinr_loop(P,self.act_H,self.act_V,Phi=self.act_Phi)
        SINR, MCS = compute_sinr_loop(P, self.act_H, self.act_V)


        ave_mcs, ave_count, ave_sinr=getBaselineMcsSinr(MCS,SINR)

        ave_sinr[np.where(ave_sinr>28)]=28 #fix

        event = self.error.Error_cal(pre_MCS=ave_mcs, req_SINR=ave_sinr)
        rate =getSendRate(ave_mcs, ave_count)
        actual_rate = np.minimum(self.B_N[self.active_UE], rate)

        real_rate = getCorrectedData(event, actual_rate)

        self.B_N[self.active_UE]=self.B_N[self.active_UE]-real_rate

        self.VIP_rate_record[self.active_UE] += real_rate


        #reomve finished UE
        in_indices = np.where(self.B_N[self.active_UE]==0)[0]
        out_indices=np.array(self.active_UE)[in_indices]
        print("完成用户：",out_indices)
        if len(out_indices)!=0:
            vip_check=common_elements(out_indices,self.Vip_index)

            if vip_check is not None:
                vip_finish=True

            edge_check=common_elements(out_indices,self.edge_index)
            if edge_check is not None:
                edge_finish=True


        self.epf_param[out_indices]=0

        # print("finished:",out_indices)
        self.active_UE = remove_elements(self.active_UE, out_indices)
        # print("removed:",self.active_UE)
        print("一次发送剩余活跃用户：",self.active_UE)
        if vip_finish:
            vip_choice=random.choice(self.Vip_index)
            self.active_UE.extend([vip_choice])
        if edge_finish:
            edge_choice=random.choice(self.edge_index)
            self.active_UE.extend([edge_choice])

        rest_UE=remove_elements([i for i in range(self.N)], self.active_UE)

        print("增加特殊用户后，可选用户数：",len(rest_UE))
        print(self.active_UE,111)
        av_set=remove_elements([i for i in range(self.N)], self.active_UE)
        av_set = remove_elements(av_set, self.Vip_index)
        av_set = remove_elements(av_set, self.edge_index)

        # self.active_Delta = sample_vector(self.gamma, self.N,rest_list=rest_UE,time=self.slot)

        self.active_Delta = sample_vector(self.gamma, self.N, rest_list=rest_UE, time=self.slot)
        vector = np.random.choice([0, 1], size=len(av_set), p=[1 - self.gamma, self.gamma])

        mid_term=np.where(vector>0)[0]

        av_set=np.array(av_set)
        delta=av_set[mid_term].tolist()

        print("计划新增用户：",delta)
        if len(self.active_UE + delta) - self.N_max > 0:
            num = len(self.active_UE + delta)- self.N_max
            delta = remove_random_elements(delta, num)
        print("新增用户:", delta)
        self.active_UE.extend(delta)


        # if len(self.active_UE+self.active_Delta)+(vip_finish+edge_finish) - self.N_max > 0:
        #     num = len(self.active_UE+self.active_Delta) - self.N_max
        #     self.active_Delta = remove_random_elements(self.active_Delta, num)
        # print("新增用户:", self.active_Delta)
        # self.active_UE.extend(self.active_Delta)

        self.active_vip = common_elements(self.active_UE,self.Vip_index)#update
        # print("final UE:", self.active_UE)

        self.refresh_data(delta=self.active_Delta)
        # for index in self.active_Delta:
        #     if index in self.Vip_index:
        #         self.B_N[index]=random.gauss(self.mu_B_vip, self.sigma_B_vip)##
        #     else:
        #         self.B_N[index] = random.gauss(self.mu_B, self.sigma_B)##

        self.VIP_times_record[self.active_UE] += 1
        if self.active_vip == None:
            self.active_normal = self.active_UE
        else:
            self.active_normal = remove_elements(self.active_UE, self.active_vip)
            # vip_add_info = [self.Vip_index.index(item) if item in self.Vip_index else -1 for item in self.active_vip]
            # print("当前VIP用户",self.active_vip,vip_add_info,self.active_UE)
            # self.VIP_times_record[vip_add_info] += 1
        # slot
        self.slot += 1
        self.reset_Phi()
        if (self.slot % 80 == 0 and self.slot != 0):
            self.reset_H()
        if (self.slot % 300 == 0 and self.slot != 0):
            self.preference = generate_random_vector()



        ##Prb
        col_sum = np.sum(P == 0, axis=0)
        col_zero=np.where(col_sum==0)[0]
        if len(col_zero) == 0:
            Prb=1
        else:
            Prb=1-len(col_zero)/self.M

        ##UE satisfaction
        # event
        actual_su=np.where(ave_mcs>-1)
        error_rate=np.sum(event[actual_su])/len(event[actual_su])
        if len(event[actual_su])==0:
            error_rate=0
        # MCS
        data=[ave_mcs,ave_sinr,Prb,error_rate]

        return real_rate,data

    def softmax(self,f):
        # 坏的实现: 数值问题
        return np.exp(f) / np.sum(np.exp(f))

    def scale_normalize_vector(self,vector,norms=None):
        # 转换为 NumPy 数组，便于向量化计算
        vector = np.array(vector)
        min_val = 10000
        max_val = 50000

        if norms is not  None:
            min_val,max_val=norms

        # 防止最大值和最小值相同，避免除以0
        if max_val == min_val:
            return np.zeros_like(vector)

        # 归一化计算公式：(x - min) / (max - min)
        normalized = (vector - min_val) / (max_val - min_val)
        return normalized
    def normalize_vector(self,vector):
        # 转换为 NumPy 数组，便于向量化计算
        vector = np.array(vector)
        min_val = np.min(vector)
        max_val = np.max(vector)

        # 防止最大值和最小值相同，避免除以0
        if max_val == min_val:
            return np.zeros_like(vector)

        # 归一化计算公式：(x - min) / (max - min)
        normalized = (vector - min_val) / (max_val - min_val)
        return normalized




#set a solution

# if __name__=="__main__":
#     Env=WirelessNetwork()
#     for time in range(100):
#
#         UE_num=len(Env.active_UE)
#         X = np.random.binomial(n=1, p=0.1, size=(UE_num, 17))
#
#         X = normalize_columns(X)
#         rate = Env.update(X)
#         print(rate, UE_num,Env.active_UE)


