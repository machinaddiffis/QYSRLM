import copy
import functools
import itertools
import torch
import math

import numpy as np
from matplotlib import pyplot as plt

import time
from error_event import Mcs_Sinr_p as Mcs_Sinr_p
'''
User distance is uniform. 0~9
'''

# SINR_TABLE = [   -6.55,  -4.51,  -2.8,   -0.84,  0.98,   2.63,   4.69,   5.59,   6.53,   7.5,
#                 8.38,   8.92,   10.26,  11.11,  12.03,  12.91,  13.94,  14.96,  15.91,  16.93,
#                 18.02,  18.93,  19.56,  20.46,  21.45,  22.32,  23.74,  24.54,  25.43 ]

MCS_TABLE = [   0.1523, 0.2344, 0.377,  0.6016, 0.877,  1.1758, 1.4766, 1.6953, 1.9141, 2.1602,
                2.4063, 2.5703, 2.7305, 3.0293, 3.3223, 3.6094, 3.9023, 4.2129, 4.5234, 4.8164,
                5.1152, 5.332,  5.5547, 5.8906, 6.2266, 6.5703, 6.9141, 7.1602, 7.4063]

# SINR_TABLE = [0.31126, 0.09484, 0.13312, 0.14236, 0.15482, 0.12404, 0.13868, 0.1494, 0.14626, 0.17662,
#             0.17454, 0.1531, 0.12472, 0.13966, 0.1322, 0.13974, 0.13216, 0.12616, 0.1426, 0.15916,
#             0.16594, 0.18384, 0.25542, 0.29776, 0.11504, 0.14404, 0.289, 0.11964, 0.24792, 0.10134,
#             0.29456, 0.13976, 0.0438, 0.01042, 0.00206, 0.0003, 0.00018]
SINR_TABLE = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 11, 12, 13, 14, 15, 16, 18, 19, 20,
                21, 22, 23, 24, 24, 25, 26, 26, 27, 27,
                28, 28, 28, 28, 28, 28, 28]

class RRMSimulator:
    def __init__(self, CellNum=1, RBGsNum=17, UserNum=30, nT=32, nR=1, initial_seed = 1, seed_gap=10):
        self.error_event_generator = Mcs_Sinr_p()
        self.error_event_generator.load_table()
        self.error_event_generator.create_inte()
        self.RBGsNum = RBGsNum
        self.CellNum = CellNum
        self.UserNum = UserNum
        self.w = np.ones([CellNum, UserNum]) #用户权重
        self.nT = nT
        self.nR = nR
        self.all_HV_2 = None
        self.sigma2 = 10 ** -13
        self.seed = initial_seed
        self.seed_gap = seed_gap
        np.random.seed(self.seed)
        self.updateHV()

    def getPhi(self):
        return np.random.rayleigh(1e-12, size=(self.UserNum, self.RBGsNum))

    def getSingleSinrChangedPhi(self, V, P, I, H, all_HV_2=None):
        P = np.expand_dims(P, axis=0)
        I = I.squeeze()
        I = np.expand_dims(I, axis=0)
        assert self.nR == 1, "only support nR==1"
        mcs = np.zeros([self.CellNum, self.UserNum, self.RBGsNum]) + -1
        sinr_matrix = np.zeros([self.CellNum, self.UserNum, self.RBGsNum]) + -1000
        for b in range(self.CellNum):
            for r in range(self.RBGsNum):
                # if r not in calculate_list:
                #     continue
                for k in range(self.UserNum):
                    if P[b, k, r] == 0:
                        continue
                    Temp = 0.0
                    for b1 in range(self.CellNum):
                        for k1 in range(self.UserNum):
                            # 这里计算的是小区内 + 小区间的干扰 , 针对 r 号 RBG
                            # Temp += np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2 * p[b1, k1, r]
                            Temp += all_HV_2[b, k, b1, k1] * P[b1, k1, r]
                    # cur_signal = np.linalg.norm(np.dot(H[b, b, k],V[b, k])) ** 2 * p[b, k, r]
                    cur_signal = all_HV_2[b, k, b, k] * P[b, k, r]
                    sinr = cur_signal /( Temp - cur_signal + self.sigma2 + I[b, k, r])

                    # get single mcs
                    if sinr == 0:
                        sinr_matrix[b, k, r] = -1000
                    else:
                        sinr_matrix[b, k, r] = sinr
        ave_count = np.zeros([self.CellNum, self.UserNum])
        ave_sinr = np.zeros([self.CellNum, self.UserNum])
        for b in range(self.CellNum):
            for r in range(self.RBGsNum):
                for k in range(self.UserNum):
                    if sinr_matrix[b, k, r] < -10:
                        continue
                    else:
                        ave_sinr[b, r] += sinr_matrix[b, k, r]
                        ave_count[b, r] += 1

        # replace ave_mcs, ave_sinr
        for b in range(self.CellNum):
            for r in range(self.RBGsNum):
                ave_sinr[b, r] = 10 * np.log(ave_sinr[b, r] / ave_count[b, r])
        return ave_sinr


    def _getH(self, CellNum, UserNum, nT, nR):
        AllEsN0 = np.random.uniform(low=20, high=30, size=[CellNum, UserNum]) #与距离的多少次方成反比的量，大尺度衰落；
        # AllEsN0 = np.array([[10.4406752,25.75946832, 50.1381688 , 17.24415915],[11.18273997, 12.29470565, 11.87936056, 54.58865004],[58.18313803, 19.17207594, 19.5862519 , 16.44474599]],dtype=np.float)
        AllEsN0 = 10**(AllEsN0/10) #转化为线性值
        AllEcN0 = np.random.uniform(low=0, high=5, size=[CellNum*(CellNum-1)*UserNum,1]) #其他基站对自己的干扰，大尺度衰落；小区1用户、小区2用户、小区3用户
        AllEcN0 = 10**(AllEcN0/10)
        AllHc = np.sqrt(0.5)*(np.random.randn(CellNum*CellNum*UserNum, nR, nT)+np.random.randn(CellNum*CellNum*UserNum, nR, nT)*1j)  #小区外用户小尺度衰落
        for i in range(CellNum*CellNum*UserNum):
            AllHc[i] = AllHc[i]/np.linalg.norm(AllHc[i])#除以范数 归一化

        # AllH = np.sqrt(0.5)*(np.random.randn(N,1)+np.random.randn(N,1)*1j) # 小区内用户小尺度衰落(1,2)(1,3)(2,1)(2,3)(3,1)(3,2)
        # for i in range(N):
        #     AllH[i] = AllH[i]/np.linalg.norm(AllH[i])

        H = np.zeros((CellNum, CellNum, UserNum, nR, nT),dtype = complex) #CellNum基站序号；CellNum, UserNum共同表示用户序号(某个小区的某个用户)；nR：接收天线数；nT:发送天线数
        k = 0
        k1 = 0
        for i in range(CellNum):
            for j in range(CellNum):
                for jj in range(UserNum):
                    if i == j:
                        H[i][j][jj] = AllHc[k] * np.sqrt(AllEsN0[j][jj]) #本小区信道
                        k += 1
                    else:
                        H[i][j][jj] = AllHc[k] * np.sqrt(AllEcN0[k1])    #邻区信道 物理含义是啥？
                        k += 1
                        k1 += 1

        self.sigma2 = 10**-13 #噪声功率=-100dBm ? -> 底噪 -130dB
        H = H*np.sqrt(self.sigma2)
        return H

    #******************************计算V矩阵********************************
    def _getV(self, H):
        self.seed += self.seed_gap
        np.random.seed(self.seed)
        p = np.ones([self.CellNum, self.UserNum])/self.UserNum
        noise = 1e-13
        V = np.zeros([self.CellNum, self.UserNum, self.nT, self.nR], dtype = np.complex128)
        HTemp = np.zeros([self.UserNum, self.nT], dtype = np.complex128)
        VTemp = np.zeros([self.nT, self.UserNum], dtype = np.complex128)
        for b in range(self.CellNum):
            HTemp = np.zeros([self.UserNum, self.nT], dtype = np.complex128)
            for k in range(self.UserNum):
                HTemp[k, :] = H[b, b, k, 0, :]
            HT = np.conjugate(HTemp).T
            GB = np.dot(HTemp, HT) + noise
            VTemp = np.dot(HT, np.linalg.pinv(GB))
            V[b, :, :, 0] = VTemp.T
        V = self.ColNorm(V)
        return V

    def ColNorm(self,V):
        V1 = copy.deepcopy(V)
        TempPower = 0.0
        for b in range(self.CellNum):
            for UserIdx in range(self.UserNum):
                TempPower = np.linalg.norm(V1[b, UserIdx, :, :])**2 #求二范数
                V1[b, UserIdx, :, :] = V1[b, UserIdx, :, :]/np.sqrt(TempPower)
        return V1


    def getHV(self):
        H = self._getH(self.CellNum, self.UserNum, self.nT, self.nR)
        V = self._getV(H)
        all_HV_2 = self._calculate_all_AllHV_2(H, V)
        return H, V, all_HV_2
    ############################################################### multi base station
    '''
        output: 
            H: [UserNum, nR, nT]
            V: [UserNum, nT, nR]
            all_HV_2: [cell, usernum, cell, usernum]
            
    '''
    def getHV_single_cell(self, cell_id):
        assert cell_id <= self.CellNum, ""
        H = self.H[cell_id, cell_id]
        V = self.V[cell_id]
        return H, V, self.all_HV_2

    def getHV_v2(self):
        return self.H, self.V, self.all_HV_2

    '''
        output:
            None 
        update:
            H: [cell, cell, usernum, nR, nT]
            V: [cell, UserNum, nT, nR]
            all_HV_2: [cell, usernum, cell, usernum]
    '''

    def updateHV(self):
        self.H = self._getH(self.CellNum, self.UserNum, self.nT, self.nR)
        self.V = self._getV(self.H)
        self.all_HV_2 = self._calculate_all_AllHV_2(self.H, self.V)

    def getAllPhi(self, allP, all_HV_2):
        '''
            Input:
                allH = [cell, cell, usernum, nR, nT]
                allV = [cell, UserNum, nT, nR]
                allP = [cell, UserNum, RBGs]
                all_HV_2 = [cell, UserNum, cell, UserNum]
            Output:
                I = [Cell, UserNum, RBGs]

        '''
        assert self.nR == 1, "only support nR==1"
        AllPhi = np.zeros([self.CellNum, self.UserNum, self.RBGsNum])
        for b in range(self.CellNum):
            for r in range(self.RBGsNum):
                for k in range(self.UserNum):
                    phi = 0.0
                    for b1 in range(self.CellNum):
                        if b1 == b:
                            continue
                        for k1 in range(self.UserNum):
                            # 这里计算的是小区内 + 小区间的干扰 , 针对 r 号 RBG
                            # Temp += np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2 * p[b1, k1, r]
                            phi += all_HV_2[b, k, b1, k1] * allP[b1, k1, r]
                    AllPhi[b, k, r] = phi
        return AllPhi
    
    def getMcsSinrSingleCell(self, P, I, H, V, all_HV_2):
        '''
            不考虑小区间的
            Input:
                V: [UserNum, RBGsNum]
                P: [UserNum, RBGsNum]
                H: [UserNum, nR, nT]
                all_HV_2: [cell, UserNum, cell, UserNum]
                I: [UserNum, RBGsNum]
            output:
                ave_mcs: [UserNum]
                ave_sinr: [UserNum]
                ave_count: [UserNum]
        '''
        sinr_matrix = np.zeros([self.UserNum, self.RBGsNum]) - 1000
        for r in range(self.RBGsNum):
            for k in range(self.UserNum):
                if P[k, r] == 0:
                    continue
                Temp = 0.0
                for k1 in range(self.UserNum):
                    Temp += all_HV_2[0, k, 0, k1] * P[k1, r]
                cur_signal = all_HV_2[0, k, 0, k] * P[k, r]
                sinr = cur_signal /( Temp - cur_signal + self.sigma2 + I[k, r])

                if sinr == 0:
                    sinr_matrix[k, r] = -1000
                else:
                    sinr_matrix[k, r] = sinr
        
        # get ave sinr
        ave_sinr = np.zeros([self.UserNum])
        ave_count = np.zeros([self.UserNum])
        for k in range(self.UserNum):
            count = 0
            sinr_all = 0
            for k1 in range(self.UserNum):
                for r in range(self.RBGsNum):
                    sinr = sinr_matrix[k1, r]
                    if sinr != -1000:
                        sinr_all += sinr
                        count += 1
            ave_count[k] = count
            if ave_count[k] != 0:
                ave_sinr[k] = sinr_all / ave_count[k]
            else:
                ave_sinr[k] = -1000
        
        # get ave mcs
        ave_mcs = np.zeros([self.UserNum]) + -1
        for k in range(self.UserNum):
            if ave_count[k] != 0:
                ave_sinr[k] = 10 * np.log(ave_sinr[k])
                ave_mcs[k] = self._findMcsIndex(ave_sinr[k])

        ave_mcs = ave_mcs.reshape(1, *(ave_mcs.shape))
        ave_count = ave_count.reshape(1, *(ave_count.shape))
        ave_sinr = ave_sinr.reshape(1, *(ave_sinr.shape))
        # print(ave_mcs.shape)
        # exit()
        return ave_mcs, ave_count, ave_sinr



    #******************************计算用户SINR*****************************
    def _getMcsSinr(self, V, p, I, H, all_HV_2, calculate_list=None):
        assert H.shape[0] == 1, "Use getMcsSinrSingleCell instead of getMcsSinr"
        # 需要考虑self.nR 的
        '''
            这里修改了一处“错误”，由计算abs 改为了计算二范数 np.linalg.norm
            增加了噪音项
            P:[Cell, UEs, RBGs]
            I:[Cell, UEs, RBGs]
        '''
        assert self.nR == 1, "only support nR==1"
        mcs = np.zeros([1, self.UserNum, self.RBGsNum]) + -1
        sinr_matrix = np.zeros([1, self.UserNum, self.RBGsNum]) + -1000
        for b in range(1):
            for r in range(self.RBGsNum):
                # if r not in calculate_list:
                #     continue
                for k in range(self.UserNum):
                    if p[b, k, r] == 0:
                        continue
                    Temp = 0.0
                    for b1 in range(1):
                        for k1 in range(self.UserNum):
                            # 这里计算的是小区内 + 小区间的干扰 , 针对 r 号 RBG
                            # Temp += np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2 * p[b1, k1, r]
                            Temp += all_HV_2[b, k, b1, k1] * p[b1, k1, r]
                    # cur_signal = np.linalg.norm(np.dot(H[b, b, k],V[b, k])) ** 2 * p[b, k, r]
                    cur_signal = all_HV_2[b, k, b, k] * p[b, k, r]
                    # print(np.min(all_HV_2))
                    # print("-----------------")
                    sinr = cur_signal /( Temp - cur_signal + self.sigma2 + I[b, k, r])

                    # get single mcs
                    if sinr == 0:
                        mcs[b, k, r] = -1
                        sinr_matrix[b, k, r] = -1000
                    else:
                        sinr_matrix[b, k, r] = sinr
                        sinr = 10 * np.log(sinr)
                        mcs[b, k, r] = self._findMcsIndex(sinr)

                        # print(b, k, r, mcs[b, k, r], sinr,cur_signal, Temp)
        ave_mcs, ave_count, ave_sinr = self.getBaselineMcsSinr(mcs, sinr_matrix)

        # replace ave_mcs, ave_sinr
        mcs_size = ave_mcs.shape
        for i in range(mcs_size[0]):
            for j in range(mcs_size[1]):
                # print(10 * np.log(ave_sinr))
                ave_mcs[i, j] = self._findMcsIndex(10 * np.log(ave_sinr[i][j]))
                ave_sinr[i, j] = 10 * np.log(ave_sinr[i, j])
        return ave_mcs, ave_count, ave_sinr

    def _findMcsIndex(self, sinr):
        if math.isnan(sinr):
            return -1
        mcs_index = -1
        sinr_int = int(sinr) + 8
        if sinr_int < 0:
            mcs_index = -1
        elif sinr_int > 36:
            mcs_index = SINR_TABLE[36]
        else:
            mcs_index = SINR_TABLE[sinr_int]

        return mcs_index

    def getBaselineMcsSinr(self, MCS, Sinr):
        ave_mcs = np.zeros([1, self.UserNum])
        ave_sinr = np.zeros([1, self.UserNum])
        ave_count = np.zeros([1, self.UserNum])
        for b in range(1):
            for k in range(self.UserNum):
                mcs_count = 0
                total_mcs = 0
                total_sinr = 0
                for r in range(self.RBGsNum):
                    if MCS[b, k, r] == -1:
                        continue
                    else:
                        total_mcs += MCS[b, k, r]
                        total_sinr += Sinr[b, k, r]
                        mcs_count += 1
                if mcs_count != 0:
                    ave_mcs[b,k] = total_mcs // mcs_count
                    ave_sinr[b,k] = total_sinr / mcs_count
                else:
                    ave_mcs[b,k] = -1
                    ave_sinr[b,k] = -10000
                ave_count[b, k] = mcs_count
        return ave_mcs, ave_count, ave_sinr

    def getMcsSinr(self, P, I, H, V, all_HV_2, calculate_list=None, squeeze=False):
        # HAN更改维度
        if squeeze:
            P = np.expand_dims(P, axis=0)
            I = np.expand_dims(I, axis=0)

        if calculate_list is None:
            calculate_list = [i for i in range(self.RBGsNum)]
        return self._getMcsSinr(V, P, I, H, all_HV_2, calculate_list)

    def _calculate_all_AllHV_2(self, H, V):
        all_HV_2 = np.zeros([self.CellNum, self.UserNum, self.CellNum, self.UserNum])
        for b in range(self.CellNum):
            for k in range(self.UserNum):
                for b1 in range(self.CellNum):
                    for k1 in range(self.UserNum):
                        all_HV_2[b, k, b1, k1] = np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2
        return all_HV_2

    def _getSinrSingleRbg(self, cell_id: int, rbg_id : int, V, p, I, H, all_HV_2):
        # 需要考虑self.nR 的
        '''
            这里修改了一处“错误”，由计算abs 改为了计算二范数 np.linalg.norm
            增加了噪音项
            P:[Cell, UEs, RBGs]
            I:[Cell, UEs, RBGs]
        '''
        assert self.nR == 1, "only support nR==1"
        all_sinr = np.zeros([self.UserNum]) + -1000
        for b in [cell_id]:
            for r in [rbg_id]:
                for k in range(self.UserNum):
                    if p[b, k, r] == 0:
                        continue
                    Temp = 0.0
                    # 干扰的计算要计算所有的
                    for b1 in range(self.CellNum):
                        for k1 in range(self.UserNum):
                            # 这里计算的是小区内 + 小区间的干扰 , 针对 r 号 RBG
                            Temp += all_HV_2[b, k, b1, k1] * p[b1, k1, r]
                            # Temp += np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2 * p[b1, k1, r]
                    
                    # cur_signal = np.linalg.norm(np.dot(H[b, b, k],V[b, k])) ** 2 * p[b, k, r]
                    cur_signal = all_HV_2[b, k, b, k] * p[b, k, r]
                    sinr = cur_signal /( Temp - cur_signal + self.sigma2 + I[b, k, r])

                    if sinr == 0:
                        all_sinr[k] = -1000
                    else:
                        all_sinr[k] = sinr
        return all_sinr

    def getMcsSinrSingleRbg(self, cell_id, rbg_id, P, I, H, V=None, squeeze=False, all_HV_2 = None):
        # HV 其实是不使用的，使用HV，直接使用内部的HV
        # HAN更改维度
        if squeeze:
            P = np.expand_dims(P, axis=0)
            I = np.expand_dims(I, axis=0)
        if V is None:
            V = self._getV(H)
        if all_HV_2 is None:
            all_HV_2 = self._calculate_all_AllHV_2(H, V)
        # HV 其实是不使用的
        return self._getSinrSingleRbg(cell_id, rbg_id, V, P, I, H, all_HV_2)

    def getAveMcsSinr(self, sinr_matrix):
        shape = sinr_matrix.shape
        # if shape[1] != 1:
        #     sinr_matrix[sinr_matrix == -1000] = 0
        #     print(np.sum(sinr_matrix, axis=1))
        # cell
        ave_mcs = np.zeros([shape[0], shape[2]]) - 1
        ave_sinr = np.zeros([shape[0], shape[2]])
        count = np.zeros([shape[0], shape[2]])

        for b in range(shape[0]):
            # RBG
            for r in range(shape[1]):
                # ue 
                for k in range(shape[2]):
                    if sinr_matrix[b, r, k] >= -10:
                        count[b, k] += 1
                        ave_sinr[b, k] += sinr_matrix[b, r, k]
                    # else:
                    #     ave_sinr[b, k] = -1000
        ave_sinr[ave_sinr == 0] = -1000

        for b in range(shape[0]):
            for k in range(shape[2]):
                if count[b, k] > 0:
                    ave_sinr[b, k] = 10 * np.log(ave_sinr[b, k] / count[b, k])
                    if math.isnan(ave_sinr[b, k]):
                        ave_sinr[b, k] = -1000
                    ave_mcs[b, k] = self._findMcsIndex(ave_sinr[b, k])

        return ave_mcs, count, ave_sinr

    def getSendRate(self, ave_mcs, ave_count):
        assert ave_mcs.shape == ave_count.shape, f"{ave_mcs.shape}, {ave_count.shape}"
        try:
            rate = np.zeros([1, self.UserNum])
            for b in range(1):
                for k in range(self.UserNum):
                    if ave_mcs[b, k] != -1:
                        rate[b, k] = int(144 * 16 * MCS_TABLE[ave_mcs[b, k].astype(int)] * ave_count[b, k])
        except:
            print(ave_mcs)
        return rate

    def getSingleSendRate(self,mcs):
        return int(144 * 16 * MCS_TABLE[np.int64(mcs)])

    def getErrorEvent(self, pre_MCS, req_SINR):
        '''
            get Error Event situation
        '''
        assert pre_MCS.shape == req_SINR.shape, f"{pre_MCS.shape}, {req_SINR.shape}"
        return self.error_event_generator.Error_cal(pre_MCS, req_SINR)

    def getCorrectedData(self, error_event, rate):
        return rate * (1 - error_event)

# class MultiRRMSimulator():
#     def __init__(self, CellNum=1, RBGsNum=17, UserNum=30, nT=32, nR=1):
#         self.CellNum = CellNum
#         self.RBGsNum = RBGsNum
#         self.UserNum = UserNum
#         self.nT = nT
#         self.nR = nR
#         self.cell_list = []
#         self.H_list = []
#         self.V_list = []
#         self.HV_list = []

#         for i in range(CellNum):
#             self.cell_list.append(RRMSimulator(1, RBGs, UEs, nT, nR))
#             H, V, all_HV = self.cell_list[-1].getHV()
#             self.H_list.append(H)
#             self.V_list.append(V)
        
#             self.HV_list.append(all_HV)

#     def _get_HV2(self, H_list, V_list):
#         all_HV_2 = np.zeros([self.CellNum, self.UserNum, self.CellNum, self.UserNum])
#         for b in range(self.CellNum):
#             for k in range(self.UserNum):
#                 for b1 in range(self.CellNum):
#                     for k1 in range(self.UserNum):
#                         # all_HV_2[b, k, b1, k1] = np.linalg.norm(np.dot(H[b1, b, k], V[b1, k1])) ** 2
#                         all_HV_2[b, k, b1, k1] = np.linalg.norm(np.dot(H_list[][b1, b, k], V[b1, k1])) ** 2
#         return all_HV_2

#     def getHV_by_cellid(self, cellid, update = False):
#         if update is True:
#             H, V, all_HV = self.cell_list[cellid].getHV()
#             self.H_list[cellid] = H
#             self.V_list[cellid] = V
#         self.HV2 = self._get_HV2()
#         return self.H_list[cellid], self.V_list[cellid], self.HV_list[cellid]

#     def get_real_phi(self, P_list):
#         Phi_list = []
#         H = self.H_list[cell_id]
#         V = self.V_list[cell_id]
#         all_HV = self.HV_list[cell_id]
#         for b in range(self.CellNum):
#             cell_phi = np.zeros(size=(self.UserNum, self.RBGsNum))
#             for r in range(self.RBGsNum):
#                 for k in range(self.UserNum):
#                     # calculate phi
#                     phi = 0.0
#                     for b1 in range(self.CellNum):
#                         if b1 == b:
#                             continue
#                         for k1 in range(self.UserNum):
#                             phi += self.HV_list[b][0,k,b1,]
#                     # if p[b, k, r] == 0:
#                     #     continue
#                     # phi = 0.0
#                     # for b1 in range(self.CellNum):
#                     #     # 不计算内部信号
#                     #     if b1 == cell_id:
#                     #         continue
#                     #     for k1 in range(self.UserNum):

#     def get_mcs_sinr(self, ):


if __name__ == "__main__":
    Cells=4
    RBGs = 17
    UEs=32
    nT=32
    nR=1
    sinr_generator = RRMSimulator(Cells, RBGs, UEs, nT, nR)
    power = torch.ones((Cells, UEs, RBGs))/UEs
    interference = torch.zeros((Cells, UEs, RBGs))
    power = power.numpy()
    interference = interference.numpy()
    # 获取 H 矩阵,以及H矩阵对应的 V 矩阵
    H, V, all_HV = sinr_generator.getHV()
    print(H.shape)
    H, V, all_HV = sinr_generator.getHV()
    print(H.shape)
    # # 生成平均 MCS 矩阵, 第一个返回值为每个UE的平均MCS，第二个返回值为每个UE被多少个RBG传输数据
    # # 这里V矩阵是为了节约计算时间，也可以不传入，内部会重新计算V矩阵
    # start = time.time()
    # ave_mcs, ave_count, ave_sinr = sinr_generator.getMcsSinr(power, interference, H, V)
    # end = time.time()
    # print(f"All RBG update runtime: {(end-start)*1000}ms")

    # print(ave_sinr, ave_mcs)

    # start = time.time()
    # ave_mcs, ave_count, ave_sinr = sinr_generator.getMcsSinr(power, interference, H, V,[0])
    # end = time.time()
    # print(f"single RBG update runtime: {(end-start)*1000}ms")
