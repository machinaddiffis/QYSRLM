import pickle
import numpy as np
# 以二进制读模式打开文件
import pickle
import matplotlib.pyplot as plt
from utlis import compute_B
import warnings
warnings.filterwarnings("ignore")

class CallData:
    def __init__(self, file_path="Baseline_Test.pkl",UE=100,vip_target=30000):
        """
        初始化 BaselineTestData 类，并初始化各项数据列表。

        参数:
            file_path -- pkl 文件的路径，默认为 "Baseline_Test.pkl"
        """
        self.file_path = file_path
        self.all_data = []
        self.ave_mcs = []
        self.ave_sinr = []
        self.Prb = []
        self.error_rate = []
        self.times_record = []
        self.rate_record = []
        self.N=UE
        self.active_UE=[]
        self.vip_target=vip_target



        self.avg_error_rate=None
        self.avg_prb=None


    def load_data(self):
        """
        加载 pkl 文件中的数据，并分别将数据提取到各个列表中。
        """
        with open(self.file_path, "rb") as file:
            self.all_data = pickle.load(file)

        # with open("preference_RL_Test.pkl", "rb") as file:
        #     a = pickle.load(file)
        # for iter in a:
        #     print(iter)

        # 打印数据长度
        print("数据长度:", len(self.all_data))

        self.actual_rate = np.zeros((len(self.all_data),self.N))

        self.rate_sum=np.zeros((self.N))

        # 遍历每个数据项，将对应数据存入相应的列表
        for i in range(len(self.all_data)):
            item=self.all_data[i]
            self.ave_mcs.append(item[0])
            self.ave_sinr.append(item[1])
            self.Prb.append(item[2])
            self.error_rate.append(item[3])
            self.times_record.append(item[4][0])
            self.rate_record.append(item[4][1])
            self.active_UE.append(item[4][2])

            self.actual_rate[i][item[4][2]]=item[4][3]
            self.rate_sum[item[4][2]]+=item[4][3]


    def get_data(self):
        """
        返回所有提取的数据，返回一个字典。
        """
        return {
            "ave_mcs": self.ave_mcs,
            "ave_sinr": self.ave_sinr,
            "Prb": self.Prb,
            "error_rate": self.error_rate,
            "times_record": self.times_record,
            "rate_record": self.rate_record
        }
    def calculat_samples(self):
        ##Prb
        self.avg_prb = np.average(self.Prb)
        print("RB使用率", self.avg_prb)

        ##MCS & SINR
        self.avg_mcs_list=[]
        self.avg_sinr_list=[]
        for l in range(len(self.all_data)):
            this_mcs=self.ave_mcs[l]
            this_mcs=np.average(this_mcs[this_mcs!=-1])

            this_sinr=self.ave_sinr[l]
            this_sinr=np.average(this_sinr[this_sinr!=-1.00000000e+04])
            self.avg_mcs_list.append(this_mcs)
            self.avg_sinr_list.append(this_sinr)

        if not isinstance(self.avg_mcs_list, np.ndarray):
            self.avg_mcs_list = np.array(self.avg_mcs_list)
            self.avg_sinr_list = np.array(self.avg_sinr_list)

        # 使用布尔索引过滤掉 NaN 值
        self.avg_mcs_list = self.avg_mcs_list[~np.isnan(self.avg_mcs_list)]
        self.avg_sinr_list = self.avg_sinr_list[~np.isnan(self.avg_sinr_list)]


        self.cal_mcs=self.cumulative_mean(np.array(self.avg_mcs_list))
        self.part_cal_mcs=compute_B(np.array(self.avg_mcs_list))

        self.cal_sinr = self.cumulative_mean(self.avg_sinr_list)
        self.part_cal_sinr=compute_B(self.avg_sinr_list)

        print("MCS平均：",self.cal_mcs[-1])
        print("SINR平均：", self.cal_sinr[-1])

        ##Rate
        tti_rate_sum=np.sum(self.actual_rate,axis=1)
        self.part_avg_rate=compute_B(tti_rate_sum)
        self.avg_rate=self.cumulative_mean(tti_rate_sum)

        print("总平均传输量：",self.avg_rate[-1])

        ##error rate
        self.avg_error_rate = np.average(self.error_rate)
        print("误码率", self.avg_error_rate)

        ##vip satisfaction
        self.satis=[]
        vip_index=[i for i in range(10)]
        edge_index=[95,96,97,98,99]
        for k in range(len(self.all_data)):
            # print("--------------------")
            times = self.times_record[k]
            # print("相应时长",times)
            time_rate=self.rate_record[k]
            # print("总累计发送量",time_rate)
            vip_rate=(time_rate/times)[vip_index]
            edge_rate=(time_rate/times)[edge_index]

            index=self.indices_greater_than_k(vip_rate,self.vip_target)
            all_vip=np.count_nonzero(np.isfinite(vip_rate))
            # print(vip_rate)
            # print(index)
            # print(all_vip)
            if all_vip==0:
                pass
            else:

                self.satis.append(len(index)/all_vip)
                # print(len(index) / all_vip)

        vip_mean = np.mean(vip_rate)
        vip_variance = np.std(vip_rate)

        edge_mean = np.mean(edge_rate)
        edge_variance = np.std(edge_rate)

        print(f"vip发送平均值：{vip_mean}，vip发送标准差:{vip_variance},edge发送平均值：{edge_mean}，edge发送标准差:{edge_variance}")

        self.avg_satis=self.cumulative_mean(self.satis).copy()
        self.part_avg_satis=compute_B(self.satis).copy()

        print("VIP用户总平均满意度：",self.avg_satis[-1])

        ##edge Rate
        self.edge_rate=[]
        for k in range(len(self.all_data)):
            times = self.times_record[k]
            time_rate=self.rate_record[k]
            ue_rate=time_rate/times
            index=self.get_bottom_5_percent_indices(ue_rate)
            self.edge_rate.append(ue_rate[index][0])
        self.avg_edge_rate =self.cumulative_mean(np.array(self.edge_rate)).copy()
        self.part_avg_edge_rate=compute_B(np.array(self.edge_rate)).copy()

        print("边际用户传输量：",self.avg_edge_rate[-1])


    def cumulative_mean(self,B):
        B = np.array(B)
        return np.cumsum(B) / np.arange(1, len(B) + 1)

    def get_bottom_5_percent_indices(self,vector):
        # 将输入转换为 NumPy 数组
        arr = np.array(vector)

        # 筛选出非 NaN 且非 0 的元素，获取它们的索引
        valid_mask = ~np.isnan(arr) & (arr != 0)
        valid_indices = np.where(valid_mask)[0]

        # 如果没有有效数字，则返回空列表
        if len(valid_indices) == 0:
            return []

        # 对有效数字按从小到大排序，获取对应的原数组索引
        sorted_valid_indices = valid_indices[np.argsort(arr[valid_indices])]

        # 计算有效数字中 5% 个数，至少取 1 个
        count = max(1, int(np.ceil(len(sorted_valid_indices) * 0.05)))

        # 返回排序后最小的 count 个索引
        return sorted_valid_indices[:count]

    def indices_greater_than_k(self,vector, k):
        # 将输入转换为 NumPy 数组
        arr = np.array(vector)
        # 使用 np.isnan 判断非 NaN，并筛选大于 k 的值
        indices = np.where((~np.isnan(arr)) & (arr > k))[0]
        return indices

    def plot_all(self,length=None,part=True):

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        if part:

            avg_rate=self.part_avg_rate
            avg_satis=self.part_avg_satis
            avg_edge=self.part_avg_edge_rate
        else:
            avg_rate = self.avg_rate
            avg_satis = self.satis
            avg_edge = self.edge_rate
        if length is not None:
            avg_rate =avg_rate[:length]
            avg_satis = avg_rate[:length]
            avg_edge = avg_rate[:length]

        # 第一个子图绘制 vector1
        axs[0].plot(range(len(avg_rate)), avg_rate)
        axs[0].set_title("Average Rate of all UEs 1")
        axs[0].set_xlabel("TTI")
        axs[0].set_ylabel("Avg Rate")

        # 第二个子图绘制 vector2
        axs[1].plot(range(len(avg_satis)), avg_satis)
        axs[1].set_title("VIP Satisfaction")
        axs[1].set_xlabel("TTI")
        axs[1].set_ylabel("Percentage")

        # 第三个子图绘制 vector3
        axs[2].plot(range(len(avg_edge)), avg_edge)
        axs[2].set_title("Edge Rate")
        axs[2].set_xlabel("TTI")
        axs[2].set_ylabel("Avg Rate")

        plt.tight_layout()  # 调整子图之间的间距
        plt.show()


# 示例用法
if __name__ == "__main__":
    VIP=60000
    # data_obj = CallData(f"Baseline_Test_{VIP}_right_Phi.pkl",vip_target=VIP)
    # data_obj = CallData(f"RL_Test_new_version_single.pkl", vip_target=VIP)
    data_obj = CallData(f"Pretrain_58000.pkl", vip_target=VIP)
    data_obj.load_data()
    data_obj.calculat_samples()
    data_obj.plot_all()

    print(1.2*(75/(1.85*1.85))+0.23*27-16.2-5.4)

    # # 获取并打印所有数据
    # data = data_obj.get_data()
    # print("ave_mcs:", data["ave_mcs"])
    # print("ave_sinr:", data["ave_sinr"])
    # print("Prb:", data["Prb"])
    # print("error_rate:", data["error_rate"])
    # print("times_record:", data["times_record"])
    # print("rate_record:", data["rate_record"])
