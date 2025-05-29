#代码注释

models.py存储所有的模型;

Utils.py存储所有的处理函数，比如数据集读入读出，信道计算，功能函数，交互函数等;

Data_collection.py使用baseline，生成数据，会自动生成数据集。189行 Trans默认False，会自动生成正常concat在一起的Input；Trans开启，会生成Transformer(见最后的说明)所需的特殊数据格式(X1,X2,X3);

OffLine_train.py**监督学习训练**一个端到端的MLP, input 8x19,  output 8x17 (过sigmoid)； 标签为8x17的0，1二元矩阵。loss默认使用MSE(实验测试，训练效果与BCE相同)， 自动存储log和模型； acc的计算方式为，logtis>0.5的设置为1， 与真值对比，计算平均准确率;

Permu_train.py **监督学习训练**一个端到端的MLP, input [8x16,8x3,8x16],  output 8x17 (过sigmoid)； 标签为8x17的0，1二元矩阵。loss默认使用MSE(实验测试，训练效果与BCE相同);

Train_loss_plot.py 可以画log图;

Retrain_eval.py 可以用来评估训练好的模型，会自动生成log; 更改54行来控制模型选择。

python data_plot.py 可视化数据。（需要更改读入文件名，见256行）

python Baseline_scr.py 运行baseline
python Train_Power.py 运行RL4RRM
环境信息写在Params.py
	238-248行，见reward设计。环境更新见update方法



## # **置换等变Transformer设计**：
### ###输入

问题输入：需求$B\in \mathbb{Z}_{\geq0}^N$,  信道$H\in \mathbb{C}^{ N \times \text{nT}}$ 

网络输入

- 将信道信息拆分为实部与虚部：

  $X_1 = [\mathcal{Re}(H), \mathcal{Im}(H)] \in \mathbb{R}^{ N \times \text{2nT}}$

- 将VIP, edge用户信息和需求信息combine在一起：
  $X_2 = [B,c] \in \mathbb{Z}^{N\times 2}$ 

  其中$c_i=0,1,2$分别表示用户为普通用户、VIP用户、edge用户

- 提取信道的相关性，然后拆分实部与虚部：

  $X_3 = [\mathcal{Re}\left(HH^{\text{H}}\right), \mathcal{Im}(HH^{\text{H}})]\in \mathbb{R}^{ N \times \text{2N}}$

(8, 16) (8, 2) (8, 16)
###第一层

#### Type-A自注意力头

设计$m_1$个Type-A自注意力头

输入全量信息：

- $$X_A = [X_1,X_2]\in \mathbb{R}^{ N \times (\text{2nT+2})}$$ 

计算Q，K，V头：

- $Q_{1,i}=X_A W^Q_{1,i}$，其中$W_{1,i}^Q \in \mathbb{R}^{(2\text{nT}+2)\times d^K_1}$， $i=1,\cdots, m_1$
- $K_{1,i}=X_A W^K_{1,i}$，其中$W_{1,i}^K \in \mathbb{R}^{(2\text{nT}+2)\times d^K_1}$， $i=1,\cdots, m_1$
- $V_{1,i}=X_A W^V_{1,i}$，其中$W_{1,i}^V \in \mathbb{R}^{(2\text{nT}+2)\times d^V_1}$， $i=1,\cdots, m_1$

计算attention score与头输出

- $S_{1,i}=\text{softmax}\left(\frac{Q_{1,i}K_{1,i}^T}{\sqrt{d^K_1}}\right)$
- $Z_{1,i}=S_{1,i}V_{1,i}$，$i=1,\cdots,m_1$

其中softmax按列进行归一化

#### Type-B自注意力头（旋转不变特征）

设计$m_2$个Type-B自注意力头

对$X_3$用element-wise sigmoid函数，将数值归一化到$(0,1)$之间

- $X_B= \text{sigmoid}(X_3)\in \mathbb{R}^{N\times 2N}$

只需要计算V头：

- $V_{1,i}=X_2 W^V_{1,i}$，其中$W_{1,i}^V \in \mathbb{R}^{2\times d^V_2}$， $i=m_1+1,\cdots, m_1+m_2$

注意V头的输入是$X_2 = [B,c] \in \mathbb{Z}^{N\times 2}$ 

计算头输出：

- $Z_{1,i}=X_BV_{1,i}$， $i=m_1+1,\cdots, m_1+m_2$

#### 头拼接与输出投影

所有头在特征维度上进行级联：

$\tilde{Z}_1 = \text{Concat}^{m_1+m_2}_{i=1}[Z_{1,i}]\in \mathbb{R}^{N\times (m_1d_1^V+m_2d_2^V)}$

投影：

$Z_1 = \tilde{Z}_1 W^O_{1}$，其中$W_1^O\in\mathbb{R}^{(m_1d_1^V + m_2d_2^V)\times M}$

第一层可以考虑归一化或不归一化，可能影响不大

####MLP与剩余层

MLP层以及后续第$l$层Transformers（$l\geq 2$）以$Z_1\in \mathbb{R}^{N\times M}$为输入，按照标准搭建即可
