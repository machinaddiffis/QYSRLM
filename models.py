import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=1, num_layers=1):
        super(TransformerEncoder, self).__init__()

        # 输入嵌入层
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # 位置编码
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # 最大序列长度为1000

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        # 输入 x 是一个 N * d 的张量

        # 添加位置编码
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        # 通过自注意力层
        x = x.transpose(0, 1)  # [N, seq_len, hidden_dim] -> [seq_len, N, hidden_dim]
        attn_output, _ = self.attention(x, x, x)

        # 通过前馈网络
        output = self.feed_forward(attn_output.transpose(0, 1))  # [seq_len, N, output_dim] -> [N, seq_len, output_dim]

        return output


class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=17, hidden_dim=32, num_layers=1, num_heads=4, dropout=0.1):
        super(SimpleTransformerEncoder, self).__init__()
        # 将输入投影到隐藏维度
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        # 定义 Transformer Encoder 层（使用 batch_first=True 使得输入形状为 (batch, seq_length, embed_dim)）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 将隐藏维度投影到输出维度 17
        self.output_projection = nn.Linear(hidden_dim, output_dim)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        """
        参数:
          x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        返回:
          输出张量，形状为 (batch_size, seq_length, output_dim)
        """
        x = self.input_projection(x)  # (batch, seq_length, hidden_dim)
        x = self.transformer_encoder(x)  # (batch, seq_length, hidden_dim)
        x = self.output_projection(x)  # (batch, seq_length, output_dim)
        return x

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, prob=0.2):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.dropout = nn.Dropout(p=prob)
        self.affine2 = nn.Linear(hidden_size, 2 * hidden_size)
        self.affine3 = nn.Linear(2 * hidden_size, hidden_size)
        self.affine4 = nn.Linear(hidden_size, num_outputs)

        self.affine5= nn.Linear(hidden_size, hidden_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x1 = self.affine1(x)
        x = self.dropout(x1)
        x = F.relu(x)

        x = self.affine2(x)
        x = F.relu(x)

        x = self.affine3(x)
        x = F.relu(x)

        x5=self.affine5(x+x1)
        action_scores = self.affine4(x5)

        return action_scores


class TransLayer(nn.Module):
    """
    第一层 Transformer，包括 m1 个 Type-A 自注意力头和 m2 个 Type-B 头，
    最后在特征维度上拼接并做一次线性投影。
    """
    def __init__(self,
                 input_dim_A: int,   # 2nT+2
                 input_dim_B: int,   # 2N
                 m1: int,
                 m2: int,
                 d_k1: int,
                 d_v1: int,
                 d_v2: int,
                 M: int,
                 nhead: int = 4):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.d_k1 = d_k1

        # —— Type-A 头的参数 —— #
        # Q, K, V 三组各 m1 个线性映射
        self.W_Q = nn.ModuleList([
            nn.Linear(input_dim_A, d_k1, bias=False)
            for _ in range(m1)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(input_dim_A, d_k1, bias=False)
            for _ in range(m1)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(input_dim_A, d_v1, bias=False)
            for _ in range(m1)
        ])

        # —— Type-B 头的参数 —— #
        # 只对归一化后 X_B 做线性映射（等同于只计算 V 头）
        self.W_V_B = nn.ModuleList([
            nn.Linear(input_dim_A-input_dim_B, d_v2, bias=False)
            for _ in range(m2)
        ])
        self.ali=nn.Linear(input_dim_B, input_dim_B//2, bias=False)
        # —— 输出投影 —— #
        self.W_O = nn.Linear(m1 * d_v1 + m2 * d_v2, M, bias=False)

        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=M,
            nhead=1,
            dim_feedforward=d_v1,
            dropout=0.2,
            activation='relu'
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=M,
            nhead=1,
            dim_feedforward=d_v1,
            dropout=0.2,
            activation='relu'
        )
        # 也可以用 nn.TransformerEncoder 来堆叠，但这里演示手动两层
        self.transformer1 = encoder_layer1
        self.transformer2 = encoder_layer2

        self.saved_log_probs = []
        self.rewards = []

    def forward(self,
                X1: torch.Tensor,   # [N, input_dim_A]
                X2: torch.Tensor,    # [N, 2]  （本文没直接用到）
                X3: torch.Tensor     # [N, input_dim_B]
               ) -> torch.Tensor:
        X_A=torch.concatenate((X1,X2),dim=1)

        N = X_A.size(0)

        # --- 1. Type-A 自注意力 --- #
        Z_A = []
        for i in range(self.m1):
            # 1.1 投影 Q, K, V
            Q = self.W_Q[i](X_A)            # [N, d_k1]
            K = self.W_K[i](X_A)            # [N, d_k1]
            V = self.W_V[i](X_A)            # [N, d_v1]

            # 1.2 计算打分矩阵
            #    scores[j,k] = Q[j]·K[k] / √d_k1
            scores = torch.matmul(Q, K.transpose(0,1)) \
                     / math.sqrt(self.d_k1)    # [N, N]

            # 1.3 按“列”归一化（softmax over dim=0）
            S = F.softmax(scores, dim=0)     # [N, N]

            # 1.4 得到每个头的输出
            Z_A.append(torch.matmul(S, V))   # [N, d_v1]

        # --- 2. Type-B 头（旋转不变特征）--- #
        # 2.1 先对 X3 做 sigmoid
        X_B = torch.sigmoid(X3)              # [N, input_dim_B]

        Z_B = []
        for i in range(self.m2):
            # 2.2 只做一次线性映射，等价于“只计算 V 头”

            z_b = self.W_V_B[i](X2)         # [N, d_v2]
            zz=torch.matmul(self.ali(X_B),z_b)
            Z_B.append(zz)

        # --- 3. 拼接 & 输出投影 --- #
        # 在特征维度上把所有头的输出拼在一起
        Z_cat = torch.cat(Z_A + Z_B, dim=1)  # [N, m1*d_v1 + m2*d_v2]

        # 再做一次线性变换投影到 M 维
        Z1 = self.W_O(Z_cat)                 # [N, M]

        Z_seq = Z1.unsqueeze(1)  # [N, 1, M]
        Z_seq = self.transformer1(Z_seq)  # 第一层
        Z_seq = self.transformer2(Z_seq)  # 第二层
        Z_out = Z_seq.squeeze(1)  # [N, M]


        return Z_out

class FCNN(nn.Module):
    """
    A simple fully-connected neural network with two hidden layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x=self.net(x)
        # return torch.nn.functional.sigmoid(x)
        return x


class TransPermuNet(nn.Module):
    """
    第一层 Transformer，包括 m1 个 Type-A 自注意力头和 m2 个 Type-B 头，
    最后在特征维度上拼接并做一次线性投影。
    """
    def __init__(self,
                 input_dim_A: int,   # 2nT+2
                 input_dim_B: int,   # 2N
                 m1: int,
                 m2: int,
                 d_k1: int,
                 d_v1: int,
                 d_v2: int,
                 M: int,
                 nhead: int = 4):
        super().__init__()
        self.m1 = m1
        self.m2 = m2
        self.d_k1 = d_k1

        # —— Type-A 头的参数 —— #
        # Q, K, V 三组各 m1 个线性映射
        self.W_Q = nn.ModuleList([
            nn.Linear(input_dim_A, d_k1, bias=False)
            for _ in range(m1)
        ])
        self.W_K = nn.ModuleList([
            nn.Linear(input_dim_A, d_k1, bias=False)
            for _ in range(m1)
        ])
        self.W_V = nn.ModuleList([
            nn.Linear(input_dim_A, d_v1, bias=False)
            for _ in range(m1)
        ])

        # —— Type-B 头的参数 —— #
        # 只对归一化后 X_B 做线性映射（等同于只计算 V 头）
        self.W_V_B = nn.ModuleList([
            nn.Linear(input_dim_A-input_dim_B, d_v2, bias=False)
            for _ in range(m2)
        ])
        self.ali=nn.Linear(input_dim_B, input_dim_B//2, bias=False)
        # —— 输出投影 —— #
        self.W_O = nn.Linear(m1 * d_v1 + m2 * d_v2, M, bias=False)

        encoder_layer1 = nn.TransformerEncoderLayer(
            d_model=M,
            nhead=1,
            dim_feedforward=d_v1,
            dropout=0.2,
            activation='relu'
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            d_model=M,
            nhead=1,
            dim_feedforward=d_v1,
            dropout=0.2,
            activation='relu'
        )
        # 也可以用 nn.TransformerEncoder 来堆叠，但这里演示手动两层
        self.transformer1 = encoder_layer1
        self.transformer2 = encoder_layer2

        self.saved_log_probs = []
        self.rewards = []

    def forward(self,
                X
               ) -> torch.Tensor:
        X1,X2,X3=X

        X_A=torch.concatenate((X1,X2),dim=2)

        N = X_A.size(0)

        # --- Type-A 自注意力 --- #
        Z_A = []
        for i in range(self.m1):
            # 投影 Q, K, V
            Q = self.W_Q[i](X_A)            # [N, d_k1]
            K = self.W_K[i](X_A)            # [N, d_k1]
            V = self.W_V[i](X_A)            # [N, d_v1]

            # 计算打分矩阵
            scores = torch.matmul(Q, K.transpose(1, 2)) \
                     / math.sqrt(self.d_k1)
            # 按“列”归一化
            S = F.softmax(scores, dim=2)     # [N, N]


            # 得到每个头的输出
            Z_A.append(torch.matmul(S, V))   # [N, d_v1]

        # --- Type-B 头（旋转不变特征）--- #
        # 先对 X3 做 sigmoid
        X_B = torch.sigmoid(X3)              # [N, input_dim_B]

        Z_B = []
        for i in range(self.m2):
            # 2.2 只做一次线性映射，等价于“只计算 V 头”

            z_b = self.W_V_B[i](X2)         # [N, d_v2]
            zz=torch.matmul(self.ali(X_B),z_b)
            Z_B.append(zz)

        # -拼接 & 输出投影 --- #
        # 在特征维度上把所有头的输出拼在一起
        Z_cat = torch.cat(Z_A + Z_B, dim=2)  # [N, m1*d_v1 + m2*d_v2]

        # 再做一次线性变换投影到 M 维
        Z1 = self.W_O(Z_cat)                 # [N, M]


        Z_seq = Z1.transpose(0, 1) # [N, 1, M]

        Z_seq = self.transformer1(Z_seq)  # 第一层
        Z_seq = self.transformer2(Z_seq)  # 第二层
        Z_out = Z_seq.transpose(0, 1) # [N, M]
        return torch.sigmoid(Z_out)


class FCNN(nn.Module):
    """
    A simple fully-connected neural network with two hidden layers.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x=self.net(x)
        return torch.nn.functional.sigmoid(x)

class ResidualBlock(nn.Module):
    """
    A single residual block: two linear layers with BatchNorm and a skip connection.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        return self.relu(out + x)


class ResidualMLP(nn.Module):
    """
    A fully-connected MLP with several residual blocks.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Stack residual blocks
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        x = self.output_layer(x)

        return torch.nn.functional.sigmoid(x)