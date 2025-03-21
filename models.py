import torch
import torch.nn as nn
import torch.nn.functional as F


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
        print(x.shape)
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

        return F.softmax(action_scores)