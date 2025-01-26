##############################################################
# (1) ライブラリ・デバイス設定
##############################################################
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################################
# (2) 粒子IDマッピング
##############################################################
PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"

# 粒子名一覧
daughter_tokens = [
    "ph",
    "ne", "ne~", "nm", "nm~",
    "e-", "e+",
    "mu-", "mu+",
    "pi0", "pi-", "pi+",
    "K-", "K+",
    "p", "p~", "n", "n~",
    "La", "La~",
    "Si+", "Si+~", "Si0", "Si0~", "Si-", "Si-~",
    "Xi0", "Xi0~", "Xi-", "Xi+~",
    "Om-", "Om+~",
]

# 母粒子用に、頭に"M"を付けたバージョンの粒子名を作る
mother_tokens = ["M" + t for t in daughter_tokens]

# <PAD>と<EOS>を追加
all_tokens = [PAD_TOKEN, EOS_TOKEN] + daughter_tokens + mother_tokens

# トークン→ID, ID→トークン の変換辞書
particle_vocab = {token: idx for idx, token in enumerate(all_tokens)}
id_to_particle = {idx: token for token, idx in particle_vocab.items()}

vocab_size = len(all_tokens)

##############################################################
# (3) 実験データ(崩壊系列)の定義 + 頻度分だけ複製
##############################################################
# 指定された全崩壊モードを (系列, 頻度) 形式で定義
# 系列は文字列粒子名のリストにし、最後に <EOS> を含む
# 頻度だけ同じ系列を重複させることでデータを作る

decay_data_with_freq = [
    # Om-, Om+~
    (["MOm-", "La", "K-", "<EOS>"], 6727),
    (["MOm-", "Xi0", "pi-", "<EOS>"], 2414),
    (["MOm-", "Xi-", "pi0", "<EOS>"], 849),
    (["MOm-", "Xi-", "pi+", "pi-", "<EOS>"], 2),
    (["MOm-", "Xi-", "pi-", "pi+", "<EOS>"], 2),
    (["MOm-", "Xi0", "e-", "ne~", "<EOS>"], 6),

    (["MOm+~", "La~", "K+", "<EOS>"], 6727),
    (["MOm+~", "Xi0~", "pi+", "<EOS>"], 2414),
    (["MOm+~", "Xi+~", "pi0", "<EOS>"], 849),
    (["MOm+~", "Xi+~", "pi-", "pi+", "<EOS>"], 2),
    (["MOm+~", "Xi+~", "pi+", "pi-", "<EOS>"], 2),
    (["MOm+~", "Xi0~", "e+", "ne", "<EOS>"], 6),

    # Xi0, Xi0~
    (["MXi0", "La", "pi0", "<EOS>"], 9952),
    (["MXi0", "La", "ph", "<EOS>"], 12),
    (["MXi0", "Si0", "ph", "<EOS>"], 33),
    (["MXi0", "Si+", "e-", "ne~", "<EOS>"], 3),
    (["MXi0~", "La~", "pi0", "<EOS>"], 9952),
    (["MXi0~", "La~", "ph", "<EOS>"], 12),
    (["MXi0~", "Si0~", "ph", "<EOS>"], 33),
    (["MXi0~", "Si-~", "e+", "ne", "<EOS>"], 3),

    # Xi-, Xi+~
    (["MXi-", "La", "pi-", "<EOS>"], 9989),
    (["MXi-", "Si-", "ph", "<EOS>"], 1),
    (["MXi-", "La", "e-", "ne~", "<EOS>"], 6),
    (["MXi-", "La", "mu-", "nm~", "<EOS>"], 3),
    (["MXi-", "Si0", "e-", "ne~", "<EOS>"], 1),

    (["MXi+~", "La~", "pi+", "<EOS>"], 9989),
    (["MXi+~", "Si+~", "ph", "<EOS>"], 1),
    (["MXi+~", "La~", "e+", "ne", "<EOS>"], 6),
    (["MXi+~", "La~", "mu+", "nm", "<EOS>"], 3),
    (["MXi+~", "Si0~", "e+", "ne", "<EOS>"], 1),

    # Si-, Si+~
    (["MSi-", "n", "pi-", "<EOS>"], 9980),
    (["MSi-", "n", "pi-", "ph", "<EOS>"], 5),
    (["MSi-", "n", "e-", "ne~", "<EOS>"], 10),
    (["MSi-", "n", "mu-", "nm~", "<EOS>"], 4),
    (["MSi-", "La", "e-", "ne~", "<EOS>"], 1),

    (["MSi+~", "n~", "pi+", "<EOS>"], 9980),
    (["MSi+~", "n~", "pi+", "ph", "<EOS>"], 5),
    (["MSi+~", "n~", "e+", "ne", "<EOS>"], 10),
    (["MSi+~", "n~", "mu+", "nm", "<EOS>"], 4),
    (["MSi+~", "La~", "e+", "ne", "<EOS>"], 1),

    # Si0, Si0~
    (["MSi0", "La", "ph", "<EOS>"], 9950),
    (["MSi0", "La", "e-", "e+", "<EOS>"], 25),
    (["MSi0", "La", "e+", "e-", "<EOS>"], 25),

    (["MSi0~", "La~", "ph", "<EOS>"], 9950),
    (["MSi0~", "La~", "e+", "e-", "<EOS>"], 25),
    (["MSi0~", "La~", "e-", "e+", "<EOS>"], 25),

    # Si+, Si-~
    (["MSi+", "p", "pi0", "<EOS>"], 5145),
    (["MSi+", "n", "pi+", "<EOS>"], 4841),
    (["MSi+", "p", "ph", "<EOS>"], 10),
    (["MSi+", "n", "pi+", "ph", "<EOS>"], 4),

    (["MSi-~", "p~", "pi0", "<EOS>"], 5145),
    (["MSi-~", "n~", "pi-", "<EOS>"], 4841),
    (["MSi-~", "p~", "ph", "<EOS>"], 10),
    (["MSi-~", "n~", "pi-", "ph", "<EOS>"], 4),

    # La, La~
    (["MLa", "p", "pi-", "<EOS>"], 6393),
    (["MLa", "n", "pi0", "<EOS>"], 3580),
    (["MLa", "n", "ph", "<EOS>"], 8),
    (["MLa", "p", "pi-", "ph", "<EOS>"], 9),
    (["MLa", "p", "e-", "ne~", "<EOS>"], 8),
    (["MLa", "p", "mu-", "nm~", "<EOS>"], 2),

    (["MLa~", "p~", "pi+", "<EOS>"], 6393),
    (["MLa~", "n~", "pi0", "<EOS>"], 3580),
    (["MLa~", "n~", "ph", "<EOS>"], 8),
    (["MLa~", "p~", "pi+", "ph", "<EOS>"], 9),
    (["MLa~", "p~", "e+", "ne", "<EOS>"], 8),
    (["MLa~", "p~", "mu+", "nm", "<EOS>"], 2),

    # n, n~
    (["Mn", "p", "e-", "ne~", "<EOS>"], 9991),
    (["Mn", "p", "e-", "ne~", "ph", "<EOS>"], 9),
    (["Mn~", "p~", "e+", "ne", "<EOS>"], 9991),
    (["Mn~", "p~", "e+", "ne", "ph", "<EOS>"], 9),

    # p, p~
    (["Mp", "<EOS>"], 100),
    (["Mp~", "<EOS>"], 100),

    # K-, K+
    (["MK-", "mu-", "nm~", "<EOS>"], 6315),
    (["MK-", "pi0", "e-", "ne~", "<EOS>"], 504),
    (["MK-", "pi0", "mu-", "nm~", "<EOS>"], 333),
    (["MK-", "pi-", "pi0", "<EOS>"], 1027),
    (["MK-", "pi0", "pi-", "<EOS>"], 1027),
    (["MK-", "pi-", "pi0", "pi0", "<EOS>"], 58),
    (["MK-", "pi0", "pi-", "pi0", "<EOS>"], 58),
    (["MK-", "pi0", "pi0", "pi-", "<EOS>"], 58),
    (["MK-", "pi-", "pi-", "pi+", "<EOS>"], 185),
    (["MK-", "pi-", "pi+", "pi-", "<EOS>"], 185),
    (["MK-", "pi+", "pi-", "pi-", "<EOS>"], 185),
    (["MK-", "mu-", "nm~", "ph", "<EOS>"], 62),
    (["MK-", "pi0", "e-", "ne~", "ph", "<EOS>"], 3),

    (["MK+", "mu+", "nm", "<EOS>"], 6315),
    (["MK+", "pi0", "e+", "ne", "<EOS>"], 504),
    (["MK+", "pi0", "mu+", "nm", "<EOS>"], 333),
    (["MK+", "pi+", "pi0", "<EOS>"], 1027),
    (["MK+", "pi0", "pi+", "<EOS>"], 1027),
    (["MK+", "pi+", "pi0", "pi0", "<EOS>"], 58),
    (["MK+", "pi0", "pi+", "pi0", "<EOS>"], 58),
    (["MK+", "pi0", "pi0", "pi+", "<EOS>"], 58),
    (["MK+", "pi+", "pi+", "pi-", "<EOS>"], 185),
    (["MK+", "pi+", "pi-", "pi+", "<EOS>"], 185),
    (["MK+", "pi-", "pi+", "pi+", "<EOS>"], 185),
    (["MK+", "mu+", "nm", "ph", "<EOS>"], 62),
    (["MK+", "pi0", "e+", "ne", "ph", "<EOS>"], 3),

    # pi-, pi+, pi0
    (["Mpi-", "mu-", "nm~", "<EOS>"], 9997),
    (["Mpi-", "mu-", "nm~", "ph", "<EOS>"], 2),
    (["Mpi-", "e-", "ne~", "<EOS>"], 1),

    (["Mpi+", "mu+", "nm", "<EOS>"], 9997),
    (["Mpi+", "mu+", "nm", "ph", "<EOS>"], 2),
    (["Mpi+", "e+", "ne", "<EOS>"], 1),

    (["Mpi0", "ph", "ph", "<EOS>"], 9882),
    (["Mpi0", "e-", "e+", "ph", "<EOS>"], 59),
    (["Mpi0", "e+", "e-", "ph", "<EOS>"], 59),

    # mu-, mu+
    (["Mmu-", "e-", "ne~", "nm", "<EOS>"], 100),
    (["Mmu+", "e+", "ne", "nm~", "<EOS>"], 100),

    # e-, e+, ne, ne~, nm, nm~, ph 単独EOS
    (["Me-", "<EOS>"], 100),
    (["Me+", "<EOS>"], 100),
    (["Mne", "<EOS>"], 100),
    (["Mne~", "<EOS>"], 100),
    (["Mnm", "<EOS>"], 100),
    (["Mnm~", "<EOS>"], 100),
    (["Mph", "<EOS>"], 100),
]

# 上記の定義を元に実際の系列リストを作成
def build_dataset(decay_data_list):
    sequences = []
    for seq_tokens, freq in decay_data_list:
        seq_ids = [particle_vocab[t] for t in seq_tokens]
        for _ in range(freq):
            sequences.append(seq_ids)
    return sequences

all_seqs = build_dataset(decay_data_with_freq)

# 訓練用にシャッフル
random.shuffle(all_seqs)

##############################################################
# (4) Dataset / DataLoader
##############################################################
class ParticleDecayDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

def collate_fn(batch, pad_id=particle_vocab[PAD_TOKEN]):
    """
    1) バッチ内の最大長に合わせてパディング
    2) 入力(input)は seq[:-1], 教師(target)は seq[1:]
    """
    max_len = max(len(seq) for seq in batch)
    inputs, targets = [], []

    for seq in batch:
        inp = seq[:-1]
        tgt = seq[1:]
        # パディングで長さを合わせる
        pad_len = max_len - len(seq)
        inp += [pad_id] * pad_len
        tgt += [pad_id] * pad_len
        inputs.append(inp)
        targets.append(tgt)

    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)
    return inputs_tensor, targets_tensor

train_dataset = ParticleDecayDataset(all_seqs)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, collate_fn=collate_fn)

##############################################################
# (5) Transformerベースの言語モデル定義
##############################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, max_len, d_model) にして register_buffer
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerDecayModel(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformerのエンコーダを複数層重ねる
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _generate_subsequent_mask(self, seq_len):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def forward(self, x):
        batch_size, seq_len = x.size()

        # 埋め込み + 位置エンコーディング
        x_emb = self.embed(x)                 # -> [B, L, d_model]
        x_emb = self.pos_encoder(x_emb)       # -> [B, L, d_model]

        # 未来情報を参照しないようにするマスク
        mask = self._generate_subsequent_mask(seq_len).to(x.device)

        # Transformerエンコーダに通す
        hidden = self.transformer_encoder(x_emb, mask=mask)  # [B, L, d_model]

        # 語彙数に対応した出力へ
        logits = self.fc_out(hidden)  # [B, L, vocab_size]
        return logits
    
##############################################################
# (6) 学習ループ
##############################################################
model = TransformerDecayModel(
    vocab_size=vocab_size,
    d_model=8,
    nhead=2,
    num_layers=2,
    dim_feedforward=1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=particle_vocab[PAD_TOKEN])

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # [B, L, vocab_size]
        logits = model(inputs)
        # 損失計算のため2次元へ展開
        logits_2d = logits.view(-1, vocab_size)
        targets_1d = targets.view(-1)

        loss = criterion(logits_2d, targets_1d)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# デモ的に数エポックのみ
for epoch in range(50):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1} - loss: {loss:.4f}")

##############################################################
# (7) 推論(デコード)関数
##############################################################
def generate_sequence(model, start_particle, max_length=6):
    """
    母粒子トークン(start_particle)から始め、自己回帰的に系列を生成。
    途中で<EOS>が出たら終了。
    """
    model.eval()
    start_id = particle_vocab[start_particle]
    generated = [start_id]

    for _ in range(max_length):
        x = torch.tensor([generated], dtype=torch.long).to(device)  # shape: [1, seq_len]

        with torch.no_grad():
            logits = model(x)  # [1, seq_len, vocab_size]

        # 最後のステップの出力だけ取り出し
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=-1).cpu().numpy()

        # 確率に応じてサンプリング（argmaxでも可）
        next_id = np.random.choice(vocab_size, p=probs)
        generated.append(next_id)

        # EOSならば生成終了
        if next_id == particle_vocab[EOS_TOKEN]:
            break
    return [id_to_particle[i] for i in generated]

##############################################################
# (8) 推論テスト
##############################################################
test_mothers = [
    "ph",
    "ne", "ne~", "nm", "nm~",
    "e-", "e+",
    "mu-", "mu+",
    "pi0", "pi-", "pi+",
    "K-", "K+",
    "p", "p~", "n", "n~",
    "La", "La~",
    "Si+", "Si+~", "Si0", "Si0~", "Si-", "Si-~",
    "Xi0", "Xi0~", "Xi-", "Xi+~",
    "Om-", "Om+~"
    ]
print("=== Generation Examples ===")
for mother in test_mothers:
    if mother not in particle_vocab:
        continue
    seq = generate_sequence(model, "M" + mother, max_length=6)
    print(mother, "->", " ".join(seq[1:-1]))