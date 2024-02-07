# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

ViT

models file

@author: tadahaya
"""

import math
import torch
import torch.nn as nn

class NewGELUActivation(nn.Module):
    """
    Google BERTで用いられているGELUを借用
    https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    https://arxiv.org/abs/1606.08415
    
    """
    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    

class PatchEmbeddings(nn.Module):
    """
    入力画像をパッチへと変換して埋め込む
    
    """
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # patch数の計算, 正方形を仮定
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # projection layer
        # patchをvectorへ変換
        self.projection = nn.Conv2d(
            self.num_channels, self.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size
        )

    def forward(self, x):
        """
        (batch_size, num_channels, image_size, image_size)
        -> (batch_size, num_patches, hidden_size)

        """
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings

    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # position embeddings, CLS token分lengthを追加
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"]) # dropoutも入れてる

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # cls tokenをbatch size分に増やす
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # inputとconcatする
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    """ a single attention head """
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        # Q, K, Vのprojection layers. ここはbiasの有無を選択できるようにしている
        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        inputをQ, K, Vにprojectする
        同じinputをQ, K, V作成に用いるself-attention
        (batch_size, seq_length, hidden_size)
        -> (batch_size, seq_length, attention_head_size)

        """
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # attentionの計算: softmax(Q*K.T/sqrt(head_size))*V
        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        probs = nn.functional.softmax(scores, dim=-1)
        probs = self.dropout(probs) # dropoutかけてる
        output = torch.matmul(probs, v)
        return (output, probs)


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module
    
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # hidden_sizeはnum_attention_headsで割り切れるように
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # q, k, vにてbiasを使うかどうか
        self.qkv_bias = config["qkv_bias"]
        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        # attentionの出力をhidden sizeに戻すためのprojection layer
        # 基本的にall_head_size = hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])
    
    # 240208ここまで