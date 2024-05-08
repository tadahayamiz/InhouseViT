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
        # register config
        default_config = {
            "image_size": 32,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_size": 32,
        }
        self.cfg = {**default_config, **config}
        ## patch数の計算, 正方形を仮定
        self.num_patches = (self.cfg["image_size"] // self.cfg["patch_size"]) ** 2
        # layers
        ## projection layer
        ## patchをvectorへ変換
        self.projection = nn.Conv2d(
            self.cfg["num_channels"], self.cfg["hidden_size"],
            kernel_size=self.cfg["patch_size"], stride=self.cfg["patch_size"]
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
        # register config
        default_config = {
            "image_size": 32,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_size": 32,
            "hidden_dropout_prob": 0.0,
        }
        self.cfg = {**default_config, **config}
        # layers
        self.patch_embeddings = PatchEmbeddings(
            self.cfg["image_size"], self.cfg["patch_size"], self.cfg["num_channels"], self.cfg["hidden_size"]
            )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.cfg["hidden_size"]))
        ## learnable [CLS] token
        ## position embeddings, CLS token分lengthを追加
        self.position_embeddings = \
            nn.Parameter(
                torch.randn(1, self.patch_embeddings.num_patches + 1, self.cfg["hidden_size"])
                )
        self.dropout = nn.Dropout(self.cfg["hidden_dropout_prob"]) # dropoutも入れてる


    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # cls tokenをbatch size分に増やす
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # inputとconcatする
        x = torch.cat((cls_tokens, x), dim=1)
        # position embeddingsを足す
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    """ a single attention head """
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "attention_head_size": 8,
            "attention_probs_dropout_prob": 0.0,
            "bias": True,
        }
        self.cfg = {**default_config, **config}
        ## Q, K, Vのprojection layers. ここはbiasの有無を選択できるようにしている
        # layers
        self.query = nn.Linear(
            self.cfg["hidden_size"], self.cfg["attention_head_size"], bias=self.cfg["bias"]
            )
        ## hiddenをattention_head_sizeに押し込む
        self.key = nn.Linear(
            self.cfg["hidden_size"], self.cfg["attention_head_size"], bias=self.cfg["bias"]
            )
        self.value = nn.Linear(
            self.cfg["hidden_size"], self.cfg["attention_head_size"], bias=self.cfg["bias"]
            )
        self.dropout = nn.Dropout(self.cfg["attention_probs_dropout_prob"])


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
        ## -> (batch, token, token)
        scores = scores / math.sqrt(self.cfg["attention_head_size"])
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
        # register config
        default_config = {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "attention_probs_dropout_prob": 0.0,
            "qkv_bias": True, # Q, K, Vにbiasを使うかどうか
            "hidden_dropout_prob": 0.0,
        }
        self.cfg = {**default_config, **config}
        ## hidden_sizeはnum_attention_headsで割り切れるように
        self.attention_head_size = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        self.all_head_size = self.cfg["num_attention_heads"] * self.attention_head_size
        # layers
        self.heads = nn.ModuleList([])
        for _ in range(self.cfg["num_attention_heads"]):
            head = AttentionHead(
                self.cfg["hidden_size"],
                self.attention_head_size,
                self.cfg["attention_probs_dropout_prob"],
                self.cfg["qkv_bias"]
            )
            self.heads.append(head)
        ## attentionの出力をhidden sizeに戻すためのprojection layer
        ## 基本的にall_head_size = hidden_size
        self.output_projection = nn.Linear(self.all_head_size, self.cfg["hidden_size"])
        self.output_dropout = nn.Dropout(self.cfg["hidden_dropout_prob"])
    

    def forward(self, x, output_attentions=False):
        # 各attention headでattentionの計算
        attention_outputs = [head(x) for head in self.heads]
        # attention headを結合
        attention_output = torch.cat(
            [attention_output for attention_output, _ in attention_outputs], dim=-1
            )
        ## 各attention_outputは(batch, token, att_head)
        # concat済みのattentionをhidden sizeにproject
        attention_output = self.output_projection(attention_output)
        ## (batch, token, all head) -> (batch, token, hidden)
        attention_output = self.output_dropout(attention_output)
        # 出力
        if output_attentions:
            attention_probs = torch.stack(
                [attention_probs for _, attention_probs in attention_outputs], dim=-1
                )
            # stackに注意, (batch, token, token) -> (batch, token, token, head)?
            return (attention_output, attention_probs)
        else:
            return (attention_output, None)
    

class FasterMultiHeadAttention(nn.Module):
    """
    Multi-head attention with some optimizations
    
    """
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "attention_probs_dropout_prob": 0.0,
            "qkv_bias": True,
            "hidden_dropout_prob": 0.0,
        }
        self.cfg = {**default_config, **config}
        self.attention_head_size = self.cfg["hidden_size"] // self.cfg["num_attention_heads"]
        self.all_head_size = self.cfg["num_attention_heads"] * self.attention_head_size
        # layers
        ## Q, K, Vのlinear projection
        self.qkv_projection = nn.Linear(
            self.cfg["hidden_size"], self.all_head_size * 3, bias=self.cfg["qkv_bias"]
            )
        self.attn_dropout = nn.Dropout(self.cfg["attention_probs_dropout_prob"])
        ## attention outputをhiddenに戻すprojection
        self.output_projection = nn.Linear(self.all_head_size, self.cfg["hidden_size"])
        self.output_dropout = nn.Dropout(self.cfg["hidden_dropout_prob"])


    def forward(self, x, output_attentions=False):
        # Q, K, Vをまとめてlinear projectionする
        # (batch_size, seq_length, hidden_size)
        # -> (batch_size, seq_length, all_head_size * 3)
        qkv = self.qkv_projection(x)
        # Q, K, Vにprojectionを分割
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Q, K, Vのresize
        # -> (batch_size, num_attention_heads, seq_length, attention_head_size)
        batch_size, seq_length, _ = q.size()
        q = q.view(
            batch_size, seq_length, self.cfg["num_attention_heads"], self.attention_head_size
            ).transpose(1, 2)
        k = k.view(
            batch_size, seq_length, self.cfg["num_attention_heads"], self.attention_head_size
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_length, self.cfg["num_attention_heads"], self.attention_head_size
        ).transpose(1, 2)
        # attention scoreの計算
        # -> (batch, n_heads, token, token)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        # attentionの出力を得る
        attention_output = torch.matmul(attention_probs, v)
        # attention_otputをresizeする
        # (batch_size, num_attention_heads, seq_length, attention_head_size)
        # -> (batch_size, seq_length, all_head_size)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.all_head_size
            )
        # attention_outputをhidden sizeに戻す
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)
        if output_attentions:
            return (attention_output, attention_probs)
        else:
            return (attention_output, None)
    

class MLP(nn.Module):
    # position-wise feed-forward
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "intermediate_size": 4 * 32,
            "hidden_dropout_prob": 0.0,
        }
        self.cfg = {**default_config, **config}
        # layers
        self.dense1 = nn.Linear(self.cfg["hidden_size"], self.cfg["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense2 = nn.Linear(self.cfg["intermediate_size"], self.cfg["hidden_size"])
        self.dropout = nn.Dropout(self.cfg["hidden_dropout_prob"])


    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """ a single transformer block """
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "num_attention_heads": 4,
            "attention_probs_dropout_prob": 0.0,
            "qkv_bias": True,
            "hidden_dropout_prob": 0.0,
            "use_faster_attention": True,
        }
        self.cfg = {**default_config, **config}
        if self.use_faster_attention:
            self.attention = FasterMultiHeadAttention(self.cfg)
        else:
            self.attention = MultiHeadAttention(self.cfg)
        self.layernorm1 = nn.LayerNorm(self.cfg["hidden_size"])
        self.mlp = MLP(self.cfg)
        self.layernorm2 = nn.LayerNorm(self.cfg["hidden_size"])
    

    def forward(self, x, output_attentions=False):
        # self-attention
        attention_output, attention_probs = self.attention(
            self.layernorm1(x), output_attentions=output_attentions
        )
        ## -> (batch, token, hidden)
        # skip connection
        x = x + attention_output
        # Feed forward
        mlp_output = self.mlp(self.layernorm2(x))
        # skip connection
        x = x + mlp_output
        if output_attentions:
            return (x, attention_probs)
        else:
            return (x, None)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "attention_probs_dropout_prob": 0.0,
            "qkv_bias": True,
            "hidden_dropout_prob": 0.0,
            "use_faster_attention": True,
        }
        self.cfg = {**default_config, **config}
        ## transformer blockのlist
        self.blocks = nn.ModuleList([])
        for _ in range(self.cfg["num_hidden_layers"]):
            block = Block(self.cfg)
            self.blocks.append(block)
        

    def forward(self, x, output_attentions=False):
        # 各ブロックについてblockの出力を得る
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            # encoderのoutputはblockのoutputに一致, attentionはブロック数出る
            if output_attentions:
                all_attentions.append(attention_probs)
        if output_attentions:
            ## ブロック数のlistで返る, [(batch, head, token, token), ...]
            ## fasterとそうでないのとで位置が違うように見える
            return (x, all_attentions)
        else:
            return (x, None)


class VitForClassification(nn.Module):
    """ Vit for classification """
    def __init__(self, config):
        super().__init__()
        # register config
        default_config = {
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 4 * 32, # 4 * hidden_size
            "hidden_dropout_prob": 0.0,
            "attention_probs_dropout_prob": 0.0, 
            "initializer_range": 0.02, 
            "image_size": 32,
            "num_classes": 10, # num_classes of CIFAR10
            "num_channels": 3,
            "qkv_bias": True,
            "use_faster_attention": True,
        }
        self.cfg = {**default_config, **config}
        # embedding module
        self.embedding = Embeddings(self.cfg)
        # transofer encoder
        self.encoder = Encoder(self.cfg)
        # encoderの出力を受けた分類器
        self.classifier = nn.Linear(self.cfg["hidden_size"], self.cfg["num_classes"])
        # weightの初期化
        # applyはnn.Module由来, module内のインスタンスに対して再帰的に関数を当てる
        self.apply(self._init_weights)
    

    def forward(self, x, output_attentions=False):
        # embedding
        embedding_output = self.embedding(x)
        # encoding
        encoder_output, all_attentions = self.encoder(
            embedding_output, output_attentions=output_attentions
            )
        # logits
        # [CLS]のoutputをfeatureに使用, ここは色々ある
        ## encoder_output (batch, token, hidden)
        logits = self.classifier(encoder_output[:, 0, :])
        if output_attentions:
            return (logits, all_attentions)
        else:
            return (logits, None)
    

    def _init_weights(self, module):
        # モジュールごとに適切なものがあるので分ける
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=self.cfg["initializer_range"]
                )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            # truncated normalを使ってる
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.cfg["initializer_range"]
            ).to(module.position_embeddings.dtype)
            # CLS token
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.cfg["initializer_range"]
            ).to(module.position_embeddings.dtype)

