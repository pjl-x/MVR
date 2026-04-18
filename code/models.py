import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from Intention import BiIntention

from transformers import ViTImageProcessor, ViTModel
from transformers import AutoModel, AutoTokenizer
from transformers import EsmTokenizer, EsmModel
from transformers import T5Tokenizer, T5EncoderModel


def binary_cross_entropy(pred_output, labels, pos_weight=2.0):
    
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(pred_output.device))
    n = torch.squeeze(pred_output, 1)  # 确保输出为一维
    loss = loss_fct(n, labels)
    m = nn.Sigmoid()
    return m(n), loss

def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor = ViTImageProcessor.from_pretrained("../vit_model")
        self.vit = ViTModel.from_pretrained("../vit_model")
        if True :
            for param in self.vit.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [batch, num_frames, C, H, W]
        batch_size, num_frames = x.shape[0], x.shape[1]
        features = []
        for t in range(num_frames):
            frame = x[:, t, :, :, :]
            outputs = self.vit(frame)
            features.append(outputs.last_hidden_state[:, 0, :])  # CLS token
        return torch.stack(features, dim=1)  # [batch, num_frames, 768]



class TemporalAggregator(nn.Module):
    def __init__(self, input_dim=768, lstm_hidden_dim=256, attn_hidden_dim=128, num_heads=8):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim,
                            bidirectional=True, batch_first=True)
        self.layer_norm_lstm = nn.LayerNorm(2 * lstm_hidden_dim)  # LSTM 后添加 LayerNorm
        self.multihead_attn = nn.MultiheadAttention(2 * lstm_hidden_dim, num_heads, batch_first=True)  # 多头注意力
        self.layer_norm_attn = nn.LayerNorm(2 * lstm_hidden_dim)  # 多头注意力后添加 LayerNorm
        self.attention = nn.Sequential(
            nn.Linear(2 * lstm_hidden_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Linear(attn_hidden_dim, 1)
        )
        self.dropout = nn.Dropout(0.1)
        # 添加一个全连接层，将输出维度从 2 * lstm_hidden_dim 映射到 256


    def forward(self, x):
        # LSTM 处理
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 2 * lstm_hidden_dim]
        lstm_out = self.layer_norm_lstm(lstm_out)  # LSTM 输出后归一化
        lstm_out = self.dropout(lstm_out)

        # 多头注意力机制
        attn_output, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)  # [batch, seq_len, 2 * lstm_hidden_dim]
        attn_output = self.layer_norm_attn(attn_output)  # 多头注意力输出后归一化
        attn_output = self.dropout(attn_output)

        # 注意力机制
        attn_weights = torch.softmax(self.attention(attn_output), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * attn_output, dim=1)  # [batch, 2 * lstm_hidden_dim]

        return context

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = nn.Dropout(0.1)
        self.act = nn.ReLU()

    def forward(self, v):
        v = self.linear(v)
        v = self.dropout(v)
        v = self.act(v)
        return v

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):  #64

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_f = nn.Linear(256, embed_dim)
        self.proj_kg = nn.Linear(256, embed_dim)
        self.proj_temporal = nn.Linear(512, embed_dim)
        self.proj_smiles = nn.Linear(128, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, f, v_kg, temporal_feat, smiles_feats):
        f = self.proj_f(f)
        v_kg = self.proj_kg(v_kg)
        temporal_feat = self.proj_temporal(temporal_feat)
        smiles_feats = self.proj_smiles(smiles_feats)
        modalities = torch.stack([f, v_kg, temporal_feat, smiles_feats], dim=1)  # [batch_size, 4, 256]
        attn_output, _ = self.attn(modalities, modalities, modalities)
        fused = attn_output.mean(dim=1)
        fused = self.layer_norm(fused + modalities.mean(dim=1))
        fused = self.dropout(fused)
        return fused


class BINDTI(nn.Module):
    def __init__(self, device='cuda', use_precomputed_vit=False, use_esm2=False, **config):
        super(BINDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.use_esm2 = use_esm2
        if use_esm2:
            plm_hidden_dim = config["PROTEIN"].get("HIDDEN_DIM", config["PROTEIN"].get("ESM_HIDDEN_DIM", 480))
            adapter_layers = config["PROTEIN"].get("ADAPTER_LAYERS", config["PROTEIN"].get("ESM_ADAPTER_LAYERS", 3))
            use_transformer = config["PROTEIN"].get("USE_TRANSFORMER", config["PROTEIN"].get("ESM_USE_TRANSFORMER", True))
            transformer_layers = config["PROTEIN"].get("TRANSFORMER_LAYERS", config["PROTEIN"].get("ESM_TRANSFORMER_LAYERS", 2))
            transformer_heads = config["PROTEIN"].get("TRANSFORMER_HEADS", config["PROTEIN"].get("ESM_TRANSFORMER_HEADS", 8))
            self.protein_extractor = ProteinPLMAdapter(
                plm_hidden_dim=plm_hidden_dim,
                output_dim=num_filters[-1],
                num_layers=adapter_layers,
                dropout=0.1,
                use_transformer=use_transformer,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads
            )
        else:
            self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer, device=device)

        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

        self.use_precomputed_vit = use_precomputed_vit
        if not use_precomputed_vit:
            self.feature_extractor = ViTFeatureExtractor()
        self.temporal_agg = TemporalAggregator()

        self.bert = AutoModel.from_pretrained('../molformer', trust_remote_code=True)
        
        
        frozen_layers = 10
        for name, param in self.bert.named_parameters():
            # 匹配 encoder.layer.{i}.xxx 格式的参数名
            if 'encoder.layer.' in name:
                try:
                    # 提取层号：encoder.layer.{i}.xxx -> {i}
                    layer_idx = int(name.split('encoder.layer.')[1].split('.')[0])
                    if layer_idx < frozen_layers:
                        param.requires_grad = False
                except (ValueError, IndexError):
                    # 如果解析失败，跳过（保持可训练）
                    pass
        
        # 打印冻结信息（可选，用于验证）
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.bert.parameters())
        print(f"MoLFormer: 冻结前 {frozen_layers} 层，可训练参数: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
        
        bert_hidden_dim = self.bert.config.hidden_size
        self.smiles_proj = nn.Linear(bert_hidden_dim, drug_hidden_feats[-1])
        self.relu = nn.ReLU()
        self.K_pro = MLP(400, 256)
        self.LayerNorm = LayerNorm(256)
        self.fusion = AttentionFusion(embed_dim=256, num_heads=8)
    def forward(self, bg_d, v_p,mol_frames,d_kg,p_kg,input_ids,attention_mask,mode="train"):
        v_p = self.protein_extractor(v_p)
        
        if self.use_precomputed_vit:
            features = mol_frames
        else:
            features = self.feature_extractor(mol_frames)
        temporal_feat = self.temporal_agg(features)

        d_kg=d_kg
        p_kg=p_kg
        v_kg = torch.cat((d_kg, p_kg), 1)

        v_kg = self.K_pro(v_kg)
        v_kg = self.LayerNorm(v_kg)

        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        smiles_feats = bert_outputs.last_hidden_state[:, 0, :]
        smiles_feats = self.smiles_proj(smiles_feats)
        v_d = self.drug_extractor(bg_d)

        if mode == "train":
            f, v_d, v_p, z_d, z_p, att = self.cross_intention(drug=v_d, protein=v_p, mode="train")
        else:
            f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p, mode="eval")

        f = self.fusion(f, v_kg, temporal_feat, smiles_feats)
        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score, z_d, z_p
        return v_d, v_p, score, att

class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]
        self.fc = nn.Linear(512, 128)

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size

        

        node_feats = node_feats.view(batch_size, -1, self.output_feats)


        return node_feats


class ProteinPLMAdapter(nn.Module):
    """
    Adapter for Protein Language Models (ESM-2, ProtT5, etc.)
    Maps PLM embeddings to task-specific representations
    """
    def __init__(self, plm_hidden_dim=480, output_dim=128, num_layers=3, dropout=0.1,
                 use_transformer=True, transformer_layers=2, transformer_heads=8):
        super(ProteinPLMAdapter, self).__init__()
        
        # 定义逐步降维的维度序列（自动适配不同的 hidden_dim）
        if num_layers == 2:
            dims = [plm_hidden_dim, (plm_hidden_dim + output_dim) // 2, output_dim]
        elif num_layers == 3:
            # 动态计算中间层维度，确保平滑降维
            mid1 = max(384, (plm_hidden_dim * 2 + output_dim) // 3)
            mid2 = max(256, (plm_hidden_dim + output_dim * 2) // 3)
            dims = [plm_hidden_dim, mid1, mid2, output_dim]
        elif num_layers == 4:
            mid1 = max(512, (plm_hidden_dim * 3 + output_dim) // 4)
            mid2 = max(384, (plm_hidden_dim * 2 + output_dim * 2) // 4)
            mid3 = max(256, (plm_hidden_dim + output_dim * 3) // 4)
            dims = [plm_hidden_dim, mid1, mid2, mid3, output_dim]
        else:
            step = (plm_hidden_dim - output_dim) / num_layers
            dims = [plm_hidden_dim] + [int(plm_hidden_dim - step * (i+1)) for i in range(num_layers-1)] + [output_dim]
        
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.adapter = nn.Sequential(*layers)
        
        self.use_residual = (plm_hidden_dim != output_dim) and num_layers >= 3
        if self.use_residual:
            self.residual_proj = nn.Linear(plm_hidden_dim, output_dim)
        
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=transformer_heads,
                dim_feedforward=output_dim * 4,
                dropout=dropout,
                activation='relu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=transformer_layers
            )
            self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, plm_features):
        identity = plm_features
        adapted = self.adapter(plm_features)
        
        if self.use_residual:
            identity = self.residual_proj(identity)
            adapted = adapted + identity
        
        if self.use_transformer:
            transformed = self.transformer(adapted)
            adapted = self.layer_norm(adapted + transformed)
        
        adapted = adapted.transpose(2, 1).transpose(1, 2)
        return adapted

# Backward compatibility alias
ProteinESM2Adapter = ProteinPLMAdapter

class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)#64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))

        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))

        v = v.view(v.size(0), v.size(2), -1)
        return v

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):#x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
