import torch.utils.data as data
import torch
import numpy as np
from functools import partial
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from utils import integer_label_protein
import os
from PIL import Image
class DTIDataset(data.Dataset):

    def __init__(self, dict_pro,input_ids,attention_mask,mol_data_root, max_drug_nodes=290,mol_transform=None,num_frames=60,precomputed_vit_features=None,precomputed_esm2_features=None,use_unique_features=False):

        self.dict_pro = dict_pro
        self.list_IDs = dict_pro['df'].index.values
        self.df = dict_pro['df']
        self.max_drug_nodes = max_drug_nodes

        self.atom_featurizer = CanonicalAtomFeaturizer()
        self.bond_featurizer = CanonicalBondFeaturizer(self_loop=True)
        self.fc = partial(smiles_to_bigraph, add_self_loop=True)

        self.mol_data_root = mol_data_root
        self.mol_transform = mol_transform
        self.num_frames = num_frames

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
        self.precomputed_vit_features = precomputed_vit_features
        self.use_precomputed = precomputed_vit_features is not None
        
        self.precomputed_esm2_features = precomputed_esm2_features
        self.use_esm2 = precomputed_esm2_features is not None
        
        # 方案A：唯一特征索引
        self.use_unique_features = use_unique_features
        if use_unique_features:
            self.protein_indices = dict_pro.get('protein_indices', None)
            self.smiles_indices = dict_pro.get('smiles_indices', None)

    def __len__(self):
        drugs_len = len(self.list_IDs)
        return drugs_len

    def _load_molecule_frames(self, index):
        """加载分子动态图像数据"""
        # 假设每个样本对应一个子目录，目录名与df中的index一致
        sample_dir = os.path.join(self.mol_data_root, str(index))

        # 获取排序后的帧文件路径
        frame_files = sorted([os.path.join(sample_dir, f)
                              for f in os.listdir(sample_dir)
                              if f.endswith(".png")])[:self.num_frames]

        # 加载并转换图像
        frames = []
        for path in frame_files:
            img = Image.open(path).convert("RGB")
            if self.mol_transform:
                img = self.mol_transform(img)
            frames.append(img)
        return torch.stack(frames)

    def __getitem__(self, index):
        index = self.list_IDs[index]
        v_d = self.df.iloc[index]['SMILES']
        v_d = self.fc(smiles=v_d, node_featurizer=self.atom_featurizer,edge_featurizer=self.bond_featurizer)

        actual_node_feats = v_d.ndata.pop('h')
        num_actual_nodes = actual_node_feats.shape[0]
        num_virtual_nodes = self.max_drug_nodes - num_actual_nodes
        virtual_node_bit = torch.zeros([num_actual_nodes, 1])
        actual_node_feats = torch.cat((actual_node_feats, virtual_node_bit), 1)
        v_d.ndata['h'] = actual_node_feats
        virtual_node_feat = torch.cat((torch.zeros(num_virtual_nodes, 74),
                                       torch.ones(num_virtual_nodes, 1)), 1)
        v_d.add_nodes(num_virtual_nodes, {'h':virtual_node_feat})
        v_d = v_d.add_self_loop()

        if self.use_esm2:
            if self.use_unique_features:
                protein_idx = self.protein_indices[index]
                v_p = self.precomputed_esm2_features[protein_idx]
            else:
                v_p = self.precomputed_esm2_features[index]
        else:
            v_p = self.df.iloc[index]['Protein']
            v_p = integer_label_protein(v_p)
        y = self.df.iloc[index]['Y']

        if self.use_precomputed:
            if self.use_unique_features and self.smiles_indices is not None:
                smiles_idx = self.smiles_indices[index]
                mol_frames = self.precomputed_vit_features[smiles_idx]
            else:
                mol_frames = self.precomputed_vit_features[index]
        else:
            mol_frames = self._load_molecule_frames(index)

        input_ids = self.input_ids[index]
        attention_mask = self.attention_mask[index]

        return v_d, v_p, y, mol_frames, input_ids, attention_mask


class MultiDataLoader(object):
    def __init__(self, dataloaders, n_batches):
        if n_batches <= 0:
            raise ValueError('n_batches should be > 0')
        self._dataloaders = dataloaders
        self._n_batches = np.maximum(1, n_batches)
        self._init_iterators()

    def _init_iterators(self):
        self._iterators = [iter(dl) for dl in self._dataloaders]

    def _get_nexts(self):
        def _get_next_dl_batch(di, dl):
            try:
                batch = next(dl)
            except StopIteration:
                new_dl = iter(self._dataloaders)
                self._iterators[di] = new_dl
                batch = next(new_dl)
            return batch

        return [_get_next_dl_batch(di, dl) for di, dl in enumerate(self._iterators)]

    def __iter__(self):
        for _ in range(self._n_batches):
            yield self._get_nexts()
        self._init_iterators()

    def __len__(self):
        return self._n_batches
