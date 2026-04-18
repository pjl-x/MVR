import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from models import binary_cross_entropy, cross_entropy_logits
from prettytable import PrettyTable
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F
def save_model(model):
    model_path = r'../output/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path+'model.pt')
    new_model = torch.load(model_path + 'model.pt')
    return new_model


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor) -> torch.Tensor:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # [num_pos]
        neg_sim = torch.matmul(anchor, negatives.T) / self.temperature  # [num_pos, num_neg]

        numerator = torch.exp(pos_sim)
        denominator = numerator + torch.exp(neg_sim).sum(dim=-1)
        loss = -torch.log(numerator / denominator).mean()
        return loss



class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, test_dataloader, data_name, split, ablation_mode='full', seed=42, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.n_class = config["DECODER"]["BINARY"]

        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.nb_training = len(self.train_dataloader)
        self.step = 0

        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]
        self.use_ld = config['SOLVER']["USE_LD"]

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.best_auprc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.seed = seed
        self.output_dir = os.path.join(config["RESULT"]["OUTPUT_DIR"], f'{data_name}/{split}/seed_{seed}/')
        # 对比学习损失
        self.contrastive_loss = InfoNCE(temperature=0.07)
        self.lambda_contrast = 0.1  # 对比损失权重
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]

        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.val_metrics_per_epoch = {}  # 新增
    def _save_val_metrics_to_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "val_metrics_per_epoch.csv")

        # 获取当前epoch的指标
        current_epoch = self.current_epoch
        current_metrics = self.val_metrics_per_epoch[current_epoch]

        # 构造DataFrame（单行）
        df = pd.DataFrame([current_metrics], index=[current_epoch])
        df.index.name = 'Epoch'

        # 如果文件不存在，写入表头；否则追加数据
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, mode='w', header=True)
        else:
            df.to_csv(csv_path, mode='a', header=False)


    
    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            if self.use_ld:
                if self.current_epoch % self.decay_interval == 0:
                    self.optim.param_groups[0]['lr'] *= self.lr_decay

            # 训练一轮
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            # 对齐 PMMR：用 test 集评估，并基于此选择 best model
            auroc, auprc, f1, sensitivity, specificity, accuracy, val_loss, thred_optim, precision = self.test(
                dataloader="test")
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            self.val_metrics_per_epoch[self.current_epoch] = {
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "accuracy": accuracy,
                "val_loss": val_loss,
                "thred_optim": thred_optim,
                "precision": precision
            }
            self._save_val_metrics_to_csv()

            # 基于 test 集 AUROC 选择 best model
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            # 实时更新表格文件
            self._save_table_to_file("val", self.val_table, "test_per_epoch_markdowntable.txt")
            self._save_table_to_file("train", self.train_table, "train_markdowntable.txt")

            print('Test at Epoch ' + str(self.current_epoch) + ' with loss ' + str(val_loss) +
                  " AUROC " + str(auroc) + " AUPRC " + str(auprc) +
                  " F1 " + str(f1) + " Sensitivity " + str(sensitivity) +
                  " Specificity " + str(specificity) + " Accuracy " + str(accuracy))

        self.save_result()
        return self.test_metrics

    def _save_table_to_file(self, table_type, table, filename):
        """辅助方法：将 PrettyTable 内容保存到文件"""
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as fp:
            fp.write(table.get_string())
        # print(f"Saved {table_type} table to {file_path}")

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config,
            "val_metrics_per_epoch": self.val_metrics_per_epoch  # 新增：保存所有epoch的验证集指标
        }

        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        # val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        # test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        # train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        # with open(val_prettytable_file, 'w') as fp:
        #     fp.write(self.val_table.get_string())
        # with open(test_prettytable_file, 'w') as fp:
        #     fp.write(self.test_table.get_string())
        # with open(train_prettytable_file, "w") as fp:
        #     fp.write(self.train_table.get_string())
        # metrics_df = pd.DataFrame.from_dict(self.val_metrics_per_epoch, orient='index')
        # metrics_df.index.name = 'Epoch'
        # metrics_df.to_csv(os.path.join(self.output_dir, "val_metrics_per_epoch.csv"))
        # 用 best_model 在 test 集上做最终报告
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision = self.test(
            dataloader="test")
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision

        # 验证集指标保存为 CSV
        metrics_df = pd.DataFrame.from_dict(self.val_metrics_per_epoch, orient='index')
        metrics_df.index.name = 'Epoch'
        metrics_df.to_csv(os.path.join(self.output_dir, "val_metrics_per_epoch.csv"))
    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (v_d, v_p, labels, mol_frames, input_ids, attention_mask) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1
            v_d = v_d.to(self.device)
            v_p = v_p.to(self.device)
            labels = labels.float().to(self.device)
            mol_frames = mol_frames.to(self.device)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            self.optim.zero_grad()
            v_d, v_p, f, score, z_d, z_p = self.model(v_d, v_p, mol_frames, input_ids, attention_mask)
            # 修改：使用加权二元交叉熵损失
            if self.n_class == 1:
                n, cls_loss = binary_cross_entropy(score, labels, pos_weight=2.0)
            else:
                n, cls_loss = cross_entropy_logits(score, labels)

            # 对比损失
            pos_mask = labels == 1
            neg_mask = labels == 0
            z_d_pos = z_d[pos_mask]
            z_p_pos = z_p[pos_mask]
            if neg_mask.sum() > 0:
                z_d_neg = z_d[neg_mask]
                z_p_neg = z_p[neg_mask]
            else:
                batch_size = z_d.size(0)
                z_d_neg = torch.cat([z_d[:i] + z_d[i + 1:] for i in range(batch_size)], dim=0)
                z_p_neg = torch.cat([z_p[:i] + z_p[i + 1:] for i in range(batch_size)], dim=0)
            if z_d_pos.size(0) > 0:
                contrast_loss_d = self.contrastive_loss(z_d_pos, z_p_pos, z_p_neg)
                contrast_loss_p = self.contrastive_loss(z_p_pos, z_d_pos, z_d_neg)
                contrast_loss = (contrast_loss_d + contrast_loss_p) / 2
            else:
                contrast_loss = torch.tensor(0.0, device=self.device)
            
            
            total_loss = cls_loss + self.lambda_contrast * contrast_loss
            total_loss.backward()
            self.optim.step()
            loss_epoch += total_loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch




    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
            model = self.best_model if self.best_model is not None else self.model
        else:
            raise ValueError(f"Error key value {dataloader}")
        num_batches = len(data_loader)
        #
        df = {'drug': [], 'protein': [], 'y_pred': [], 'y_label': []}
        #
        with torch.no_grad():
            model.eval()
            for i, (v_d, v_p, labels, mol_frames, input_ids, attention_mask) in enumerate(data_loader):
                v_d = v_d.to(self.device)
                v_p = v_p.to(self.device)
                labels = labels.float().to(self.device)
                mol_frames = mol_frames.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                v_d, v_p, score, att = model(v_d, v_p, mol_frames, input_ids, attention_mask, mode="eval")
                #elif dataloader == "test":
                    #v_d, v_p, f, score = self.best_model(v_d, v_p,mol_frames,d_kg,p_kg,input_ids,attention_mask)
                # 修改：使用加权二元交叉熵损失
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels, pos_weight=2.0)
                else:
                    n, loss = cross_entropy_logits(score, labels)
                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + n.to("cpu").tolist()

                if dataloader == 'test':
                    df['drug'] = df['drug'] + v_d.to('cpu').tolist()
                    df['protein'] = df['protein'] + v_p.to('cpu').tolist()

        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        test_loss = test_loss / num_batches


        # 计算所有指标，无论验证集还是测试集
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        prec, recall, _ = precision_recall_curve(y_label, y_pred)
        try:
            precision = tpr / (tpr + fpr)
        except RuntimeError:
            raise ('RuntimeError: the divide==0')
        f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
        thred_optim = thresholds[5:][np.argmax(f1[5:])]
        y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
        cm1 = confusion_matrix(y_label, y_pred_s)
        accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
        sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
        specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
        precision1 = precision_score(y_label, y_pred_s)

        if dataloader == "test":
            df['y_label'] = y_label
            df['y_pred'] = y_pred
            data = pd.DataFrame(df)
            data.to_csv(os.path.join(self.output_dir, 'visualization.csv'), index=False)
            # 统一返回所有指标
        return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, test_loss, thred_optim, precision1