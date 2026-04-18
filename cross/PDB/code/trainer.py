import pandas as pd
import torch
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from prettytable import PrettyTable
from tqdm import tqdm
import copy
import torch.nn as nn
import torch.nn.functional as F

# 如果有自定的 loss，请确保 models.py 中存在
from models import mse_loss


def save_model(model):
    model_path = r'../output/model/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, model_path + 'model.pt')
    new_model = torch.load(model_path + 'model.pt')
    return new_model


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def pearson(y_true, y_pred):
    return pearsonr(y_true, y_pred)[0]


def spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred)[0]


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
    def __init__(self, model, optim, device, train_dataloader, test_dataloader, data_name, split, ablation_mode='full',
                 seed=42, **config):
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
        self.best_mae = float('inf')  # 回归任务：MAE/RMSE 越小越好

        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.test_metrics = {}
        self.config = config
        self.seed = seed
        self.output_dir = os.path.join(config["RESULT"]["OUTPUT_DIR"], f'{data_name}/{split}/seed_{seed}/')

        valid_metric_header = ["# Epoch", "RMSE", "MAE", "Pearson", "Spearman", "Val_loss"]
        test_metric_header = ["# Best Epoch", "RMSE", "MAE", "Pearson", "Spearman", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.val_metrics_per_epoch = {}

    def _save_val_metrics_to_csv(self):
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "val_metrics_per_epoch.csv")

        current_epoch = self.current_epoch
        current_metrics = self.val_metrics_per_epoch[current_epoch]

        df = pd.DataFrame([current_metrics], index=[current_epoch])
        df.index.name = 'Epoch'

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

            # 评估一轮 (注意：此时一定要传入 use_best=False，否则一直测的是旧的 best_model)
            rmse_val, mae_val, pearson_val, spearman_val, val_loss = self.test(dataloader="test", use_best=False)
            val_lst = ["epoch " + str(self.current_epoch)] + list(
                map(float2str, [rmse_val, mae_val, pearson_val, spearman_val, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)

            self.val_metrics_per_epoch[self.current_epoch] = {
                "rmse": rmse_val,
                "mae": mae_val,
                "pearson": pearson_val,
                "spearman": spearman_val,
                "val_loss": val_loss
            }
            self._save_val_metrics_to_csv()

            # 基于 test 集 MAE 挑选 best model
            if mae_val < self.best_mae:
                self.best_model = copy.deepcopy(self.model)
                self.best_mae = mae_val
                self.best_epoch = self.current_epoch
                print(f'[Improved!] MAE improved at epoch {self.current_epoch}; best_mae: {self.best_mae:.4f}')
            else:
                print(f'[No improve] No improvement since epoch {self.best_epoch}; best_mae: {self.best_mae:.4f}')

            # 实时更新表格文件
            self._save_table_to_file("val", self.val_table, "test_per_epoch_markdowntable.txt")
            self._save_table_to_file("train", self.train_table, "train_markdowntable.txt")

            # 修复：移除 AUROC 等分类指标，打印回归指标
            print(f'Test at Epoch {self.current_epoch} with val_loss {val_loss:.4f} '
                  f'RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | Pearson: {pearson_val:.4f} | Spearman: {spearman_val:.4f}\n')

        self.save_result()
        return self.test_metrics

    def _save_table_to_file(self, table_type, table, filename):
        file_path = os.path.join(self.output_dir, filename)
        with open(file_path, 'w') as fp:
            fp.write(table.get_string())

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "config": self.config,
            "val_metrics_per_epoch": self.val_metrics_per_epoch
        }

        # 用 best_model 在 test 集上做最终报告 (注意：use_best=True)
        rmse_test, mae_test, pearson_test, spearman_test, test_loss = self.test(dataloader="test", use_best=True)

        self.test_metrics["rmse"] = rmse_test
        self.test_metrics["mae"] = mae_test
        self.test_metrics["pearson"] = pearson_test
        self.test_metrics["spearman"] = spearman_test
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["best_epoch"] = self.best_epoch
        state["test_metrics"] = self.test_metrics

        # 将最终最好模型的测试结果加入表格并保存
        float2str = lambda x: '%0.4f' % x
        test_lst = ["epoch " + str(self.best_epoch)] + list(
            map(float2str, [rmse_test, mae_test, pearson_test, spearman_test, test_loss]))
        self.test_table.add_row(test_lst)
        self._save_table_to_file("test", self.test_table, "best_test_markdowntable.txt")

        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

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
            score = self.model(v_d, v_p, mol_frames, input_ids, attention_mask)
            pred, loss = mse_loss(score, labels)

            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print(f'Training at Epoch {self.current_epoch} with training loss {loss_epoch:.4f}')
        return loss_epoch

    def test(self, dataloader="test", use_best=False):
        """
        use_best: 如果为 True，则使用 self.best_model (用于最终报告)
                  如果为 False，则使用当前的 self.model (用于每个 Epoch 的日常跟踪评估)
        """
        test_loss = 0
        y_label, y_pred = [], []

        if dataloader == "test":
            data_loader = self.test_dataloader
            # 修复模型选择逻辑
            model = self.best_model if (use_best and self.best_model is not None) else self.model
        else:
            raise ValueError(f"Error key value {dataloader}")

        num_batches = len(data_loader)

        with torch.no_grad():
            model.eval()
            for i, (v_d, v_p, labels, mol_frames, input_ids, attention_mask) in enumerate(data_loader):
                v_d = v_d.to(self.device)
                v_p = v_p.to(self.device)
                labels = labels.float().to(self.device)
                mol_frames = mol_frames.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                score = model(v_d, v_p, mol_frames, input_ids, attention_mask)
                pred, loss = mse_loss(score, labels)

                test_loss += loss.item()
                y_label = y_label + labels.to("cpu").tolist()
                y_pred = y_pred + pred.to("cpu").tolist()

        y_label = np.array(y_label)
        y_pred = np.array(y_pred)

        rmse_val = rmse(y_label, y_pred)
        mae_val = mae(y_label, y_pred)
        pearson_val = pearson(y_label, y_pred)
        spearman_val = spearman(y_label, y_pred)
        test_loss = test_loss / num_batches

        # 只在调用最后一次(即 use_best=True)的时候生成最终的可视化散点CSV
        if dataloader == "test" and use_best:
            df = {'y_label': y_label.tolist(), 'y_pred': y_pred.tolist()}
            data = pd.DataFrame(df)
            data.to_csv(os.path.join(self.output_dir, 'visualization.csv'), index=False)

        return rmse_val, mae_val, pearson_val, spearman_val, test_loss