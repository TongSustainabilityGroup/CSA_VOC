import math
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    coverage_error,
    f1_score,
    label_ranking_loss,
    precision_score,
    recall_score,
)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from resnet import ResNet, BasicBlock


class DyeDataset(Dataset):
    def __init__(self, csa_data, csa_label, index_list, mean, std, stage='train'):
        super().__init__()
        self.csa_data = csa_data
        self.csa_label = csa_label
        self.index_list = index_list
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.stage = stage
        self.applier = DyeAugment()

    def raw_frames(self, idx):
        experiment, block_start, phase, indices = self.index_list[idx]
        inp = np.asarray(self.csa_data[experiment, list(indices)]).copy()
        return torch.from_numpy(inp).float() / 255.0

    def __getitem__(self, idx):
        experiment, block_start, phase, indices = self.index_list[idx]
        inp = self.raw_frames(idx)
        if self.stage == 'train':
            inp = self.applier(inp)
        inp = (inp - self.mean) / self.std
        label = torch.from_numpy(self.csa_label[experiment].astype(np.float32))
        return inp, label, experiment, phase, block_start

    def __len__(self):
        return len(self.index_list)


class DyeDataModule:
    def __init__(self, batch_size, data_dir='csa_dataset.hdf5', cache_path=None,
                 kf_split=3, kf_fold=0, seed=0, num_workers=0):
        self.data_dir = Path(data_dir)
        self.cache_path = Path(cache_path) if cache_path else Path('dyeV8_cache/voc_full_images_64.npy')
        self.batch_size = batch_size
        self.kf_split = kf_split
        self.kf_fold = kf_fold
        self.seed = seed
        self.num_workers = num_workers

        self.img_size = 64       # Resize each frame to 64 × 64 pixels
        self.label_len = 37      # Total 37 experiments in the dataset
        self.tot_frames = 576    # Total frames in the complete sequence (576 × 5 min = 48 hrs)
        self.block_frames = 144  # Frames in each processing block (144 × 5 min = 12 hrs)
        self.skip = 18           # Sample one frame every 18 frames (18 × 5 min = 1.5 hrs)
        self.inp_len = 8         # Number of sampled frames given to the model (12 hour block)
        self.phases = 18         # Number of possible starting offsets (0–17)
        # 576 / 144 = 4 blocks per complete sequence.
        # 144 / 18 = 8 sampled frames per block, matching inp_len.
        # A input block sample frames e.g. 0, 18, 36, …, 126.
        # phases = 18 allows sampling to begin at any offset from 0 through 17.
        # num_class=14 to differentiate between 14 different VOCs.

    def _load_data(self):
        if self.cache_path.exists():
            images = np.load(self.cache_path, mmap_mode='r')
            expected = (self.label_len, self.tot_frames, 3, self.img_size, self.img_size)
        else:
            import cv2
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            images = np.lib.format.open_memmap(
                self.cache_path, mode='w+', dtype=np.uint8,
                shape=(self.label_len, self.tot_frames, 3, self.img_size, self.img_size),
            )
            with h5py.File(self.data_dir, 'r') as hf:
                source = hf['csa_data']
                for experiment in range(self.label_len):
                    for frame in range(self.tot_frames):
                        rgb = np.moveaxis(source[experiment, frame], 0, -1)
                        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
                        images[experiment, frame] = np.moveaxis(rgb, -1, 0)
                    print(f'cache: experiment {experiment + 1}/{self.label_len}', flush=True)
            images.flush()
            images = np.load(self.cache_path, mmap_mode='r')

        with h5py.File(self.data_dir, 'r') as hf:
            labels = np.asarray(hf['csa_labels'][:self.label_len], dtype=np.int64)
        return images, labels

    def _make_index(self, block_start):
        index_list = []
        for experiment in range(self.label_len):
            for phase in range(self.phases):
                indices = tuple(
                    block_start + phase + i * self.skip for i in range(self.inp_len)
                )
                if indices[0] < block_start or indices[-1] >= block_start + self.block_frames:
                    raise AssertionError('A sequence crossed its 12-hour block boundary')
                index_list.append((experiment, block_start, phase, indices))
        return index_list

    def _normalization(self, block_start):
        channel_sum = np.zeros(3, dtype=np.float64)
        channel_sq_sum = np.zeros(3, dtype=np.float64)
        count = 0
        frame_index = np.arange(block_start, block_start + self.block_frames, 6)
        for experiment in range(self.label_len):
            data = np.asarray(self.csa_data[experiment, frame_index], dtype=np.float64) / 255.0
            channel_sum += data.sum(axis=(0, 2, 3))
            channel_sq_sum += np.square(data).sum(axis=(0, 2, 3))
            count += data.shape[0] * data.shape[2] * data.shape[3]
        mean = channel_sum / count
        var = np.maximum(channel_sq_sum / count - np.square(mean), 1e-8)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

    def setup(self, stage=None):
        if self.kf_fold < 0 or self.kf_fold >= self.kf_split:
            raise ValueError(f'kf_fold must be between 0 and {self.kf_split - 1}')

        self.csa_data, self.csa_label = self._load_data()
        # Starting frame of the 12-hour training block.
        # Each fold shifts it forward by 72 frames (6 hrs).
        # Training starting hr for 3 rolling kf folds: 0, 6, 12.
        self.train_start = self.kf_fold * 72
        # Starting frame of the 12 hr purge block immediately after training block.
        self.purge_start = self.train_start + self.block_frames
        self.test_start = self.train_start + 2 * self.block_frames
        self.test_end = self.test_start + self.block_frames
        if self.test_end > self.tot_frames:
            raise AssertionError('Fold extends beyond the 48-hour experiment')

        self.train_index = self._make_index(self.train_start)
        self.test_index = self._make_index(self.test_start)
        if max(i for row in self.train_index for i in row[3]) >= self.purge_start:
            raise AssertionError('Training frames crossed into the purge interval')
        if min(i for row in self.test_index for i in row[3]) <= max(i for row in self.train_index for i in row[3]):
            raise AssertionError('Test frames are not later than all training frames')

        self.mean, self.std = self._normalization(self.train_start)
        self.train_dataset = DyeDataset(
            self.csa_data, self.csa_label, self.train_index,
            self.mean, self.std, stage='train',
        )
        self.test_dataset = DyeDataset(
            self.csa_data, self.csa_label, self.test_index,
            self.mean, self.std, stage='test',
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=False, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return self.val_dataloader()


class DyeAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = (0.75, 1.50)
        self.offset = (0.0, 0.20)
        self.gamma = (0.70, 1.35)
        self.gradient = 0.12
        self.degrees = 9.0
        self.translate = 0.045
        self.scale = (0.92, 1.08)
        self.noise_sd = 0.10

    def forward(self, x):
        # x: (T, C, H, W)
        height, width = x.shape[-2:]
        gain = torch.empty(1, 1, 1, 1).uniform_(*self.gain)
        offset = torch.empty(1, 1, 1, 1).uniform_(*self.offset)
        gamma = float(torch.empty(1).uniform_(*self.gamma))
        xx = torch.linspace(-1, 1, width).view(1, 1, 1, width)
        yy = torch.linspace(-1, 1, height).view(1, 1, height, 1)
        gx = torch.empty(1).uniform_(-self.gradient, self.gradient)
        gy = torch.empty(1).uniform_(-self.gradient, self.gradient)
        x = (x.clamp(1e-4, 1).pow(gamma) * gain + offset + gx * xx + gy * yy).clamp(0, 1)

        pad = max(4, int(round(max(height, width) * 0.08)))
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        padded_height, padded_width = x.shape[-2:]
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees))
        translation = [
            int(round(float(torch.empty(1).uniform_(-self.translate, self.translate)) * padded_width)),
            int(round(float(torch.empty(1).uniform_(-self.translate, self.translate)) * padded_height)),
        ]
        scale = float(torch.empty(1).uniform_(*self.scale))
        x = TVF.affine(
            x, angle=angle, translate=translation, scale=scale,
            shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR,
            fill=None,
        ).clamp(0, 1)
        top = (x.shape[-2] - height) // 2
        left = (x.shape[-1] - width) // 2
        x = x[..., top:top + height, left:left + width]

        return (x + self.noise_sd * torch.randn_like(x)).clamp(0, 1)


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, smooth_factor=0.03, pos_weight=None):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, y_pred, y_true):
        soft_targets = y_true * (1 - self.smooth_factor) + 0.5 * self.smooth_factor
        return F.binary_cross_entropy_with_logits(
            y_pred, soft_targets, pos_weight=self.pos_weight,
        )


class DyeModel(torch.nn.Module):
    def __init__(self, num_class, resnet_size=5, feature_dim=32, dim_ff=64,
                 num_head=4, tf_layers=1, dropout=0.20, img_size=64,
                 inp_len=8, pool_size=4,
                 pretrained_path='resnet32_check_point.pth'):
        super(DyeModel, self).__init__()
        self.img_size = img_size
        self.feature_dim = feature_dim
        self.pool_size = pool_size

        self.resnet = ResNet(
            BasicBlock,
            [resnet_size, resnet_size, resnet_size],
            num_classes=feature_dim,
        )
        pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        state_dict = pretrained.state_dict()
        state_dict.pop('linear.weight', None)
        state_dict.pop('linear.bias', None)
        self.resnet.load_state_dict(state_dict, strict=False)

        pe = torch.zeros(inp_len, feature_dim)
        position = torch.arange(0, inp_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, feature_dim, 2).float()
            * (-math.log(10000.0) / feature_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_head, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=tf_layers, norm=nn.LayerNorm(feature_dim)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * feature_dim, num_class),
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.shape
        x = x.reshape(batch_size * seq_len, channels, height, width)
        x = F.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)
        x = self.resnet(x).reshape(batch_size, seq_len, self.feature_dim)
        x = self.transformer(x + self.position[:, :seq_len])
        x = torch.cat([x.mean(dim=1), x[:, -1]], dim=1)
        return self.classifier(x)


class DyePredictor(nn.Module):
    def __init__(self, labels, num_class=14, resnet_size=5, feature_dim=32,
                 dim_ff=64, num_head=4, tf_layers=1, dropout=0.20,
                 img_size=64, inp_len=8, pool_size=4,
                 pretrained_path='resnet32_check_point.pth', lr=1.2e-3,
                 weight_decay=1e-3, epochs=20, label_smoothing=0.01,
                 pretrained_lr=None, pos_weight_power=1.0, device=None):
        super().__init__()
        self.model = DyeModel(
            num_class=num_class, resnet_size=resnet_size,
            feature_dim=feature_dim, dim_ff=dim_ff, num_head=num_head,
            tf_layers=tf_layers, dropout=dropout, img_size=img_size,
            inp_len=inp_len, pool_size=pool_size,
            pretrained_path=pretrained_path,
        )

        prevalence = np.clip(labels.mean(axis=0), 1e-4, 1 - 1e-4)
        with torch.no_grad():
            self.model.classifier[-1].bias.copy_(
                torch.from_numpy(np.log(prevalence / (1 - prevalence)).astype(np.float32))
            )

        positives = labels.sum(axis=0).astype(np.float64)
        negatives = len(labels) - positives
        pos_weight = np.clip(
            (negatives / np.maximum(positives, 1.0)) ** pos_weight_power,
            1.0, 12.0,
        ).astype(np.float32)

        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.loss = SoftBCEWithLogitsLoss(
            smooth_factor=label_smoothing,
            pos_weight=torch.from_numpy(pos_weight).to(self.device),
        )
        if pretrained_lr is None:
            optimizer_params = self.model.parameters()
        else:
            body_params = [
                parameter for name, parameter in self.model.resnet.named_parameters()
                if not name.startswith('linear.')
            ]
            new_params = [
                *self.model.resnet.linear.parameters(),
                *self.model.transformer.parameters(),
                *self.model.classifier.parameters(),
            ]
            optimizer_params = [
                {'params': body_params, 'lr': pretrained_lr},
                {'params': new_params, 'lr': lr},
            ]
        self.optimizer = torch.optim.AdamW(
            optimizer_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(epochs, 1),
        )

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def evaluate_epoch(self, test_loader, labels, label_names, threshold=0.5):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            self.model.eval()
            total_loss = 0.0
            total_examples = 0
            probabilities = []
            experiments = []
            for x, y, experiment, *_ in test_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                logits = self(x)
                loss = self.loss(logits, y)
                total_loss += float(loss.detach()) * len(x)
                total_examples += len(x)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                experiments.append(np.asarray(experiment))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        window_scores = np.concatenate(probabilities)
        experiment_ids = np.concatenate(experiments)
        scores = average_by_experiment(window_scores, experiment_ids)
        metrics = multilabel_metrics(labels, scores, threshold)
        classes = per_class_metrics(labels, scores, label_names, threshold)
        return total_loss / max(total_examples, 1), metrics, classes

    def fit(self, train_loader, test_loader, labels, label_names, epochs,
            kf_fold, seed, threshold=0.5):
        history = []
        class_history = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_examples = 0
            for x, y, *_ in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self(x)
                loss = self.loss(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                total_loss += float(loss.detach()) * len(x)
                total_examples += len(x)
            self.lr_scheduler.step()
            test_loss, test_metrics, test_classes = self.evaluate_epoch(
                test_loader, labels, label_names, threshold,
            )
            row = {
                'fold': kf_fold + 1,
                'seed': seed,
                'epoch': epoch,
                'train_loss': total_loss / max(total_examples, 1),
                'test_loss': test_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'head_learning_rate': self.optimizer.param_groups[-1]['lr'],
            }
            row.update({f'test_{name}': value for name, value in test_metrics.items()})
            history.append(row)
            test_classes.insert(0, 'epoch', epoch)
            test_classes.insert(0, 'seed', seed)
            test_classes.insert(0, 'fold', kf_fold + 1)
            class_history.append(test_classes)
            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                print(
                    f'fold {kf_fold + 1} seed {seed} epoch {epoch:02d}/{epochs}: '
                    f'train_loss={row["train_loss"]:.5f}, '
                    f'test_loss={row["test_loss"]:.5f}, '
                    f'test_micro_ap={row["test_micro_ap"]:.5f}', flush=True,
                )
        return pd.DataFrame(history), pd.concat(class_history, ignore_index=True)

    @torch.no_grad()
    def predict(self, test_loader):
        self.model.eval()
        probabilities = []
        experiments = []
        for x, y, experiment, *_ in test_loader:
            pred = torch.sigmoid(self(x.to(self.device))).cpu().numpy()
            probabilities.append(pred)
            experiments.append(np.asarray(experiment))
        return np.concatenate(probabilities), np.concatenate(experiments)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def average_by_experiment(window_scores, experiments, label_len=37,
                          num_class=14, phases=18):
    scores = np.zeros((label_len, num_class), dtype=np.float64)
    for experiment in range(label_len):
        selected = window_scores[experiments == experiment]
        if len(selected) != phases:
            raise AssertionError(
                f'Experiment {experiment} has {len(selected)}, not {phases}, windows'
            )
        scores[experiment] = selected.mean(axis=0)
    return scores


def multilabel_metrics(labels, scores, threshold=0.5):
    pred = (scores >= threshold).astype(np.int64)
    return {
        'micro_ap': float(average_precision_score(labels, scores, average='micro')),
        'macro_ap': float(average_precision_score(labels, scores, average='macro')),
        'micro_precision': float(precision_score(labels, pred, average='micro', zero_division=0)),
        'macro_precision': float(precision_score(labels, pred, average='macro', zero_division=0)),
        'micro_recall': float(recall_score(labels, pred, average='micro', zero_division=0)),
        'macro_recall': float(recall_score(labels, pred, average='macro', zero_division=0)),
        'micro_f1': float(f1_score(labels, pred, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(labels, pred, average='macro', zero_division=0)),
        'coverage_error': float(coverage_error(labels, scores)),
        'label_ranking_loss': float(label_ranking_loss(labels, scores)),
        'binary_accuracy': float(accuracy_score(labels.reshape(-1), pred.reshape(-1))),
        'exact_match_accuracy': float(accuracy_score(labels, pred)),
    }


def per_class_metrics(labels, scores, label_names, threshold=0.5):
    pred = (scores >= threshold).astype(np.int64)
    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            'class_index': i,
            'class_name': name,
            'prevalence': float(labels[:, i].mean()),
            'positive_experiments': int(labels[:, i].sum()),
            'ap': float(average_precision_score(labels[:, i], scores[:, i])),
            'precision': float(precision_score(labels[:, i], pred[:, i], zero_division=0)),
            'recall': float(recall_score(labels[:, i], pred[:, i], zero_division=0)),
            'f1': float(f1_score(labels[:, i], pred[:, i], zero_division=0)),
        })
    return pd.DataFrame(rows)


def run_seed(DyeData, labels, params_dict, seed, models_dir, predictions_dir,
             histories_dir, resume=True, label_names=None, threshold=0.5):
    fold = DyeData.kf_fold + 1
    prediction_path = predictions_dir / f'fold{fold}_seed{seed}_experiment_predictions.npz'
    checkpoint_path = models_dir / f'fold{fold}_seed{seed}.pt'
    history_path = histories_dir / f'fold{fold}_seed{seed}_history.csv'
    class_history_path = (
        histories_dir / f'fold{fold}_seed{seed}_epoch_per_class_metrics.csv'
    )
    history_complete = history_path.exists() and class_history_path.exists()

    if resume and prediction_path.exists() and history_complete:
        print(f'fold {fold} seed {seed}: using completed prediction', flush=True)
        return np.load(prediction_path)['scores'], True

    set_seed(seed)
    dmodel = DyePredictor(labels=labels, device=params_dict['device'], **params_dict['model'])
    if resume and checkpoint_path.exists() and history_complete:
        checkpoint = torch.load(checkpoint_path, map_location=params_dict['device'], weights_only=False)
        dmodel.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'fold {fold} seed {seed}: using checkpoint for inference', flush=True)
    else:
        if label_names is None:
            label_names = np.asarray([f'class_{i}' for i in range(labels.shape[1])])
        history, class_history = dmodel.fit(
            DyeData.train_dataloader(), DyeData.test_dataloader(), labels,
            label_names, params_dict['model']['epochs'], DyeData.kf_fold,
            seed, threshold,
        )
        history.to_csv(history_path, index=False)
        class_history.to_csv(class_history_path, index=False)
        torch.save({
            'model_state_dict': dmodel.model.state_dict(),
            'model_config': params_dict['checkpoint_config'],
            'fold': fold,
            'seed': seed,
        }, checkpoint_path)

    window_scores, experiments = dmodel.predict(DyeData.test_dataloader())
    scores = average_by_experiment(window_scores, experiments)
    np.savez_compressed(
        prediction_path, scores=scores, labels=labels,
        experiments=np.arange(37), fold=fold, seed=seed,
    )
    return scores, False


if __name__ == '__main__':

    img_size = 64
    batch_size = 32
    num_class = 14

    resnet_size = 5
    feature_dim = 32
    dim_ff = 64
    num_head = 4
    tf_layers = 1
    dropout = 0.20
    inp_len = 8
    pool_size = 4

    lr = 1.2e-3
    pretrained_lr = None
    weight_decay = 1e-3
    label_smoothing = 0.01
    pos_weight_power = 1.0
    epochs = 20
    threshold = 0.5

    data_dir = Path('csa_dataset.hdf5')
    cache_path = Path('dyeV8_cache/voc_full_images_64.npy')
    pretrained_path = Path('resnet32_check_point.pth')
    output_dir = Path('dyeV8_results')
    models_dir = output_dir / 'models'
    predictions_dir = output_dir / 'predictions'
    histories_dir = output_dir / 'histories'
    tables_dir = output_dir / 'tables'

    kf_split = 3
    seeds = (20287501, 20287502, 20287503)
    resume = True

    label_names = np.array([
        'ethanol', 'hexanal', 'acetic acid (DIW)', 'DIW',
        '3-methyl-1-butanol', 'nonane (DIW)', 'nonane (DMSO)',
        'acetone', 'propanol', 'toluene', 'limonene',
        'ethyl acetate', 'morpholine', 'acetaldehyde',
    ])

    for directory in (output_dir, models_dir, predictions_dir, histories_dir, tables_dir):
        directory.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_params = {
        'num_class': num_class,
        'resnet_size': resnet_size,
        'feature_dim': feature_dim,
        'dim_ff': dim_ff,
        'num_head': num_head,
        'tf_layers': tf_layers,
        'dropout': dropout,
        'img_size': img_size,
        'inp_len': inp_len,
        'pool_size': pool_size,
        'pretrained_path': str(pretrained_path),
        'lr': lr,
        'pretrained_lr': pretrained_lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'label_smoothing': label_smoothing,
        'pos_weight_power': pos_weight_power,
    }
    checkpoint_config = dict(model_params)
    params_dict = {
        'model': model_params,
        'checkpoint_config': checkpoint_config,
        'device': device,
    }

    seed_rows = []
    fold_rows = []
    class_tables = []
    for kf_fold in range(kf_split):
        DyeData = DyeDataModule(
            batch_size=batch_size, data_dir=data_dir, cache_path=cache_path,
            kf_split=kf_split, kf_fold=kf_fold, seed=seeds[0],
        )
        DyeData.setup()
        print(f'Fold {kf_fold + 1}/{kf_split}')
        print(f'Train data: {len(DyeData.train_dataset)}')
        print(f'Test data: {len(DyeData.test_dataset)}')

        fold_scores = []
        for seed in seeds:
            scores, resumed = run_seed(
                DyeData, DyeData.csa_label, params_dict, seed,
                models_dir, predictions_dir, histories_dir, resume=resume,
                label_names=label_names, threshold=threshold,
            )
            fold_scores.append(scores)
            seed_rows.append({
                'fold': kf_fold + 1, 'seed': seed, 'resumed': resumed,
                **multilabel_metrics(DyeData.csa_label, scores, threshold),
            })

        ensemble_scores = np.mean(fold_scores, axis=0)
        np.savez_compressed(
            predictions_dir / f'fold{kf_fold + 1}_ensemble_experiment_predictions.npz',
            scores=ensemble_scores, seed_scores=np.stack(fold_scores),
            labels=DyeData.csa_label, experiments=np.arange(37), seeds=np.asarray(seeds),
        )
        metrics = multilabel_metrics(DyeData.csa_label, ensemble_scores, threshold)
        fold_rows.append({
            'fold': kf_fold + 1,
            'train_interval': f'{kf_fold * 6}-{kf_fold * 6 + 12} h',
            'purge_interval': f'{kf_fold * 6 + 12}-{kf_fold * 6 + 24} h',
            'test_interval': f'{kf_fold * 6 + 24}-{kf_fold * 6 + 36} h',
            'n_physical_experiments': 37,
            'n_test_windows': len(DyeData.test_dataset),
            **metrics,
            'micro_ap_above_0.95': metrics['micro_ap'] > 0.95,
        })
        classes = per_class_metrics(
            DyeData.csa_label, ensemble_scores, label_names, threshold
        )
        classes.insert(0, 'fold', kf_fold + 1)
        classes.insert(1, 'level', 'fold_ensemble')
        class_tables.append(classes)

    seed_table = pd.DataFrame(seed_rows)
    fold_table = pd.DataFrame(fold_rows)
    class_table = pd.concat(class_tables, ignore_index=True)
    seed_table.to_csv(tables_dir / 'seed_metrics.csv', index=False)
    fold_table.to_csv(tables_dir / 'fold_metrics.csv', index=False)
    class_table.to_csv(tables_dir / 'per_class_metrics.csv', index=False)
    metric_names = [
        'micro_ap', 'macro_ap', 'micro_precision', 'macro_precision',
        'micro_recall', 'macro_recall', 'micro_f1', 'macro_f1',
        'coverage_error', 'label_ranking_loss', 'binary_accuracy',
        'exact_match_accuracy',
    ]
    pd.DataFrame({
        'metric': metric_names,
        'three_fold_mean': [fold_table[name].mean() for name in metric_names],
        'three_fold_sd': [fold_table[name].std(ddof=1) for name in metric_names],
        'minimum_fold': [fold_table[name].min() for name in metric_names],
        'maximum_fold': [fold_table[name].max() for name in metric_names],
    }).to_csv(tables_dir / 'overall_metrics.csv', index=False)
    epoch_table = pd.concat([
        pd.read_csv(histories_dir / f'fold{fold}_seed{seed}_history.csv')
        for fold in range(1, kf_split + 1) for seed in seeds
    ], ignore_index=True)
    epoch_class_table = pd.concat([
        pd.read_csv(
            histories_dir / f'fold{fold}_seed{seed}_epoch_per_class_metrics.csv'
        )
        for fold in range(1, kf_split + 1) for seed in seeds
    ], ignore_index=True)
    epoch_table.to_csv(tables_dir / 'epoch_metrics.csv', index=False)
    epoch_class_table.to_csv(tables_dir / 'epoch_per_class_metrics.csv', index=False)
    epoch_table[epoch_table.epoch == epochs].to_csv(
        tables_dir / 'final_epoch_metrics.csv', index=False,
    )
    best_loss_rows = epoch_table.groupby(['fold', 'seed']).test_loss.idxmin()
    epoch_table.loc[best_loss_rows].sort_values(['fold', 'seed']).to_csv(
        tables_dir / 'best_epoch_by_test_loss.csv', index=False,
    )
    print(fold_table.to_string(index=False))
