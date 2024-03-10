import argparse
import datetime
import hashlib
import os
import shutil
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import sys
import numpy as np
import classification.datasets
import classification.model
from util.util import enumerate_with_estimate
from util.logconf import logging
from matplotlib import pyplot


METRICS_LABEL_index = 0
METRICS_PRED_index = 1
METRICS_PRED_P_index = 2
METRICS_LOSS_index = 3
METRICS_SIZE = 4

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)




class ClassificationTrainingApp:
    def __init__(self, sys_argv=None):
        self.cli_args = self.parse_arguments(sys_argv)
        self.training_writer = None
        self.validation_writer = None
        self.total_training_samples_count = 0
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.augmentation_dict = self.init_augmentation_dict()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model = self.init_model()
        self.optimizer = self.init_optimizer()

    @staticmethod
    def parse_arguments(sys_argv):
        """
        Parses the command line arguments.

        Args:
            sys_argv: Command line arguments passed to the application.

        Returns:
            Parsed arguments as an argparse.Namespace object.
        """
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size',
                            help='Batch size',
                            default=8,
                            type=int)
        parser.add_argument('--num-workers',
                            help='Number of workers',
                            default=8,
                            type=int)
        parser.add_argument('--epochs',
                            help='Number of epochs for training',
                            default=1,
                            type=int)
        parser.add_argument('--dataset',
                            help="What dataset are we using.",
                            action='store',
                            default='LunaDataset')
        parser.add_argument('--model',
                            help="Model classname",
                            action='store',
                            default='LunaModel')
        parser.add_argument('--malignant',
                            help="Train the model to classify benign or malignant.",
                            action='store_true',
                            default=False)

        return parser.parse_args(sys_argv)

    @staticmethod
    def init_augmentation_dict():
        """
        Initializes the augmentation dictionary.

        Returns:
            A dictionary containing augmentation parameters.
        """
        return {
            'offset': 0.1,
            'scale': 0.2,
            'rotate': True,
            'noise': 25.0,
            'flip': True
        }

    def init_model(self):
        model_cls = getattr(classification.model, self.cli_args.model)
        model = model_cls()

        if self.cli_args.finetune:
            data = torch.load(self.cli_args.finetune, map_location='cpu')
            blocks = [n for n, subm in model.named_children() if len(list(subm.parameters())) > 0]
            finetune_blocks = blocks[-self.cli_args.finetune_depth:]
            model.load_state_dict({k: v for k, v in data['model_state'].items() if k.split('.')[0] not in blocks[-1]},
                                  strict=False)
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info("Using {} !!!".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        """
        Initializes the optimizer.

        Returns:
            The initialized optimizer.
        """
        lr = 0.003 if self.cli_args.finetune else 0.001
        return SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def init_training(self):
        """
        Initializes the training DataLoader.

        Returns:
            The DataLoader for training.
        """
        ds_cls = getattr(classification.dsets, self.cli_args.dataset)

        train_ds = ds_cls(val_stride=10, is_value_set=False, ratio_int=1)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

        return train_dl

    def init_validation(self):
        """
        Initializes the validation DataLoader.

        Returns:
            The DataLoader for validation.
        """
        ds_cls = getattr(classification.datasets, self.cli_args.dataset)

        val_ds = ds_cls(val_stride=10, is_value_set=True)

        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)

        return val_dl

    def main(self):
        """
        The main entry point for the application.
        """
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        train_dl = self.init_training()
        val_dl = self.init_validation()

        best_score = 0.0
        validation_cadence = 5 if not self.cli_args.finetune else 1
        for epoch_index in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(epoch_index, self.cli_args.epochs, len(train_dl), len(val_dl), self.cli_args.batch_size, (torch.cuda.device_count() if self.use_cuda else 1)))
            training_metrics = self.do_training(epoch_index, train_dl)
            self.log_metrics(epoch_index, 'trn', training_metrics)

            if epoch_index == 1 or epoch_index % validation_cadence == 0:
                validation_metrics = self.do_validation(epoch_index, val_dl)
                score = self.log_metrics(epoch_index, 'val', validation_metrics)
                best_score = max(score, best_score)
                self.save_model('cls', epoch_index, score == best_score)

        if hasattr(self, 'training_writer'):
            self.training_writer.close()
            self.validation_writer.close()

    def do_training(self, epoch_index, train_dl):
        """
        Performs the training for one epoch.

        Args:
            epoch_index: The index of the current epoch.
            train_dl: The DataLoader for training.

        Returns:
            The training metrics.
        """
        self.model.train()
        train_dl.dataset.shuffle_samples()
        training_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)

        batch_iteration = enumerate_with_estimate(train_dl, "E{} Training".format(epoch_index), start_index=train_dl.num_workers)
        for batch_index, batch_tup in batch_iteration:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(batch_index, batch_tup, train_dl.batch_size, training_metrics_g, augment=True)
            loss_var.backward()
            self.optimizer.step()

        self.total_training_samples_count += len(train_dl.dataset)

        return training_metrics_g.to('cpu')

    def do_validation(self, epoch_index, val_dl):
        """
        Performs the validation for one epoch.

        Args:
            epoch_index: The index of the current epoch.
            val_dl: The DataLoader for validation.

        Returns:
            The validation metrics.
        """
        with torch.no_grad():
            self.model.eval()
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)

            batch_iteration = enumerate_with_estimate(val_dl, "E{} Validation ".format(epoch_index), start_index=val_dl.num_workers)
            for batch_index, batch_tup in batch_iteration:
                self.compute_batch_loss(batch_index, batch_tup, val_dl.batch_size, valMetrics_g, augment=False)

        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_index, batch_tup, batch_size, metrics_g, augment=True):
        """
        Computes the loss for a batch.

        Args:
            batch_index: The index of the current batch.
            batch_tup: The tuple containing the batch data.
            batch_size: The size of the batch.
            metrics_g: The metrics tensor to update.
            augment: Whether to apply augmentation.

        Returns:
            The loss tensor for the batch.
        """
        input_t, label_t, index_t, _, _ = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            input_g = classification.model.augment3d(input_g)

        logits_g, probability_g = self.model(input_g)
        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1], reduction="none")
        start_index = batch_index * batch_size
        end_index = start_index + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False, out=None)

        metrics_g[METRICS_LABEL_index, start_index:end_index] = index_g
        metrics_g[METRICS_PRED_index, start_index:end_index] = predLabel_g
        metrics_g[METRICS_PRED_P_index, start_index:end_index] = probability_g[:, 1]
        metrics_g[METRICS_LOSS_index, start_index:end_index] = loss_g

        return loss_g.mean()

    def log_metrics(self, epoch_index, mode_str, metrics_t, classificationThreshold=0.5):
        """
        Logs the metrics for a training or validation epoch.

        Args:
            epoch_index: The index of the current epoch.
            mode_str: The mode of the metrics ('trn' for training, 'val' for validation).
            metrics_t: The metrics tensor.
            classificationThreshold: The threshold for classifying a prediction as positive.

        Returns:
            The score for the epoch.
        """
        log.info("E{} {}".format(epoch_index, type(self).__name__))

        if self.cli_args.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'

        negLabel_mask = metrics_t[METRICS_LABEL_index] == 0
        negPred_mask = metrics_t[METRICS_PRED_index] == 0

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask

        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_index].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_index, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_index, posLabel_mask].mean()
        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = truePos_count / np.float64(truePos_count + falsePos_count)
        recall = metrics_dict['pr/recall'] = truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0)
        tpr = (metrics_t[None, METRICS_PRED_P_index, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_index, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:] - fpr[:-1]
        tp_avg = (tpr[1:] + tpr[:-1]) / 2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(("E{} {:8} {loss/all:.4f} loss, " + "{correct/all:-5.1f}% correct, " + "{pr/precision:.4f} precision, " + "{pr/recall:.4f} recall, " + "{pr/f1_score:.4f} f1 score, " + "{auc:.4f} auc").format(epoch_index, mode_str, **metrics_dict))
        log.info(("E{} {:8} {loss/neg:.4f} loss, " + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})").format(epoch_index, mode_str + '_' + neg, neg_correct=neg_correct, neg_count=neg_count, **metrics_dict))
        log.info(("E{} {:8} {loss/pos:.4f} loss, " + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})").format(epoch_index, mode_str + '_' + pos, pos_correct=pos_correct, pos_count=pos_count, **metrics_dict))
        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            writer.add_scalar(key, value, self.total_training_samples_count)

        fig = pyplot.figure()
        pyplot.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.total_training_samples_count)

        writer.add_scalar('auc', auc, self.total_training_samples_count)
        bins = np.linspace(0, 1)

        writer.add_histogram('label_neg', metrics_t[METRICS_PRED_P_index, negLabel_mask], self.total_training_samples_count, bins=bins)
        writer.add_histogram('label_pos', metrics_t[METRICS_PRED_P_index, posLabel_mask], self.total_training_samples_count, bins=bins)

        if not self.cli_args.malignant:
            score = metrics_dict['pr/f1_score']
        else:
            score = metrics_dict['auc']

        return score

    def save_model(self, type_str, epoch_index, is_best=False):
        """
        Saves the model state.

        Args:
            type_str: The type of the model (e.g., 'cls' for classification).
            epoch_index: The index of the current epoch.
            is_best: Whether this model state is the best so far.
        """
        file_path = os.path.join('data-unversioned', 'part2', 'models', self.cli_args.tb_prefix, '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, self.total_training_samples_count))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {'model_state': model.state_dict(), 'model_name': type(model).__name__, 'optimizer_state': self.optimizer.state_dict(), 'optimizer_name': type(self.optimizer).__name__, 'epoch': epoch_index, 'total_training_samples_count': self.total_training_samples_count}
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if is_best:
            best_path = os.path.join('data-unversioned', 'part2', 'models', self.cli_args.tb_prefix, '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, 'best'))
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':
    ClassificationTrainingApp().main()
