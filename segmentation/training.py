import argparse
import hashlib
import os
import shutil
import datetime
import sys

import torch
import torch.nn as nn
import torch.optim

from torch.optim import Adam
from torch.utils.data import DataLoader

import numpy as np
from util.util import enumerate_with_estimate
from .datasets import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset, get_Ct
from util.logconf import logging
from .model import WrapperUNet, SegmentationAugmentation

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LOSS_index = 1
METRICS_TP_index = 7
METRICS_FN_index = 8
METRICS_FP_index = 9
METRICS_SIZE = 10

class SegmentationTrainingApp:
    def __init__(self, sys_argv=None):
        """
        Initializes the SegmentationTrainingApp with command line arguments.

        Args:
            sys_argv: Command line arguments passed to the application.
        """
        self.cli_args = self.parse_arguments(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.totalTrainingSamples_count = 0
        self.training_writer = None
        self.validation_writer = None
        self.augmentation_dict = self.init_augmentation_dict()
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.segmentation_model, self.augmentation_model = self.init_model()
        self.optimizer = self.init_optimizer()

    def parse_arguments(self, sys_argv):
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
                            help='Batch size to use for training',
                            default=16,
                            type=int)
        parser.add_argument('--num-workers',
                            help='Number of worker processes for background data loading',
                            default=8,
                            type=int)
        parser.add_argument('--epochs',
                            help='Number of epochs to train for',
                            default=1,
                            type=int)
        parser.add_argument('--augmented',
                            help="Augment the training data.",
                            action='store_true',
                            default=False)
        parser.add_argument('--augment-flip',
                            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
                            action='store_true',
                            default=False)
        parser.add_argument('--augment-offset',
                            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
                            action='store_true',
                            default=False)
        parser.add_argument('--augment-scale',
                            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
                            action='store_true',
                            default=False)
        parser.add_argument('--augment-rotate',
                            help="Augment the training data by randomly rotating the data around the head-foot axis.",
                            action='store_true',
                            default=False)
        parser.add_argument('--augment-noise',
                            help="Augment the training data by randomly adding noise to the data.",
                            action='store_true',
                            default=False)

        return parser.parse_args(sys_argv)

    def init_augmentation_dict(self):
        """
        Initializes the augmentation dictionary based on the command line arguments.

        Returns:
            A dictionary containing the augmentation parameters.
        """
        augmentation_dict = {}
        if self.cli_args.augmented or self.cli_args.augment_flip:
            augmentation_dict['flip'] = True
        if self.cli_args.augmented or self.cli_args.augment_scale:
            augmentation_dict['scale'] = 0.2
        if self.cli_args.augmented or self.cli_args.augment_rotate:
            augmentation_dict['rotate'] = True
        if self.cli_args.augmented or self.cli_args.augment_noise:
            augmentation_dict['noise'] = 25.0
        if self.cli_args.augmented or self.cli_args.augment_offset:
            augmentation_dict['offset'] = 0.03
        return augmentation_dict

    def init_model(self):
        """
        Initializes the segmentation model and the augmentation model.

        Returns:
            A tuple containing the segmentation model and the augmentation model.
        """
        segmentation_model = WrapperUNet(in_channels=7, n_classes=1, depth=3, wf=4, padding=True, batch_norm=True, up_mode='upconv')
        augmentation_model = SegmentationAugmentation(**self.augmentation_dict)

        if self.use_cuda:
            log.info("Running on {} devices...".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model

    def init_optimizer(self):
        """
        Initializes the optimizer for the segmentation model.

        Returns:
            The optimizer object.
        """
        return Adam(self.segmentation_model.parameters())

    def init_training(self):
        """
        Initializes the training DataLoader.

        Returns:
            The DataLoader for the training dataset.
        """
        train_ds = TrainingLuna2dSegmentationDataset(val_stride=10, is_value_set=False, contextSlices_count=3)
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
        train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=self.cli_args.num_workers, pin_memory=self.use_cuda)
        return train_dl

    def init_validation(self):
        """
        Initializes the validation DataLoader.

        Returns:
            The DataLoader for the validation dataset.
        """
        val_ds = Luna2dSegmentationDataset(val_stride=10, is_value_set=True, contextSlices_count=3)
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
        self.validation_cadence = 5
        for epoch_index in range(1, self.cli_args.epochs + 1):
            log.info("Epoch {} of {}, {}/{} batches of size {}*{}".format(epoch_index, self.cli_args.epochs, len(train_dl), len(val_dl), self.cli_args.batch_size, (torch.cuda.device_count() if self.use_cuda else 1)))

            training_metrics = self.do_training(epoch_index, train_dl)
            self.log_metrics(epoch_index, 'trn', training_metrics)

            if epoch_index == 1 or epoch_index % self.validation_cadence == 0:
                validation_metrics = self.do_validation(epoch_index, val_dl)
                score = self.log_metrics(epoch_index, 'val', validation_metrics)
                best_score = max(score, best_score)

                self.save_model('seg', epoch_index, score == best_score)

                self.logImages(epoch_index, 'trn', train_dl)
                self.logImages(epoch_index, 'val', val_dl)

        self.training_writer.close()
        self.validation_writer.close()

    def do_training(self, epoch_index, train_dl):
        """
        Performs the training for one epoch.

        Args:
            epoch_index: The index of the current epoch.
            train_dl: The DataLoader for the training dataset.

        Returns:
            The training metrics for the epoch.
        """
        training_metrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        train_dl.dataset.shuffle_samples()

        batch_iteration = enumerate_with_estimate(train_dl, "E{} Training".format(epoch_index), start_index=train_dl.num_workers)
        for batch_index, batch_tup in batch_iteration:
            self.optimizer.zero_grad()
            loss_var = self.compute_batch_loss(batch_index, batch_tup, train_dl.batch_size, training_metrics_g)
            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += training_metrics_g.size(1)

        return training_metrics_g.to('cpu')

    def do_validation(self, epoch_index, val_dl):
        """
        Performs the validation for one epoch.

        Args:
            epoch_index: The index of the current epoch.
            val_dl: The DataLoader for the validation dataset.

        Returns:
            The validation metrics for the epoch.
        """
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            batch_iteration = enumerate_with_estimate(val_dl, "E{} Validation ".format(epoch_index), start_index=val_dl.num_workers)
            for batch_index, batch_tup in batch_iteration:
                self.compute_batch_loss(batch_index, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')

    def compute_batch_loss(self, batch_index, batch_tup, batch_size, metrics_g, classificationThreshold=0.5):
        """
        Computes the loss for a batch.

        Args:
            batch_index: The index of the current batch.
            batch_tup: The tuple containing the batch data.
            batch_size: The size of the batch.
            metrics_g: The metrics tensor to update.
            classificationThreshold: The threshold for classifying a prediction as positive.

        Returns:
            The loss tensor for the batch.
        """
        input_t, label_t, series_list, _slice_index_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)

        if self.segmentation_model.training and self.augmentation_dict:
            input_g, label_g = self.augmentation_model(input_g, label_g)

        prediction_g = self.segmentation_model(input_g)

        diceLoss_g = self.diceLoss(prediction_g, label_g)
        fnLoss_g = self.diceLoss(prediction_g * label_g, label_g)

        start_index = batch_index * batch_size
        end_index = start_index + input_t.size(0)

        with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1] > classificationThreshold).to(torch.float32)

            tp = (predictionBool_g * label_g).sum(dim=[1, 2, 3])
            fn = ((1 - predictionBool_g) * label_g).sum(dim=[1, 2, 3])
            fp = (predictionBool_g * (~label_g)).sum(dim=[1, 2, 3])

            metrics_g[METRICS_LOSS_index, start_index:end_index] = diceLoss_g
            metrics_g[METRICS_TP_index, start_index:end_index] = tp
            metrics_g[METRICS_FN_index, start_index:end_index] = fn
            metrics_g[METRICS_FP_index, start_index:end_index] = fp

        return diceLoss_g.mean() + fnLoss_g.mean() * 8

    def diceLoss(self, prediction_g, label_g, epsilon=1):
        """
        Computes the Dice loss for the predictions and labels.

        Args:
            prediction_g: The predicted tensor.
            label_g: The ground truth label tensor.
            epsilon: A small value to avoid division by zero.

        Returns:
            The Dice loss.
        """
        diceLabel_g = label_g.sum(dim=[1, 2, 3])
        dicePrediction_g = prediction_g.sum(dim=[1, 2, 3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1, 2, 3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g

    def logImages(self, epoch_index, mode_str, dl):
        """
        Logs images to the tensorboard writer.

        Args:
            epoch_index: The index of the current epoch.
            mode_str: The mode of the images ('trn' for training, 'val' for validation).
            dl: The DataLoader for the dataset.
        """
        self.segmentation_model.eval()

        images = sorted(dl.dataset.series_list)[:12]
        for series_index, series_uid in enumerate(images):
            ct = get_Ct(series_uid)

            for slice_index in range(6):
                ct_index = slice_index * (ct.hu_a.shape[0] - 1) // 5
                sample_tup = dl.dataset.getitem_fullSlice(series_uid, ct_index)

                ct_t, label_t, series_uid, ct_index = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > 0.5
                label_a = label_g.cpu().numpy()[0][0] > 0.5

                ct_t[:-1, :, :] /= 2000
                ct_t[:-1, :, :] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()

                image_a = np.zeros((512, 512, 3), dtype=np.float32)
                image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                image_a[:, :, 0] += prediction_a & (1 - label_a)
                image_a[:, :, 0] += (1 - prediction_a) & label_a
                image_a[:, :, 1] += ((1 - prediction_a) & label_a) * 0.5

                image_a[:, :, 1] += prediction_a & label_a
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(f'{mode_str}/{series_index}_prediction_{slice_index}', image_a, self.totalTrainingSamples_count, dataformats='HWC')

                if epoch_index == 1:
                    image_a = np.zeros((512, 512, 3), dtype=np.float32)
                    image_a[:, :, :] = ctSlice_a.reshape((512, 512, 1))
                    image_a[:, :, 1] += label_a

                    image_a *= 0.5
                    image_a[image_a < 0] = 0
                    image_a[image_a > 1] = 1
                    writer.add_image('{}/{}_label_{}'.format(mode_str, series_index, slice_index), image_a, self.totalTrainingSamples_count, dataformats='HWC')
                writer.flush()

    def log_metrics(self, epoch_index, mode_str, metrics_t):
        """
        Logs metrics to the console and tensorboard writer.

        Args:
            epoch_index: The index of the current epoch.
            mode_str: The mode of the metrics ('trn' for training, 'val' for validation).
            metrics_t: The metrics tensor.

        Returns:
            The recall score.
        """
        log.info("E{} {}".format(epoch_index, type(self).__name__))

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_index] + sum_a[METRICS_FN_index]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_index].mean()

        metrics_dict['percent_all/tp'] = sum_a[METRICS_TP_index] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = sum_a[METRICS_FN_index] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = sum_a[METRICS_FP_index] / (allLabel_count or 1) * 100

        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_index] / ((sum_a[METRICS_TP_index] + sum_a[METRICS_FP_index]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_index] / ((sum_a[METRICS_TP_index] + sum_a[METRICS_FN_index]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) / ((precision + recall) or 1)

        log.info(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(epoch_index, mode_str, **metrics_dict))
        log.info(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(epoch_index, mode_str + '_all', **metrics_dict))

        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'
        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)
        writer.flush()

        score = metrics_dict['pr/recall']

        return score

    def save_model(self, type_str, epoch_index, is_best=False):
        """
        Saves the model state to a file.

        Args:
            type_str: The type of the model (e.g., 'seg' for segmentation).
            epoch_index: The index of the current epoch.
            is_best: Whether this model state is the best so far.
        """
        file_path = os.path.join('data-unversioned', 'part2', 'models', self.cli_args.tb_prefix, '{}_{}_{}.{}.state'.format(type_str, self.time_str, self.cli_args.comment, self.totalTrainingSamples_count))

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {'sys_argv': sys.argv, 'time': str(datetime.datetime.now()), 'model_state': model.state_dict(), 'model_name': type(model).__name__, 'optimizer_state': self.optimizer.state_dict(), 'optimizer_name': type(self.optimizer).__name__, 'epoch': epoch_index, 'totalTrainingSamples_count': self.totalTrainingSamples_count}
        torch.save(state, file_path)

        log.info("Saved model params to {}".format(file_path))

        if is_best:
            best_path = os.path.join('data-unversioned', 'part2', 'models', self.cli_args.tb_prefix, f'{type_str}_{self.time_str}_{self.cli_args.comment}.best.state')
            shutil.copyfile(file_path, best_path)

            log.info("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

if __name__ == '__main__':
    SegmentationTrainingApp().main()
