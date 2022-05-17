import os
import sys
import math
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init
from utils.util import AverageMeter, ensure_dir
from tqdm import tqdm
from utils.metrics import Evaluator
import shutil
import utils.mask_gen as mask_gen
from torch.cuda.amp import autocast
from torch.cuda.amp import grad_scaler


class Trainer(object):

    def __init__(self,
                 model,
                 config,
                 args,
                 train_data_loader,
                 valid_data_loader,
                 train_unsup_data_loader0,
                 train_unsup_data_loader1,
                 begin_time,
                 resume_file=None):

        print("     + Training Start ... ...")
        # for general
        self.config = config
        self.args = args
        self.device = (self._device(self.args.gpu))
        self.model = model.to(self.device)

        self.train_data_loader = train_data_loader
        self.valid_data_loder = valid_data_loader
        self.unsupervised_train_loader_0 = train_unsup_data_loader0
        self.unsupervised_train_loader_1 = train_unsup_data_loader1

        # for time
        self.begin_time = begin_time  # part of ckpt name
        self.save_period = self.config.save_period  # for save ckpt

        self.model_name = self.config.model_name

        if self.config.use_seed:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time + '_seed' + str(self.config.random_seed))
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time + '_seed' + str(self.config.random_seed), 'log')
        else:
            self.checkpoint_dir = os.path.join(self.args.output, self.model_name,
                                               self.begin_time)
            self.log_dir = os.path.join(self.args.output, self.model_name,
                                        self.begin_time, 'log')

        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)

        # output config to log file
        log_file_path = os.path.join(self.log_dir, self.model_name + '.txt')
        self.config.write_to_file(log_file_path)

        self.history = {
            'train': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            },
            'valid': {
                'epoch': [],
                'loss': [],
                'acc': [],
                'miou': [],
                'prec': [],
                'recall': [],
                'f_score': [],
            }
        }
        # for optimize
        self.weight_init_algorithm = self.config.init_algorithm
        self.current_lr = self.config.init_lr

        # for train
        self.start_epoch = 0
        self.early_stop = self.config.early_stop  # early stop steps
        self.monitor_mode = self.config.monitor.split('/')[0]
        self.monitor_metric = self.config.monitor.split('/')[1]
        self.monitor_best = 0
        self.best_epoch = -1
        self.not_improved_count = 0
        self.monitor_iou = 0

        # resume file
        self.resume_file = resume_file
        self.resume_ = True if resume_file else False
        if self.resume_file is not None:
            with open(log_file_path, 'a') as f:
                f.write('\n')
                f.write('resume_file:' + resume_file + '\n')

        self.loss = self._loss().to(self.device)
        self.optimizer_1 = self._optimizer(lr_algorithm=self.config.lr_algorithm)
        self.optimizer_2 = self._optimizer(lr_algorithm=self.config.lr_algorithm)

        # monitor init
        if self.monitor_mode != 'off':
            assert self.monitor_mode in ['min', 'max']
            self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf

        if self.config.use_one_cycle_lr:
            self.lr_scheduler_1 = self._lr_scheduler_onecycle(self.optimizer_1)
            self.lr_scheduler_2 = self._lr_scheduler_onecycle(self.optimizer_2)
        else:
            self.lr_scheduler_1 = self._lr_scheduler_lambda(self.optimizer_1, last_epoch=self.start_epoch - 1)
            self.lr_scheduler_2 = self._lr_scheduler_lambda(self.optimizer_2, last_epoch=self.start_epoch - 1)

        # Evaluator
        self.evaluator = Evaluator(self.config.nb_classes, self.device)

    def _device(self, gpu):
        if gpu == -1:
            device = torch.device('cpu')
            return device
        else:
            device = torch.device('cuda:{}'.format(gpu))
            return device

    def _optimizer(self, lr_algorithm):
        assert lr_algorithm in ['adam', 'adamw', 'sgd']
        if lr_algorithm == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=self.current_lr,
                                   betas=(0.9, 0.999),
                                   eps=1e-08,
                                   weight_decay=self.config.weight_decay,
                                   amsgrad=False
                                   )
        elif lr_algorithm == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                  lr=self.current_lr,
                                  momentum=self.config.momentum,
                                  dampening=0,
                                  weight_decay=self.config.weight_decay,
                                  nesterov=True)
        elif lr_algorithm == 'adamw':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.current_lr,
                                    betas=(0.9, 0.999),
                                    eps=1e-08,
                                    weight_decay=self.config.weight_decay,
                                    amsgrad=False
                                    )
        return optimizer

    def _loss(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def _lr_scheduler_onecycle(self, optimizer):
        lr_scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.config.init_lr * 6,
                                                     steps_per_epoch=len(self.train_data_loader),
                                                     epochs=self.config.epochs + 1,
                                                     div_factor=6)
        return lr_scheduler

    def _lr_scheduler_lambda(self, optimizer, last_epoch):
        lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.epochs)), 0.9)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1, last_epoch=last_epoch)

        return lr_scheduler

    def _weight_init(self, m):
        classname = m.__class__.__name__
        if classname.find(' Conv') != -1:
            if self.weight_init_algorithm == 'kaiming':
                init.kaiming_normal_(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def train(self):
        if self.resume_ == False:
            # init weights
            self.model.apply(self._weight_init)
            print("     + Init weight ... Done !")
        else:
            # load the checkpoint file: resume_file
            self._resume_ckpt(resume_file=self.resume_file)
            print("     + Loading pth model file ... Done!")

        epochs = self.config.epochs
        assert self.start_epoch < epochs

        for epoch in range(self.start_epoch, epochs + 1):
            train_log = self._train_epoch_classmix(epoch)
            eval_log = self._eval_epoch(epoch)

            # lr update
            if not self.config.use_one_cycle_lr:
                if self.lr_scheduler_1 is not None:
                    self.lr_scheduler_1.step(epoch)
                    for param_group in self.optimizer_1.param_groups:
                        self.current_lr = param_group['lr']
                if self.lr_scheduler_2 is not None:
                    self.lr_scheduler_2.step(epoch)
                    for param_group in self.optimizer_2.param_groups:
                        self.current_lr = param_group['lr']

            best = False
            if self.monitor_mode != 'off':
                improved = (self.monitor_mode == 'min' and eval_log[
                    'val_' + self.monitor_metric] < self.monitor_best) or \
                           (self.monitor_mode == 'max' and eval_log['val_' + self.monitor_metric] > self.monitor_best)
                if improved:
                    self.monitor_best = eval_log['val_' + self.monitor_metric]
                    self.monitor_iou = eval_log['val_MIoU']
                    best = True
                    self.best_epoch = eval_log['epoch']
                    self.not_improved_count = 0
                else:
                    self.not_improved_count += 1

                if self.not_improved_count > self.early_stop:
                    print("     + Validation Performance didn\'t improve for {} epochs."
                          "     + Training stop :/"
                          .format(self.not_improved_count))
                    break
            if epoch % self.save_period == 0 or best == True:
                self._save_ckpt(epoch, best=best)

        # save history file
        print("     + Saving History ... ... ")
        hist_path = os.path.join(self.log_dir, 'history.txt')
        with open(hist_path, 'w') as f:
            f.write(str(self.history))

    def _train_epoch_classmix(self, epoch):
        ave_total_loss = AverageMeter()

        scaler = grad_scaler.GradScaler()
        self.evaluator.reset()

        # set model mode
        self.model.train()

        train_dataloader = iter(self.train_data_loader)
        unsupervised_dataloader_0 = iter(self.unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(self.unsupervised_train_loader_1)

        max_samples = max(len(self.train_data_loader), len(self.unsupervised_train_loader_0)) * self.config.batch_size
        niters_per_epoch = max_samples // self.config.batch_size
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        for idx in pbar:
            train_minibatch = train_dataloader.next()
            imgs = train_minibatch[0].to(self.device, non_blocking=True)
            gts = train_minibatch[1].to(self.device, non_blocking=True)

            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()

            cps_loss = 0.0
            if epoch > self.config.warmup_period:
                unsup_minibatch_0 = unsupervised_dataloader_0.next()
                unsup_minibatch_1 = unsupervised_dataloader_1.next()
                unsup_imgs_0 = unsup_minibatch_0[0].to(self.device, non_blocking=True)
                unsup_imgs_1 = unsup_minibatch_1[0].to(self.device, non_blocking=True)

                with torch.no_grad():
                    # Estimate the pseudo-label with branch#1 & supervise branch#2
                    logits_u0_tea_1 = self.model(unsup_imgs_0, step=1)
                    prob_u0_tea_1 = torch.sigmoid(logits_u0_tea_1).detach()

                    if self.config.use_mix:
                        logits_u1_tea_1 = self.model(unsup_imgs_1, step=1)
                        prob_u1_tea_1 = torch.sigmoid(logits_u1_tea_1).detach()
                    else:
                        prob_u1_tea_1 = torch.zeros_like(prob_u0_tea_1)

                    # Estimate the pseudo-label with branch#2 & supervise branch#1
                    logits_u0_tea_2 = self.model(unsup_imgs_0, step=2)
                    prob_u0_tea_2 = torch.sigmoid(logits_u0_tea_2).detach()

                    if self.config.use_mix:
                        logits_u1_tea_2 = self.model(unsup_imgs_1, step=2)
                        prob_u1_tea_2 = torch.sigmoid(logits_u1_tea_2).detach()
                    else:
                        prob_u1_tea_2 = torch.zeros_like(prob_u0_tea_2)

                    ps_u1_tea_label_1 = torch.argmax(prob_u1_tea_1, dim=1)

                batch_mix_masks = torch.zeros_like(ps_u1_tea_label_1)
                for img_i in range(unsup_imgs_0.shape[0]):
                    classes = torch.unique(ps_u1_tea_label_1[img_i], sorted=True)
                    nclasses = classes.shape[0]
                    if nclasses > 2:
                        classes = classes[torch.Tensor(
                            np.random.choice(nclasses, int((nclasses - nclasses % 2) / 2), replace=False)).long()]
                    elif nclasses == 2:
                        classes = classes[1].unsqueeze(0)
                    elif nclasses == 1:
                        continue
                    batch_mix_masks[img_i] = mask_gen.generate_class_mask(ps_u1_tea_label_1[img_i], classes)

                batch_mix_masks = batch_mix_masks.unsqueeze(1)
                unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

                # Mix teacher predictions using same mask
                # the mask pixels are either 1 or 0
                prob_cons_tea_1 = prob_u0_tea_1 * (1 - batch_mix_masks) + prob_u1_tea_1 * batch_mix_masks
                prob_cons_tea_2 = prob_u0_tea_2 * (1 - batch_mix_masks) + prob_u1_tea_2 * batch_mix_masks

                ps_label_1 = torch.argmax(prob_cons_tea_1, dim=1)
                ps_label_2 = torch.argmax(prob_cons_tea_2, dim=1)

            with autocast():
                if epoch > self.config.warmup_period:  # warmup
                    # Get student#1 prediction for mixed image
                    logits_cons_stu_1 = self.model(unsup_imgs_mixed, step=1)

                    # Get student#2 prediction for mixed image
                    logits_cons_stu_2 = self.model(unsup_imgs_mixed, step=2)

                    ps_label_1 = ps_label_1.long()
                    ps_label_2 = ps_label_2.long()

                    cps_loss = self.loss(logits_cons_stu_1, ps_label_2) + self.loss(logits_cons_stu_2, ps_label_1)

                # empirically set coefficient to 1.0
                cps_loss = cps_loss * 1.0

                # sup loss
                sup_logits_l = self.model(imgs, step=1)
                sup_logits_r = self.model(imgs, step=2)

                gts = gts.long()

                loss_sup_l = self.loss(sup_logits_l, gts)
                loss_sup_r = self.loss(sup_logits_r, gts)

                loss = loss_sup_l + loss_sup_r + cps_loss

                scaler.scale(loss).backward()
                scaler.step(self.optimizer_1)
                scaler.step(self.optimizer_2)
                scaler.update()

            pred = torch.argmax(sup_logits_l, dim=1)
            pred = pred.view(-1).long()

            label = gts.view(-1).long()

            # Add batch sample into evaluator
            self.evaluator.add_batch(label, pred)
            ave_total_loss.update(loss.item())

            if self.config.use_one_cycle_lr:
                # lr update
                if self.lr_scheduler_1 is not None:
                    self.lr_scheduler_1.step()
                    for param_group in self.optimizer_1.param_groups:
                        self.current_lr = param_group['lr']
                if self.lr_scheduler_2 is not None:
                    self.lr_scheduler_2.step()
                    for param_group in self.optimizer_2.param_groups:
                        self.current_lr = param_group['lr']

        acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
        miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
        TP, FP, FN, TN = self.evaluator.get_base_value()
        iou = self.evaluator.get_iou().cpu().detach().numpy()
        prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
        recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
        f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

        #  train log and return
        self.history['train']['epoch'].append(epoch)
        self.history['train']['loss'].append(ave_total_loss.average())
        self.history['train']['acc'].append(acc.tolist())
        self.history['train']['miou'].append(miou.tolist())

        self.history['train']['prec'].append(prec[1])
        self.history['train']['recall'].append(recall[1])
        self.history['train']['f_score'].append(f1_score[1])

        if self.config.nb_classes == 2:
            miou = iou[1]

        return {
            'epoch': epoch,
            'loss': ave_total_loss.average(),
            'acc': acc,
            'miou': miou,
            'prec': prec[1],
            'recall': recall[1],
            'f_score': f1_score[1],
        }

    def _eval_epoch(self, epoch):
        ave_total_loss = AverageMeter()
        self.evaluator.reset()
        # set model mode
        self.model.eval()

        with torch.no_grad():
            for steps, (imgs, gts, filename) in enumerate(self.valid_data_loder, start=1):
                imgs = imgs.to(self.device, non_blocking=True)
                gts = gts.to(self.device, non_blocking=True)

                # sup loss
                sup_logits_l = self.model(imgs, step=1)
                gts = gts.long()

                loss = self.loss(sup_logits_l, gts)

                pred = torch.argmax(sup_logits_l, dim=1)
                pred = pred.view(-1).long()
                label = gts.view(-1).long()

                # Add batch sample into evaluator
                self.evaluator.add_batch(label, pred)

                # update ave metrics
                ave_total_loss.update(loss.item())

            # calculate metrics
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

            print('Epoch {} validation done !'.format(epoch))
            print('lr: {:.8f}\n'
                  'MIoU: {:6.4f},       Accuracy: {:6.4f},    Loss: {:.6f},\n'
                  'Precision: {:6.4f},  Recall: {:6.4f},      F_Score: {:6.4f}'
                  .format(self.current_lr,
                          miou, acc, ave_total_loss.average(),
                          prec[1], recall[1], f1_score[1]))

        self.history['valid']['epoch'].append(epoch)
        self.history['valid']['loss'].append(ave_total_loss.average())
        self.history['valid']['acc'].append(acc.tolist())
        self.history['valid']['miou'].append(miou.tolist())
        self.history['valid']['prec'].append(prec[1])
        self.history['valid']['recall'].append(recall[1])
        self.history['valid']['f_score'].append(f1_score[1])

        if self.config.nb_classes == 2:
            miou = iou[1]

        #  validation log and return
        return {
            'epoch': epoch,
            'val_Loss': ave_total_loss.average(),
            'val_Accuracy': acc,
            'val_MIoU': miou,
            'val_Precision': prec[1],
            'val_Recall': recall[1],
            'val_F_score': f1_score[1],

        }

    def _save_ckpt(self, epoch, best):
        # save model ckpt
        state = {
            'epoch': epoch,
            'arch': str(self.model),
            'history': self.history,
            'state_dict': self.model.state_dict(),
            'monitor_best': self.monitor_best,
        }
        filename = os.path.join(self.checkpoint_dir, 'checkpoint-ep{}.pth'.format(epoch))
        best_filename = os.path.join(self.checkpoint_dir, 'checkpoint-best.pth')
        last_best_filename = os.path.join(self.checkpoint_dir,
                                          'checkpoint-ep{}-iou{:.4f}.pth'.format(epoch, self.monitor_iou))
        if best:
            # copy the last best model
            if os.path.exists(best_filename):
                shutil.copyfile(best_filename, last_best_filename)
            print("     + Saving Best Checkpoint : Epoch {}  path: {} ...  ".format(epoch, best_filename))
            torch.save(state, best_filename)
        else:
            start_save_epochs = 1
            if epoch > start_save_epochs:
                print("     + After {} epochs, saving Checkpoint per {} epochs, path: {} ... ".format(start_save_epochs,
                                                                                                      self.save_period,
                                                                                                      filename))
                torch.save(state, filename)

    def _resume_ckpt(self, resume_file):
        resume_path = os.path.join(resume_file)

        print("     + Loading Checkpoint: {} ... ".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.monitor_best = 0.0

        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Optimizer State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Start epoch {} Loaded !\n"
              "     + Prepare to run ! ! !"
              .format(resume_path, self.start_epoch))
