import torch
import numpy as np
import os
import time
import torch.nn as nn
from tqdm import tqdm
from utils.util import AverageMeter, ensure_dir
from utils.metrics import Evaluator

class Tester(object):
    def __init__(self,
                 model,
                 config,
                 args,
                 test_data_loader,
                 class_name,
                 begin_time,
                 resume_file):

        # for general
        self.config = config
        self.args = args
        self.device = torch.device('cpu') if self.args.gpu == -1 else torch.device('cuda:{}'.format(self.args.gpu))
        self.class_name = class_name
        # for Test
        self.model = model.to(self.device)
        self.models = []

        self.loss = self._loss().to(self.device)

        # for time
        self.begin_time = begin_time

        # for data
        self.test_data_loader = test_data_loader

        # for resume/save path
        self.history = {
            "eval": {
                "loss": [],
                "acc": [],
                "miou": [],
                "time": [],
                "prec": [],
                "recall": [],
                "f_score": [],
            },
        }

        self.model_name = self.config.model_name

        # loading args.weight or the checkpoint-best.pth
        self.test_log_path = os.path.join(self.args.output, 'test', 'log', self.model_name,
                                          self.begin_time)
        ensure_dir(self.test_log_path)

        self.predict_path = os.path.join(self.args.output, 'test', 'predict', self.model_name,
                                         self.begin_time)
        ensure_dir(self.predict_path)

        if self.config.use_seed:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time + '_seed' + str(self.config.random_seed), 'checkpoint-best.pth')
        else:
            self.resume_ckpt_path = resume_file if resume_file is not None else \
                os.path.join(self.config.save_dir, self.model_name,
                             self.begin_time, 'checkpoint-best.pth')

        # # 将使用的模型文件名写入到文件中便于查看
        # with open(os.path.join(self.predict_path, 'checkpoint.txt'), 'w') as f:
        #     f.write(self.resume_ckpt_path)

        self.evaluator = Evaluator(self.config.nb_classes, self.device)

    def _loss(self):
        loss = nn.CrossEntropyLoss()
        return loss

    def eval_and_predict(self):
        self._resume_ckpt()

        self.model.eval()
        self.evaluator.reset()

        ave_total_loss = AverageMeter()

        with torch.no_grad():
            tic = time.time()
            for steps, (imgs, gts, filenames) in tqdm(enumerate(self.test_data_loader, start=1)):
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

                ave_total_loss.update(loss.item())
            total_time = time.time() - tic
            acc = self.evaluator.Pixel_Accuracy().cpu().detach().numpy()
            miou = self.evaluator.Mean_Intersection_over_Union().cpu().detach().numpy()
            TP, FP, FN, TN = self.evaluator.get_base_value()
            confusion_matrix = self.evaluator.get_confusion_matrix().cpu().detach().numpy()
            iou = self.evaluator.get_iou().cpu().detach().numpy()
            prec = self.evaluator.Pixel_Precision_Class().cpu().detach().numpy()
            recall = self.evaluator.Pixel_Recall_Class().cpu().detach().numpy()
            f1_score = self.evaluator.Pixel_F1_score_Class().cpu().detach().numpy()

            # display evaluation result
            print('Evaluation phase !\n'
                  'Accuracy: {:6.4f}, Loss: {:.6f}'.format(
                acc, ave_total_loss.average()))
            np.set_printoptions(formatter={'int': '{: 9}'.format})
            print('Class:    ', self.class_name, ' Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            print('IoU:      ', np.hstack((iou, np.average(iou))))
            print('Precision:', np.hstack((prec, np.average(prec))))
            print('Recall:   ', np.hstack((recall, np.average(recall))))
            print('F_Score:  ', np.hstack((f1_score, np.average(f1_score))))
            np.set_printoptions(formatter={'int': '{:14}'.format})
            print('Confusion_matrix:')
            print(confusion_matrix)

            print('Prediction Phase !\n'
                  'Total Time cost: {:.2f}s\n'
                  .format(total_time,
                          ))
        self.history["eval"]["loss"].append(ave_total_loss.average())
        self.history["eval"]["acc"].append(acc.tolist())
        self.history["eval"]["miou"].append(iou.tolist())
        self.history["eval"]["time"].append(total_time)

        self.history["eval"]["prec"].append(prec.tolist())
        self.history["eval"]["recall"].append(recall.tolist())
        self.history["eval"]["f_score"].append(f1_score.tolist())

        # Save results to log file
        print("     + Saved history of evaluation phase !")
        hist_path = os.path.join(self.test_log_path, "history1.txt")
        with open(hist_path, 'w') as f:
            f.write(str(self.history).replace("'", '"'))
            f.write('\nConfusion_matrix:\n')
            f.write(str(confusion_matrix))

            np.set_printoptions(formatter={'int': '{: 9}'.format})
            f.write('\nClass:    ' + str(self.class_name) + '  Average')
            np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
            format_iou = np.hstack((iou, np.average(iou)))
            format_prec = np.hstack((prec, np.average(prec)))
            format_recall = np.hstack((recall, np.average(recall)))
            format_f1_score = np.hstack((f1_score, np.average(f1_score)))
            f.write('\nIoU:      ' + str(format_iou))
            f.write('\nPrecision:' + str(format_prec))
            f.write('\nRecall:   ' + str(format_recall))
            f.write('\nF1_score: ' + str(format_f1_score))

    def _resume_ckpt(self):
        print("     + Loading ckpt path : {} ...".format(self.resume_ckpt_path))
        checkpoint = torch.load(self.resume_ckpt_path)
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        print("     + Model State Loaded ! :D ")
        print("     + Checkpoint file: '{}' , Loaded ! \n"
              "     + Prepare to test ! ! !"
              .format(self.resume_ckpt_path))
