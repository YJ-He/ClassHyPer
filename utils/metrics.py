import numpy as np
import torch

class Evaluator(object):
    """
    Accuracy assessment
    """
    def __init__(self, num_class, device):
        self.device = device
        self.num_class = num_class
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).long().to(self.device)
        self.iou = torch.zeros(self.num_class).to(self.device)
        self.accuracy_class = torch.zeros(self.num_class).to(self.device)
        self.precision_class = torch.zeros(self.num_class).to(self.device)
        self.recall_class = torch.zeros(self.num_class).to(self.device)
        self.f1_score_class = torch.zeros(self.num_class).to(self.device)
        self.TP = torch.zeros(self.num_class).to(self.device)
        self.FP = torch.zeros(self.num_class).to(self.device)
        self.FN = torch.zeros(self.num_class).to(self.device)
        self.TN = torch.zeros(self.num_class).to(self.device)

    def Pixel_Accuracy(self):
        """
        Overall Accuracy
        """
        Acc = torch.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Intersection_over_Union(self):
        """
        Mean IOU
        """
        MIoU = torch.diag(self.confusion_matrix) / (
                torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                torch.diag(self.confusion_matrix))
        self.iou = MIoU
        MIoU = torch.mean(MIoU)
        return MIoU

    def _generate_matrix(self, gt_image, pre_image):
        """
        Calculate confusion_matrix
        """
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask] + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        tem_cm = self.confusion_matrix.clone().detach()
        self.confusion_matrix = tem_cm + self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).long().to(self.device)

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_base_value(self):
        self.FP = self.confusion_matrix.sum(dim=0) - torch.diag(self.confusion_matrix)
        self.FN = self.confusion_matrix.sum(dim=1) - torch.diag(self.confusion_matrix)
        self.TP = torch.diag(self.confusion_matrix)
        self.TN = self.confusion_matrix.sum() - (self.FP + self.FN + self.TP)
        return self.TP, self.FP, self.FN, self.TN

    def get_iou(self):
        return self.iou

    def Pixel_Precision_Class(self):
        self.precision_class = self.TP / (self.TP + self.FP + 1e-8)
        return self.precision_class

    def Pixel_Recall_Class(self):
        self.recall_class = self.TP / (self.TP + self.FN + 1e-8)
        return self.recall_class

    def Pixel_F1_score_Class(self):
        self.f1_score_class = 2 * self.TP / (2 * self.TP + self.FP + self.FN)
        return self.f1_score_class

if __name__ == '__main__':
    evaluator = Evaluator(2, torch.device('cpu'))
    a = torch.tensor([1, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    b = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 1, 1])
    evaluator.add_batch(a, b)

    acc = evaluator.Pixel_Accuracy()
    miou = evaluator.Mean_Intersection_over_Union()
    TP, FP, FN, TN = evaluator.get_base_value()
    confusion_matrix1 = evaluator.get_confusion_matrix()
    iou = evaluator.get_iou()
    prec = evaluator.Pixel_Precision_Class()
    recall = evaluator.Pixel_Recall_Class()
    f1_score = evaluator.Pixel_F1_score_Class()
    print('Class:    ', 2, ' Average')
    np.set_printoptions(formatter={'float': '{: 6.6f}'.format})
    print('IoU:      ', iou)
    print('Precision:', prec)
    print('Recall:   ', recall)
    print('F_Score:  ', f1_score)
    np.set_printoptions(formatter={'int': '{:14}'.format})
    print('Confusion_matrix:')
    print(confusion_matrix1)
