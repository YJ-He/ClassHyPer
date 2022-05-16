import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from Tester import Tester
from data.dataset_list import MyDataset
from torch.utils.data import DataLoader
from models import CPS_Network

def for_test(model, config, args, test_data_loader, class_name, begin_time, resume_file):

    myTester = Tester(model=model, config=config, args=args,
                    test_data_loader=test_data_loader,
                    class_name=class_name,
                    begin_time=begin_time,
                    resume_file=resume_file)
    myTester.eval_and_predict()
    print(" Evaluation Done ! ")

def main(config, args):
    model = CPS_Network.FCNs_CPS(in_ch=config.input_channel, out_ch=config.nb_classes, backbone='vgg16_bn', pretrained=True)

    if hasattr(model, 'name'):
        config.config.set("Directory", "model_name", model.name+'_'+config.mix_algorithm)

    test_dataset = MyDataset(config=config, args=args, subset='test')

    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size * 4,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.threads,
                                  drop_last=False)

    begin_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    begin_time = 'test-' + begin_time

    if config.use_gpu:
        model = model.cuda(device=args.gpu)

    for_test(model=model, config=config, args=args,
                   test_data_loader=test_data_loader,
                   class_name=test_dataset.class_names,
                   begin_time=begin_time,
                   resume_file=args.weight,
                   )

if __name__ == '__main__':
    config = MyConfiguration('configs/config.cfg')

    pathCkpt = r'F:\WHU\checkpoint-best.pth'

    parser = argparse.ArgumentParser(description="Model Evaluation")

    parser.add_argument('-input', metavar='input', type=str, default=config.root_dir,
                        help='root path to directory containing input images, including train & valid & test')
    parser.add_argument('-output', metavar='output', type=str, default=config.save_dir,
                        help='root path to directory containing all the output, including predictions, logs and ckpt')
    parser.add_argument('-weight', metavar='weight', type=str, default=pathCkpt,
                        help='path to ckpt which will be loaded')
    parser.add_argument('-threads', metavar='threads', type=int, default=2,
                        help='number of thread used for DataLoader')
    parser.add_argument('-is_test', action='store_true', default=True,
                        help='in test mode, is_test=True')
    if config.use_gpu:
        parser.add_argument('-gpu', metavar='gpu', type=int, default=0,
                            help='gpu id to be used for prediction')
    else:
        parser.add_argument('-gpu', metavar='gpu', type=int, default=-1,
                            help='gpu id to be used for prediction')

    args = parser.parse_args()

    if config.use_seed:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    else:
        torch.backends.cudnn.benchmark = True

    main(config=config, args=args)
