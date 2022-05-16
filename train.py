import datetime
import argparse
import torch
import random
import numpy as np
from configs.config import MyConfiguration
from Trainer import Trainer
from Tester import Tester
from data.dataset_list import MyDataset
from torch.utils.data import DataLoader
from models import CPS_Network

def for_train(model,
              config,
              args,
              train_data_loader,
              train_unsup_data_loader0,
              train_unsup_data_loader1,
              valid_data_loader,
              begin_time,
              resume_file):
    myTrainer = Trainer(model=model, config=config, args=args,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      train_unsup_data_loader0=train_unsup_data_loader0,
                      train_unsup_data_loader1=train_unsup_data_loader1,
                      begin_time=begin_time,
                      resume_file=resume_file)

    myTrainer.train()
    print(" Training Done ! ")


def for_test(model, config, args, test_data_loader, class_name, begin_time, resume_file):
    myTester = Tester(model=model, config=config, args=args,
                    test_data_loader=test_data_loader,
                    class_name=class_name,
                    begin_time=begin_time,
                    resume_file=resume_file)

    myTester.eval_and_predict()
    print(" Evaluation Done ! ")


def main(config, args):
    # model initialization
    model = CPS_Network.FCNs_CPS(in_ch=config.input_channel, out_ch=config.nb_classes, backbone='vgg16_bn',
                                 pretrained=True)

    if hasattr(model, 'name'):
        config.config.set("Directory", "model_name", model.name + '_classhyper')

    # obtain the maximum number of samples
    temp_datset_sup = MyDataset(config=config, args=args, subset='train')
    temp_datset_unsup = MyDataset(config=config, args=args, subset='train_unsup')
    l_sup = len(temp_datset_sup)
    l_unsup = len(temp_datset_unsup)
    max_samples = max(l_sup, l_unsup)
    del temp_datset_unsup, temp_datset_sup

    # initialize the training dataset
    train_dataset = MyDataset(config=config, args=args, subset='train', file_length=max_samples)
    train_unsup_dataset = MyDataset(config=config, args=args, subset='train_unsup', file_length=max_samples)
    valid_dataset = MyDataset(config=config, args=args, subset='val')
    test_dataset = MyDataset(config=config, args=args, subset='test')

    # initialize the training Dataloader
    train_data_loader = DataLoader(dataset=train_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.threads,
                                   drop_last=True)
    train_unsup_data_loader0 = DataLoader(dataset=train_unsup_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=args.threads,
                                          drop_last=True)
    train_unsup_data_loader1 = DataLoader(dataset=train_unsup_dataset,
                                          batch_size=config.batch_size,
                                          shuffle=True,
                                          pin_memory=True,
                                          num_workers=args.threads,
                                          drop_last=True)

    valid_data_loader = DataLoader(dataset=valid_dataset,
                                   batch_size=config.batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   num_workers=args.threads,
                                   drop_last=False)
    test_data_loader = DataLoader(dataset=test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.threads,
                                  drop_last=False)
    begin_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if config.use_gpu:
        model = model.cuda(device=args.gpu)

    for_train(model=model, config=config, args=args,
              train_data_loader=train_data_loader,
              valid_data_loader=valid_data_loader,
              train_unsup_data_loader0=train_unsup_data_loader0,
              train_unsup_data_loader1=train_unsup_data_loader1,
              begin_time=begin_time,
              resume_file=args.weight)

    for_test(model=model, config=config, args=args,
             test_data_loader=test_data_loader,
             class_name=test_dataset.class_names,
             begin_time=begin_time,
             resume_file=None
             )


if __name__ == '__main__':
    config = MyConfiguration('configs/config.cfg')

    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument('-input', metavar='input', type=str, default=config.root_dir,
                        help='root path to directory containing input images, including train & valid & test')
    parser.add_argument('-output', metavar='output', type=str, default=config.save_dir,
                        help='root path to directory containing all the output, including predictions, logs and ckpt')
    parser.add_argument('-weight', metavar='weight', type=str, default=None,
                        help='path to ckpt which will be loaded')
    parser.add_argument('-threads', metavar='threads', type=int, default=2,
                        help='number of thread used for DataLoader')
    parser.add_argument('-is_test', action='store_true', default=False,
                        help='in train mode, is_test=False')
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
