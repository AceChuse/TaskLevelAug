# -*- coding: utf-8 -*-
from utils import *

import argparse
import os
import re
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from metadatas import *
from metamodels import *
from loss import *


def get_model(embedding, head, args):
    assert args.epoch_size % args.batch_size == 0, 'epoch_size % batch_size should be equal to 0!'
    assert args.batch_size % args.parall_num == 0, 'batch_size % parall_num should be equal to 0!'
    DropBlock.rate = args.batch_size // args.parall_num

    # Choose the embedding network
    if embedding[:4] == 'CNN4':
        pool = 'max' if embedding[5:] != 'Avg' else 'avg'
        network = ProtoNetEmbedding(pool=pool).to(device)
    elif embedding == 'R2D2':
        network = R2D2Embedding().to(device)
    elif embedding[:8] == 'ResNet12':
        pool = 'max' if embedding[9:] != 'Avg' else 'avg'
        if args.dataset == 'miniImageNet':
            network = resnet12(avg_pool=False, drop_rate=args.drop_rate, dropblock_size=5, pool=pool).to(device)
            # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool=False, drop_rate=args.drop_rate, dropblock_size=2, pool=pool).to(device)
        if args.start_epoch != 0:
            network.init_num_batches_tracked(args.epoch_size / float(args.batch_size) * args.start_epoch)
    elif embedding == 'DenseNet121':
        network = densenet121(drop_rate=0., avg_pool=True).to(device)
    else:
        print("Cannot recognize the network type")
        assert (False)

    # Choose the classification head
    if head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').to(device)
    elif head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').to(device)
    elif head == 'R2D2':
        cls_head = R2D2Head().to(device)
    elif head == 'SVM':
        evalIter = 3
        printl('SVM-CS, maxIter(on eval)='+str(evalIter))
        cls_head = ClassificationHead(base_learner='SVM-CS', evalIter=evalIter).to(device)
    elif head == 'Bagging+R2D2':
        if embedding == 'ResNet12':
            cls_head = Bagging(base_learner=R2D2Head(), in_channels=640, n_channels=64, n_overlap=16)
    else:
        print("Cannot recognize the dataset type")
        assert (False)

    return (network, cls_head)


def get_datasets(name, phase, args):
    if name == 'miniImageNet':
        dataset = MiniImageNet(phase=phase, augment=args.feat_aug, rot90_p=args.rot90_p)
    elif name == 'CIFAR_FS':
        dataset = CIFAR_FS(phase=phase, augment=args.feat_aug, rot90_p=args.rot90_p)
    elif name == 'FC100':
        dataset = FC100(phase=phase, augment=args.feat_aug, rot90_p=args.rot90_p)
    else:
        print ("Cannot recognize the dataset type")
        assert(False)
    printl(dataset)

    if phase == 'train' or phase == 'final':
        for ta in args.task_aug:
            if ta == 'Dual':
                dataset = DualCategories(dataset, p=args.dual_p, std=args.dual_std)
            elif ta == 'Perm':
                dataset = PermuteChannels(dataset, p=args.perm_p)
            elif ta == 'Rot90':
                assert(args.feat_aug != 'w_rot90')
                dataset = Rot90(dataset, p=args.rot90_p, batch_size_down=8e4)
                dataset.batch_num.value += args.epoch_size * args.start_epoch
            elif ta == 'ARatio':
                dataset = AspectRatio(dataset, ratiomm=args.ratiomm, p=args.aratio_p)
            else:
                continue
            printl(dataset)

    return dataset


def get_optim(name, model, args):
    params = [{"params": model.parameters(), 'lr': args.lr, 'initial_lr': args.lr}]
    if name == 'SGD':
        return optim.SGD(params, lr=args.lr, momentum=args.momentum,
                         weight_decay=args.wd, nesterov=args.nesterov)
    if name == 'Adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.wd)


def get_scheduler(name, optimizer, args):
    start_epoch = args.start_epoch if args.start_epoch > 0 else -1

    class void(object):
        def step(self, epoch=None): pass

    if name == 'lambda_epoch':
        if args.optim == 'SGD':
            lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
        elif args.optim == 'Adam':
            lambda_epoch = lambda e: 1.0 if e < 30 else (0.5 if e < 40 else 0.25 if e < 50 else (0.125))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch, last_epoch=start_epoch)

    return void()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="This is task level augment network.")
    ## Experiment setting
    parser.add_argument('--seed', default=0, type=int,
                        help='the random seed of training.')
    parser.add_argument('--show_test', '-st', action='store_false', default=True,
                        help='show test or not.')
    parser.add_argument('--data_folder', default='../DataSets',
                        help='path of data folder.')
    parser.add_argument('--logdir', default='../Result',
                        help='directory for summaries and checkpoints.')
    parser.add_argument('--print_intrl', '-p', default=10, type=int,
                        help='print interval (default: 10).')
    parser.add_argument('--save_intrl', default=1, type=int,
                        help='save metamodel interval (default: 1).')
    parser.add_argument('--mode', '-mo', default='train', type=str,
                        help='train, final or test.')
    parser.add_argument('--allclasses', '-ac', action='store_true', default=False,
                        help='if use all training and valuation classes.')
    parser.add_argument('--re_val', '-rv', action='store_true', default=False,
                        help='whether to retest on validation set or not.')
    #parser.add_argument('--val_id', '-vi', default='val', type=str, help='the loadid of validation result.')

    ## Task Setting
    parser.add_argument('--dataset', default='CIFAR_FS', type=str,
                        help='dataset (CIFAR_FS [default], FC100 and so on).')
    parser.add_argument("--kway", "-kw", default=5, type=int,
                        help="number of classes used in classification (e.g. 5-way classification).")
    parser.add_argument("--trshot", default=5, type=int,
                        help="On training set, number of examples used for inner gradient update (K for K-shot learning).")
    parser.add_argument("--trquery", default=6, type=int,
                        help="On training set, number of examples used for inner test (K for K-query).")
    parser.add_argument("--vshot", default=5, type=int,
                        help="On validation set, number of examples used for inner gradient update (K for K-shot learning).")
    parser.add_argument("--vquery", default=15, type=int,
                        help="On validation set, number of examples used for inner test (K for K-query).")
    parser.add_argument("--teshot", default=1, type=int,
                        help="On test set, number of examples used for inner gradient update (K for K-shot learning).")
    parser.add_argument("--tequery", default=15, type=int,
                        help="On test set, number of examples used for inner test (K for K-query).")

    ## Data Augmentation
    parser.add_argument('--feat_aug', '-faug', default='norm', type=str,
                        help='If use feature level augmentation.')
    parser.add_argument('--task_aug', '-taug', default=[], nargs='+', type=str,
                        help='If use task level data augmentation.')
    parser.add_argument('--dual_p', '-dp', default=-1, type=float,
                        help='The possibility of sampling categories with dual categories.')
    parser.add_argument('--dual_std', '-dstd', default=0.1, type=float,
                        help='The std of weights for adding images from two categories.')
    parser.add_argument('--perm_p', '-pp', default=-1, type=float,
                        help='The possibility of sampling categories with permuting channels.')
    parser.add_argument('--rot90_p', '-rp', default=-1, type=float,
                        help='The possibility of sampling categories or images with rot90.')
    parser.add_argument('--aratio_p', '-arp', default=-1, type=float,
                        help='The possibility of sampling categories with aspect ratio.')
    parser.add_argument('--ratiomm', '-armm', default=[1.9, 2.1], nargs=2, type=float,
                        help='The max and min of aspect ratio.')

    ## Training setting
    parser.add_argument('--epochs', default=60, type=int,
                        help='number of total epochs to run.')
    parser.add_argument('-es','--epoch_size', type=int, default=8000,
                        help='number of epoch size per train epoch')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts).')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        help='mini-batch size (default: 8).')
    parser.add_argument('-pn', '--parall_num', default=8, type=int,
                        help='the number of parall tasks (default: 8).')
    parser.add_argument('--lossf', default='cross_entropy', type=str,
                        help='loss function of visionmodel. cross_entropy, slce_loss')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')

    ## Optimizer setting
    parser.add_argument('--optim', default='SGD', type=str,
                        help='the name of optimizer.')
    parser.add_argument('--lr_sche', default='lambda_epoch', type=str,
                        help='the name of lr scheduler.')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='the gamma of lr scheduler.')
    parser.add_argument('--lr', default=0.1, type=float,
                        help='initial learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum.')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum.')
    parser.add_argument('--wd', default=5e-4, type=float,
                        help='weight decay (default: 5e-4).')
    parser.add_argument('--drop_rate', default=0.1, type=float,
                        help='drop_rate in network.')

    ## Model setting
    # parser.add_argument('--name', default='CNN4E53', type=str, help='name of metamodel or algorithm used in experiment.')
    parser.add_argument('--embedding', type=str, default='ResNet12',
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='SVM',
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')

    args = parser.parse_args()
    args.model_name = args.embedding + '_' + args.head
    args.model_name = args.model_name + '_Taug' if args.task_aug else args.model_name

    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.logdir = os.path.join(args.logdir, args.dataset)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    args.logdir = os.path.join(args.logdir, args.model_name)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    tasks_file = str(args.kway) + 'way_' + str(args.trshot) + 'trshot'
    args.logdir = os.path.join(args.logdir, tasks_file)
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)

    if args.start_epoch == 0 and (args.mode == 'train' or args.mode == 'go_final'):
        date = re.sub('[: ]', '-', str(time.asctime(time.localtime(time.time()))))
    else:
        with open(os.path.join(args.logdir, 'model_training.txt'), "r") as f:
            date = f.read().strip('\n')
    set_resultpath(os.path.join(args.logdir, date + '.txt'))

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    (embedding, cls_head) = get_model(args.embedding, args.head, args)
    embedding.to(device)
    cls_head.to(device)

    if args.lossf == 'cross_entropy':
        lossf = F.cross_entropy
    elif args.lossf == 'slce_loss':
        lossf = SLCELoss(num_classes=args.kway, eps=args.eps)

    val_intrain = 'val_' + str(args.vshot) + 'vshot_' + str(args.vquery) + 'vquery'
    stati = MetaStatistics(lossf, loadids=['train', val_intrain], eval_mode='ensembles',
                           file=os.path.join(os.path.join(args.logdir, date)), val_intrain=val_intrain)

    num_workers = 4
    savesuffix = 'ac.pkl' if (args.allclasses or args.mode == 'final') else '.pkl'
    if args.mode == 'train' or args.mode == 'final':
        whole_net = nn.Sequential(embedding, cls_head)
        optimizer = get_optim(args.optim, whole_net, args)
        scheduler = get_scheduler(args.lr_sche, optimizer, args)

        # metamodel train
        if args.start_epoch != 0:
            path = os.path.join(args.logdir, str(args.start_epoch) + savesuffix)
            saved_models = torch.load(path)
            embedding.load_state_dict(saved_models['embedding'])
            cls_head.load_state_dict(saved_models['head'])
            print("=> loading model from '{}'".format(path))
            stati.load_buf(args.start_epoch)
        else:
            print_network(embedding, 'embedding')
            print_network(cls_head, 'cls_head')
            printl(optimizer)
            printl(args.lr_sche)
            printl(lossf)
            printl('trainloader: ' + str(args.trshot) + 'trshot_' + str(args.trquery) + 'trquery')
            if not (args.allclasses or args.mode == 'final'):
                printl('valloader: ' + str(args.vshot) + 'vshot_' + str(args.vquery) + 'vquery')
            printl('batch_size='+str(args.batch_size)+', parall_num='+str(args.parall_num))
            printl('feat_augment=' + str(args.feat_aug))
            printl('task_augment=' + str(args.task_aug))
            printl('seed=' + str(args.seed))

        _, _ = [x.train() for x in (embedding, cls_head)]

        nkshot = args.kway * args.trshot
        nkquery = args.kway * args.trquery

    if args.mode == 'train' and not args.allclasses:
        trainset = get_datasets(args.dataset, 'train', args)
        valset = get_datasets(args.dataset, 'val', args)

        trainloader = FewShotDataloader(trainset, kway=args.kway, kshot=args.trshot, kquery=args.trquery,
                                        batch_size=args.batch_size, num_workers=num_workers, epoch_size=args.epoch_size, shuffle=True)
        valloader = FewShotDataloader(valset, kway=args.kway, kshot=args.vshot, kquery=args.vquery,
                                      batch_size=args.batch_size//2, num_workers=num_workers, epoch_size=2000, shuffle=False, fixed=True)
        # TODO: If num_workers!=0 and in window, error "AttributeError: Can’t pickle local object" will raise!

        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            # Train on the training split

            # Fetch the current epoch's learning rate
            epoch_learning_rate = 0.1
            for param_group in optimizer.param_groups:
                epoch_learning_rate = param_group['lr']
            printl('Train Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, epoch_learning_rate))

            for i, batch in enumerate(trainloader(epoch), 0):
                #start_time = time.time()
                #data_shot, label_shot, data_query, label_query = [x.to(device) for x in batch]

                optimizer.zero_grad()

                batch_size = len(batch[0])
                logit_query_save, labels_query_save, loss_save = [], [], 0
                j = 0
                while j < batch_size:
                    data_shot, label_shot, data_query, label_query = [x[j:j + args.parall_num] for x in batch]
                    label_shot, label_query = label_shot.to(device), label_query.to(device)
                    _parall_num = len(label_shot)

                    data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                                 data_query.view([-1] + list(data_query.shape[-3:]))], dim=0).to(device)
                    emb_shot_query = embedding(data_shot_query)
                    emb_shot = emb_shot_query[:_parall_num * nkshot].view(_parall_num, nkshot, -1)
                    emb_query = emb_shot_query[_parall_num * nkshot:].view(_parall_num, nkquery, -1)

                    # data_shot, label_shot, data_query, label_query = [x[j:j + args.parall_num].to(device) for x in batch]
                    # _parall_num = len(label_shot)
                    #
                    # emb_shot = embedding(data_shot.view([-1] + list(data_shot.shape[-3:])))
                    # emb_shot = emb_shot.view(_parall_num, nkshot, -1)
                    #
                    # emb_query = embedding(data_query.view([-1] + list(data_query.shape[-3:])))
                    # emb_query = emb_query.view(_parall_num, nkquery, -1)

                    logit_query = cls_head(emb_query, emb_shot, label_shot, args.kway, args.trshot)

                    logit_query = logit_query.view(-1, args.kway)
                    labels_query = label_query.view(-1)
                    loss = lossf(logit_query, labels_query)
                    loss *= float(_parall_num) / batch_size

                    loss.backward(retain_graph=False)

                    logit_query_save.append(logit_query.detach())
                    labels_query_save.append(labels_query.detach())
                    loss_save += loss.data
                    j += _parall_num

                optimizer.step()
                stati.eval_batch(torch.cat(logit_query_save, dim=0),
                                 torch.cat(labels_query_save, dim=0), loss_save, 'train')
                #print(time.time() - start_time)

            scheduler.step()
            stati.liquidate(epoch, 'train')
            for p in whole_net.parameters(): p.grad = None
            di = stati.eval_total(epoch, embedding, cls_head, valloader, val_intrain)
            if use_cuda: torch.cuda.empty_cache()

            if epoch % args.save_intrl == 0:
                stati.save_buf()
                torch.save({'embedding': embedding.state_dict(), 'head': cls_head.state_dict()},
                           os.path.join(args.logdir, str(epoch) + savesuffix))
            if epoch == 2:
                with open(os.path.join(args.logdir, 'model_training.txt'), "w") as f:
                    f.write(date)
    elif (args.mode == 'train' and args.allclasses) or args.mode == 'final':

        if args.mode == 'final':
            stati.load_buf(-1)
            bests = [0]
            for loadid in stati.buf.keys():
                if loadid != 'train':
                    bests.append(stati.get_best(loadid, args.save_intrl))
            args.epochs = max(bests)

            bests = [0]
            for loadid in stati.buf.keys():
                if loadid != 'train' and 'max_epoch' in stati.buf[loadid].keys():
                    bests.append(stati.buf[loadid]['max_epoch'])
            args.epochs = max(args.epochs, max(bests))

        trainset = get_datasets(args.dataset, 'final', args)
        trainloader = FewShotDataloader(trainset, kway=args.kway, kshot=args.trshot, kquery=args.trquery,
                                        batch_size=args.batch_size, num_workers=num_workers, epoch_size=args.epoch_size, shuffle=True)
        # TODO: If num_workers!=0 and in window, error "AttributeError: Can’t pickle local object" will raise!

        for epoch in range(args.start_epoch + 1, args.epochs + 1):
            # Train on the training split

            # Fetch the current epoch's learning rate
            epoch_learning_rate = 0.1
            for param_group in optimizer.param_groups:
                epoch_learning_rate = param_group['lr']
            printl('Train Epoch: {}\tLearning Rate: {:.4f}'.format(epoch, epoch_learning_rate))

            for i, batch in enumerate(trainloader(epoch), 0):

                optimizer.zero_grad()

                batch_size = len(batch[0])
                logit_query_save, labels_query_save, loss_save = [], [], 0
                j = 0
                while j < batch_size:
                    data_shot, label_shot, data_query, label_query = [x[j:j + args.parall_num] for x in batch]
                    label_shot, label_query = label_shot.to(device), label_query.to(device)
                    _parall_num = len(label_shot)

                    data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                                 data_query.view([-1] + list(data_query.shape[-3:]))], dim=0).to(device)
                    emb_shot_query = embedding(data_shot_query)
                    emb_shot = emb_shot_query[:_parall_num * nkshot].view(_parall_num, nkshot, -1)
                    emb_query = emb_shot_query[_parall_num * nkshot:].view(_parall_num, nkquery, -1)

                    # emb_shot = embedding(data_shot.view([-1] + list(data_shot.shape[-3:])))
                    # emb_shot = emb_shot.view(_parall_num, nkshot, -1)
                    #
                    # emb_query = embedding(data_query.view([-1] + list(data_query.shape[-3:])))
                    # emb_query = emb_query.view(_parall_num, nkquery, -1)

                    logit_query = cls_head(emb_query, emb_shot, label_shot, args.kway, args.trshot)

                    logit_query = logit_query.view(-1, args.kway)
                    labels_query = label_query.view(-1)
                    loss = lossf(logit_query, labels_query)
                    loss *= float(_parall_num) / batch_size

                    loss.backward(retain_graph=False)

                    logit_query_save.append(logit_query.detach())
                    labels_query_save.append(labels_query.detach())
                    loss_save += loss.data
                    j += _parall_num

                optimizer.step()
                stati.eval_batch(torch.cat(logit_query_save, dim=0),
                                 torch.cat(labels_query_save, dim=0), loss_save, 'train')

            scheduler.step()
            stati.liquidate(epoch, 'train')
            #for p in whole_net.parameters(): p.grad = None
            if use_cuda: torch.cuda.empty_cache()
            # _, _ = [x.eval() for x in (embedding, cls_head)]
            # _, _ = [x.train() for x in (embedding, cls_head)]

            if epoch % args.save_intrl == 0:
                #stati.save_buf()
                torch.save({'embedding': embedding.state_dict(), 'head': cls_head.state_dict()},
                           os.path.join(args.logdir, str(epoch) + savesuffix))
            if epoch == 10:
                with open(os.path.join(args.logdir, 'model_training.txt'), "w") as f:
                    f.write(date)

    elif args.mode[:4] == 'test':
        savesuffix = 'ac.pkl' if args.mode == 'testac' else '.pkl'
        val = 'val_' + str(args.vshot) + 'vshot_' + str(args.vquery) + 'vquery'
        printl('testloader: ' + str(args.teshot) + 'teshot_' + str(args.tequery) + 'tequery. ' + 'savesuffix=' + savesuffix)

        stati.load_buf(-1)
        if args.allclasses:
            best = stati.get_len('train', args.save_intrl)
        elif args.re_val or val not in stati.buf.keys():
            stati.add_loadid(val, 0)
            valset = get_datasets(args.dataset, 'val', args)
            valloader = FewShotDataloader(valset, kway=args.kway, kshot=args.vshot, kquery=args.vquery,
                                          batch_size=args.batch_size//2, num_workers=num_workers, epoch_size=2000, shuffle=False, fixed=True)

            for epoch in range(0 + args.save_intrl, args.epochs + 1, args.save_intrl):
                path = os.path.join(args.logdir, str(epoch) + '.pkl')
                saved_models = torch.load(path)
                embedding.load_state_dict(saved_models['embedding'])
                cls_head.load_state_dict(saved_models['head'])
                di = stati.eval_total(epoch, embedding, cls_head, valloader, val)

            printl('')
            stati.save_buf()
            best = stati.get_best(val, args.save_intrl)
        else:
            best = stati.get_best(val, args.save_intrl)
            # best = stati.get_best(args.val_id, args.save_intrl)
            # if val != stati.val_intrain and args.mode != 'testac':
            #     valset = get_datasets(args.dataset, 'val', args)
            #     valloader = FewShotDataloader(valset, kway=args.kway, kshot=args.vshot, kquery=args.vquery,
            #                                   batch_size=args.batch_size // 2, num_workers=num_workers, epoch_size=2000,
            #                                   shuffle=False)
            #
            #     path = os.path.join(args.logdir, str(best) + savesuffix)
            #     saved_models = torch.load(path)
            #     embedding.load_state_dict(saved_models['embedding'])
            #     cls_head.load_state_dict(saved_models['head'])
            #
            #     stati.eval_extra('best=' + str(best), embedding, cls_head, valloader, val)

        if args.show_test:

            testset = get_datasets(args.dataset, 'test', args)
            testloader = FewShotDataloader(testset, kway=args.kway, kshot=args.teshot, kquery=args.tequery,
                                           batch_size=args.batch_size//2, num_workers=num_workers, epoch_size=2000, shuffle=False, fixed=True)

            path = os.path.join(args.logdir, str(best) + savesuffix)
            saved_models = torch.load(path)
            embedding.load_state_dict(saved_models['embedding'])
            cls_head.load_state_dict(saved_models['head'])

            stati.eval_extra('best=' + str(best), embedding, cls_head, testloader, args.mode +
                             '_' + str(args.teshot) + 'teshot_' + str(args.tequery) + 'tequery')

    elif args.mode[:8] == 'ens_test':
        savesuffix = 'ac.pkl' if args.mode[-2:] == 'ac' else '.pkl'
        val = 'val_' + str(args.vshot) + 'vshot_' + str(args.vquery) + 'vquery'
        printl('testloader: ' + str(args.teshot) + 'teshot_' + str(args.tequery) + 'tequery. ' + 'savesuffix=' + savesuffix)

        if args.lossf == 'cross_entropy' or args.lossf == 'slce_loss':
            activator = nn.Softmax(dim=1)
            lossf = log_nnl

        valset = get_datasets(args.dataset, 'val', args)
        valloader = FewShotDataloader(valset, kway=args.kway, kshot=args.vshot, kquery=args.vquery,
                                      batch_size=args.batch_size // 2, num_workers=num_workers, epoch_size=2000,
                                      shuffle=False, fixed=True)

        stati.load_buf(-1)
        if args.re_val or val not in stati.buf.keys():
            stati.add_loadid(val, 0)

            for epoch in range(0 + args.save_intrl, args.epochs + 1, args.save_intrl):
                path = os.path.join(args.logdir, str(epoch) + '.pkl')
                saved_models = torch.load(path)
                embedding.load_state_dict(saved_models['embedding'])
                cls_head.load_state_dict(saved_models['head'])
                di = stati.eval_total(epoch, embedding, cls_head, valloader, val)

            printl('')
            best, ensemble_list = stati.get_best_ensemble(val, args.save_intrl, valloader, activator, lossf)
        else:
            best, ensemble_list = stati.get_best_ensemble(val, args.save_intrl, valloader, activator, lossf)
        stati.save_buf()

        if args.show_test:

            def load_model(epoch):
                path = os.path.join(args.logdir, str(epoch) + savesuffix)
                saved_models = torch.load(path)
                embedding.load_state_dict(saved_models['embedding'])
                cls_head.load_state_dict(saved_models['head'])
                return embedding, cls_head

            testset = get_datasets(args.dataset, 'test', args)
            testloader = FewShotDataloader(testset, kway=args.kway, kshot=args.teshot, kquery=args.tequery,
                                           batch_size=args.batch_size // 2, num_workers=num_workers, epoch_size=2000,
                                           shuffle=False, fixed=True)

            stati.eval_extra_ensemble(args.mode + '_' + str(args.teshot) + 'teshot_' + str(args.tequery) + 'tequery',
                                      ensemble_list, load_model, testloader, activator, lossf, best=best)
    printl('')