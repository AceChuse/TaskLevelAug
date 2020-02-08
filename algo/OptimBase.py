#!/usr/bin/python3.6

import torch.nn as nn
from torch.autograd import Variable, grad

import os
import numpy as np

from utils import *
from algo.parallel import to_parall, NonFineTune


class MetaNet(nn.Module):
    def __init__(self, net, optim, parall_num=1, save_path=None, save_iter=10, print_iter=100):
        super(MetaNet, self).__init__()
        self.net = net.to(device)
        self.optim = optim
        self.parall_num = parall_num

        if save_path is None:
            self.save_iter = np.inf
        else:
            self.save_path = save_path + '/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            self.save_iter = save_iter
        self.print_iter = print_iter

    def to_var(self, data):
        return Variable(data[0]).to(device), Variable(data[1]).to(device), \
               Variable(data[2]).to(device), Variable(data[3]).to(device)

    def calculate(self, feata, labela, featb, labelb):
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num],
                               featb[i:i + self.parall_num]).item() / batch_size
            lossbn, loss_p = self.net.inner_test(labelb[i:i + self.parall_num])
            lossbn /= batch_size
            if loss_p is not None:
                loss_p /= batch_size
                (lossbn + loss_p).backward(retain_graph=False)
            else:
                lossbn.backward(retain_graph=False)
            lossb += lossbn.item()
            i += self.parall_num
        return lossb

    def fit(self, trainset, valset, iters, resume_itr=0):
        lossb_all = 0.
        valset = iter(valset)
        self.valset = valset
        if resume_itr == 0:
            lossb_list = []
            lossbv_list = []
        else:
            list_len = resume_itr // self.save_iter
            lossb_list = list(np.load(self.save_path + 'lossb_train.npy'))[:list_len]
            lossbv_list = list(np.load(self.save_path + 'lossb_val.npy'))[:list_len]
        for itr, data in enumerate(trainset, resume_itr + 1):
            #starttime = time.time()
            feata, labela, featb, labelb = self.to_var(data)

            self.optim.zero_grad()
            lossb = self.calculate(feata, labela, featb, labelb)
            lossb_all += lossb

            for param in self.net.parameters():
                param.grad.data.clamp_(-10, 10)

            if itr % self.save_iter == 0 and hasattr(self.optim, 'see'):
                self.optim.see()
            else:
                self.optim.step()

            #print(time.time() - starttime)

            if itr % self.print_iter == 0:
                printl('[%d]: %.4f' % (itr, lossb_all / self.print_iter) )
                lossb_all = 0.

            if itr == 1 or (itr - 1) % self.save_iter == 0:
                lossb_list.append(lossb)
                lossbv = self.val(valset)
                lossbv_list.append(lossbv)

            if itr % self.save_iter == 0:
                np.save(self.save_path + 'lossb_train', np.array(lossb_list))
                np.save(self.save_path + 'lossb_val', np.array(lossbv_list))

                torch.save(self.net.state_dict(),
                       self.save_path + '_i' + str(itr) + '.pkl')
                np.save(self.save_path + 'last_iter', np.array(itr, dtype=np.int))
                if itr == self.save_iter:
                    filepath, date = os.path.split(get_resultpath())
                    with open(os.path.join(filepath, 'model_training.txt'), "w") as f:
                        f.write(date[:-4])

            if use_cuda: torch.cuda.empty_cache()

            if itr >= iters:
                break

    def val(self, valset):
        self.eval()
        feata, labela, featb, labelb = self.to_var(valset.__next__())
        lossb = 0.
        batch_size = len(labela)
        i = 0
        while i < batch_size:
            self.net.inner_fit(feata[i:i + self.parall_num], labela[i:i + self.parall_num],
                               featb[i:i + self.parall_num]).item() / batch_size
            lossbn, loss_p = self.net.inner_test(labelb[i:i + self.parall_num])
            lossbn /= batch_size
            lossb += lossbn.item()
            i += self.parall_num

        self.train()
        return lossb

    def test(self, testset, classify=False, max_inter=100000):
        lossb_list = []
        accub_list = []
        num_points = 0
        for itr, data in enumerate(testset):
            feata, labela, featb, labelb = self.to_var(data)

            batch_size = len(labela)
            num_points += batch_size
            i = 0
            while i < batch_size:
                self.net.inner_fit(feata[i:i+self.parall_num], labela[i:i+self.parall_num],
                                   featb[i:i + self.parall_num])
                if classify:
                    lossbs, accubs = self.net.inner_test(labelb[i:i + self.parall_num],
                                                         classify=True, reduce=False)
                    lossbs = [float(lossb) for lossb in lossbs]
                    accub_list.extend([float(accub) for accub in accubs])
                else:
                    lossbs = [float(lossb) for lossb in
                              self.net.inner_test(labelb[i:i+self.parall_num],
                                                  reduce=False)]
                lossb_list.extend(lossbs)
                i += self.parall_num

            if itr >= max_inter:
                break
        metalosses = np.array(lossb_list)
        means = np.mean(metalosses, 0)
        stds = np.std(metalosses, 0)
        ci95 = 1.96 * stds / np.sqrt(num_points)
        if classify:
            metaaccus = np.array(accub_list)
            means_accu = np.mean(metaaccus, 0)
            stds_accu = np.std(metaaccus, 0)
            ci95_accu = 1.96 * stds_accu / np.sqrt(num_points)
            return means, stds, ci95, means_accu, stds_accu, ci95_accu
        else:
            return means, stds, ci95


def module2parall(module):
    if isinstance(module, NonFineTune):
        return module
    module_parall = to_parall(module)
    for name, _module in module.named_children():
        module_parall.add_module(name, module2parall(_module))
    return module_parall


def finetune_list(module, module_parall):
    if isinstance(module, NonFineTune):
        return []
    modules = []
    modules_parall = []
    for _module, _module_parall in zip(module.children(), module_parall.children()):
        modules.append(_module)
        modules_parall.append(_module_parall)
        ms, ms_p = finetune_list(_module, _module_parall)
        modules.extend(ms)
        modules_parall.extend(ms_p)
    return modules, modules_parall


def parall_parameters(module, module_parall):
    if isinstance(module, NonFineTune):
        return [], []
    mdict = []
    pnames = []
    for _module, _module_parall in zip(module.children(), module_parall.children()):
        for name, p in _module.named_parameters():
            if not 'nonft' in name:
                mdict.append(_module_parall)
                pnames.append(name)
        md, pn = parall_parameters(_module, _module_parall)
        mdict.extend(md)
        pnames.extend(pn)
    return mdict, pnames


class _OptimBase(nn.Module):
    def __init__(self, inner_num, inner_lr, lossf, layers):
        super(_OptimBase, self).__init__()
        feature_embedding = []
        if isinstance(layers, nn.Module):
            self.net = layers
        else:
            for i in range(len(layers)):
                if isinstance(layers[i], NonFineTune):
                    feature_embedding.append(layers[i])
                else:
                    layers = layers[i:]
                    break
            self.net = nn.Sequential(*layers)
        self.feature_embedding = nn.Sequential(*feature_embedding)if feature_embedding else None

        self.inner_num = inner_num
        self.inner_lr = inner_lr
        self.lossf = lossf
        self.net_C = (module2parall(self.net),)
        self.mlist, self.mlist_parall = finetune_list(self.net, self.net_C[0])
        self.mdict, self.pnames = parall_parameters(self.net, self.net_C[0])

        self.len_p = len(self.pnames)
        self.inner_plist = [None] * self.len_p
        self.mdict = tuple(self.mdict)
        self.pnames = tuple(self.pnames)

    def init_inner(self, num):
        for m, m_p in zip(self.mlist, self.mlist_parall):
            m_p.get_parameters(m, num)
        for i, m, pn in zip(range(self.len_p), self.mdict, self.pnames):
            self.inner_plist[i] = getattr(m, pn)

    def inner_fit(self, feata, labela, featb):
        self.featb = featb
        self.feata_size = feata.size()
        self.num = self.feata_size[0]
        self.nkshot = self.feata_size[1]
        feata = feata.transpose(0, 1).contiguous()
        labela_size = list(labela.size())
        labela_size = [labela_size[0] * labela_size[1]] + labela_size[2:]
        labela = labela.transpose(0, 1).contiguous()
        labela = labela.view(*labela_size)
        feata = self.abstract(feata)

        self.init_inner(self.num)
        self.net_C[0].inner_train()
        for i in range(self.inner_num):
            output = self.net_C[0](feata)
            if i == 0:
                output_size = list(output.size())
                output_size = [output_size[0] * output_size[1]] + output_size[2:]
            loss = self.num * self.lossf(output.view(*output_size), labela)

            grads = grad(loss, self.inner_plist, create_graph=self.training)
            self.inner_update(grads)
        return loss

    def inner_update(self, grads):
        pass

    def abstract(self, feat):
        if self.feature_embedding is not None:
            feat_size = list(feat.size())
            feat = feat.view(*([feat_size[0] * feat_size[1]] + feat_size[2:]))
            output = self.feature_embedding(feat)
            out_size = list(output.size())
            return output.view(*(feat_size[:2] + out_size[1:]))
        else:
            return feat

    def inner_test(self, labelb, classify=False, reduce=True):
        featb = self.featb
        featb_size = featb.size()
        self.nkquery = featb_size[1]
        featb = featb.transpose(0, 1).contiguous()
        labelb_size = list(labelb.size())
        labelb_size = [labelb_size[0] * labelb_size[1]] + labelb_size[2:]
        labelb = labelb.transpose(0, 1).contiguous()
        labelb = labelb.view(*labelb_size)
        featb = self.abstract(featb)

        self.net_C[0].inner_eval()
        output = self.net_C[0](featb)
        output_size = list(output.size())
        output_size = [output_size[0] * output_size[1]] + output_size[2:]
        if reduce:
            return self.num * self.lossf(output.view(*output_size), labelb), None
        else:
            loss = self.lossf(output.view(*output_size), labelb, reduction='none')
            loss = loss.view(-1, self.num)
            if classify:
                self.label = self.label.view(self.label_size[0] // self.num, self.num)
                c = (torch.argmax(output, 2) == self.label).to(device)
                return loss.mean(0), c.mean(0)
            else:
                return loss.mean(0)

    def forward(self, feata, labela, featb):
        self.inner_fit(feata, labela, featb)
        featb_size = featb.size()
        self.num = featb_size[0]
        self.nkquery = featb_size[1]
        featb = featb.transpose(0, 1).contiguous()
        featb = self.abstract(featb)
        self.net_C[0].inner_eval()
        return self.net_C[0](featb).transpose(0, 1).contiguous()

    def train(self, mode=True):
        self.net_C[0].train(mode)
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


class MAML(_OptimBase):
    def __init__(self, inner_num, inner_lr, lossf, layers):
        super(MAML, self).__init__(inner_num, inner_lr, lossf, layers)

    def inner_update(self, grads):
        for i, g, m, pn in zip(range(self.len_p), grads, self.mdict, self.pnames):
            p = getattr(m, pn) - self.inner_lr * g
            setattr(m, pn, p)
            self.inner_plist[i] = p

    def extra_repr(self):
        s = ('inner_lr={inner_lr}, inner_num={inner_num}, ')
        return s.format(**self.__dict__)