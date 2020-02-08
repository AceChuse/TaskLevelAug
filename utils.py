from pylab import *
import time
import math
import copy
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

torch.backends.cudnn.benchmark = True


def fn_timer(func):
    def wrapper(*args, **kwargs):
        starttime = time.time()
        result = func(*args, **kwargs)
        print('run time:', time.time() - starttime)
        return result
    return wrapper


def print_network(model, name=None):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    if name is not None:
        printl(name)
    printl(model)
    printl("The number of parameters: {}".format(num_params))


resultpath = "result.txt"
def set_resultpath(path):
    global resultpath
    resultpath = path


def get_resultpath():
    global resultpath
    return resultpath


def printl(strin, end='\n'):
    with open(resultpath, "a") as log:
        log.write(str(strin)+end)
    print(strin,end=end)


def clear_log():
    with open(resultpath, "w") as log:
        log.write("")


def plt_curves(file, intrvl, curves, loadids, colors=['firebrick', 'forestgreen'],
               lstyles=['--', '--'], x_valu='Epoch', y_valu='Loss', y_lim=None):
    if isinstance(colors, str):
        colors = [colors] * len(curves)
    if isinstance(lstyles, str):
        lstyles = [lstyles] * len(curves)

    len_step = len(curves[0]) * intrvl
    x = np.arange(1, len_step + intrvl, intrvl)

    fig = plt.figure(num=1, figsize=(5, 4))
    plt.style.use('seaborn_grayback')
    fig.tight_layout()
    plt.xlim(0, len_step)
    if y_lim is not None:
        plt.ylim(y_lim[0], y_lim[1])
    for curve, loadid, color, lstyle in zip(curves, loadids, colors, lstyles):
        plt.plot(x, curve, label=loadid, color=color, linestyle=lstyle)
    plt.xlabel(x_valu)
    plt.ylabel(y_valu)
    plt.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(file, format='pdf')
    plt.pause(0.01)
    fig.clear()


def meta_caculate(epoch, embedding, cls_head, dataloader, lossf, outList=False):
    _, _ = [x.eval() for x in (embedding, cls_head)]

    kway = dataloader.kway
    kshot = dataloader.kshot
    kquery = dataloader.kquery
    batch_size = dataloader.batch_size

    nkshot = kway * kshot
    nkquery = kway * kquery

    loss_list = []
    accu_list = []
    logit_list = []
    for i, batch in enumerate(dataloader(epoch), 0):
        data_shot, label_shot, data_query, label_query = [x for x in batch]
        label_shot, label_query = label_shot.to(device), label_query.to(device)

        with torch.no_grad():
            data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                         data_query.view([-1] + list(data_query.shape[-3:]))], dim=0).to(device)
            emb_shot_query = embedding(data_shot_query)
            emb_shot = emb_shot_query[:batch_size * nkshot].view(batch_size, nkshot, -1)
            emb_query = emb_shot_query[batch_size * nkshot:].view(batch_size, nkquery, -1)

            # emb_shot = embedding(data_shot.view([-1] + list(data_shot.shape[-3:])))
            # emb_shot = emb_shot.view(batch_size, nkshot, -1)
            # emb_query = embedding(data_query.view([-1] + list(data_query.shape[-3:])))
            # emb_query = emb_query.view(batch_size, nkquery, -1)

        logit = cls_head(emb_query, emb_shot, label_shot, kway, kshot)

        with torch.no_grad():
            loss = lossf(logit.view(batch_size * nkquery, -1),
                         label_query.view(batch_size * nkquery), reduction='none')
            loss = loss.view(batch_size, nkquery).mean(1)
            accu = (torch.argmax(logit, 2) == label_query).type(FloatTensor).mean(1)

        loss_list.append(loss)
        accu_list.append(accu)
        if outList: logit_list.append(logit.cpu())

    loss_list = torch.cat(loss_list)
    accu_list = torch.cat(accu_list)
    if outList: logit_list = torch.cat(logit_list, dim=0).view(-1, kway)

    with torch.no_grad():
        loss_mean = loss_list.mean(0).item()
        loss_std = loss_list.std(0).item()
        loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss_list))

        accu_mean = accu_list.mean(0).item()
        accu_std = accu_list.std(0).item()
        accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu_list))

    _, _ = [x.train() for x in (embedding, cls_head)]
    return loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95, logit_list


'''
def meta_caculate_perm1(epoch, embedding, cls_head, dataloader, lossf):

    orders = (torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2, 0]),
              torch.LongTensor([2, 0, 1]), torch.LongTensor([1, 0, 2]),
              torch.LongTensor([0, 2, 1]), torch.LongTensor([2, 1, 0]))

    def permute_channels(feat, label):
        feat = torch.cat([feat[:, :, order] for order in orders], dim=1)
        label = label.repeat(1, 6)
        return feat, label

    _, _ = [x.eval() for x in (embedding, cls_head)]

    kway = dataloader.kway
    kshot = dataloader.kshot * 6
    kquery = dataloader.kquery
    batch_size = dataloader.batch_size

    nkshot = kway * kshot
    nkquery = kway * kquery

    loss_list = []
    accu_list = []
    for i, batch in enumerate(dataloader(epoch), 0):
        data_shot, label_shot, data_query, label_query = [x.to(device) for x in batch]
        data_shot, label_shot = permute_channels(data_shot, label_shot)

        with torch.no_grad():
            data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                         data_query.view([-1] + list(data_query.shape[-3:]))], dim=0)
            emb_shot_query = embedding(data_shot_query)
            emb_shot = emb_shot_query[:batch_size * nkshot].view(batch_size, nkshot, -1)
            emb_query = emb_shot_query[batch_size * nkshot:].view(batch_size, nkquery, -1)

        logit_query = cls_head(emb_query, emb_shot, label_shot, kway, kshot)

        with torch.no_grad():
            loss = lossf(logit_query.view(batch_size * nkquery, -1),
                         label_query.view(batch_size * nkquery), reduction='none')
            loss = loss.view(batch_size, nkquery).mean(1)
            accu = (torch.argmax(logit_query, 2) == label_query).type(FloatTensor).mean(1)

        loss_list.append(loss)
        accu_list.append(accu)

    loss_list = torch.cat(loss_list)
    accu_list = torch.cat(accu_list)

    with torch.no_grad():
        loss_mean = loss_list.mean(0).item()
        loss_std = loss_list.std(0).item()
        loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss_list))

        accu_mean = accu_list.mean(0).item()
        accu_std = accu_list.std(0).item()
        accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu_list))

    _, _ = [x.train() for x in (embedding, cls_head)]
    return loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95


def meta_caculate_perm2(epoch, embedding, cls_head, dataloader, lossf):

    orders = (torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2, 0]),
              torch.LongTensor([2, 0, 1]), torch.LongTensor([1, 0, 2]),
              torch.LongTensor([0, 2, 1]), torch.LongTensor([2, 1, 0]))

    _, _ = [x.eval() for x in (embedding, cls_head)]

    kway = dataloader.kway
    kshot = dataloader.kshot * 6
    kquery = dataloader.kquery * 6
    batch_size = dataloader.batch_size

    nkshot = kway * kshot
    nkquery = kway * kquery

    loss_list = []
    accu_list = []
    for i, batch in enumerate(dataloader(epoch), 0):
        data_shot, label_shot, data_query, label_query = [x.to(device) for x in batch]

        data_shot = torch.cat([data_shot[:, :, order] for order in orders], dim=1)
        label_shot = label_shot.repeat(1, 6)

        data_query = torch.cat([data_query[:, :, order] for order in orders], dim=1)

        with torch.no_grad():
            data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                         data_query.view([-1] + list(data_query.shape[-3:]))], dim=0)
            emb_shot_query = embedding(data_shot_query)
            emb_shot = emb_shot_query[:batch_size * nkshot].view(batch_size, nkshot, -1)
            emb_query = emb_shot_query[batch_size * nkshot:].view(batch_size, nkquery, -1)

        logit_query = cls_head(emb_query, emb_shot, label_shot, kway, kshot)
        logit_query = logit_query.view(batch_size, 6, -1, kway).mean(dim=1)

        with torch.no_grad():
            loss = lossf(logit_query.view(batch_size * nkquery // 6, -1),
                         label_query.view(batch_size * nkquery // 6), reduction='none')
            loss = loss.view(batch_size, nkquery // 6).mean(1)
            accu = (torch.argmax(logit_query, 2) == label_query).type(FloatTensor).mean(1)

        loss_list.append(loss)
        accu_list.append(accu)

    loss_list = torch.cat(loss_list)
    accu_list = torch.cat(accu_list)

    with torch.no_grad():
        loss_mean = loss_list.mean(0).item()
        loss_std = loss_list.std(0).item()
        loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss_list))

        accu_mean = accu_list.mean(0).item()
        accu_std = accu_list.std(0).item()
        accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu_list))

    _, _ = [x.train() for x in (embedding, cls_head)]
    return loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95


def meta_caculate_perm3(epoch, embedding, cls_head, dataloader, lossf):

    orders = (torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2, 0]),
              torch.LongTensor([2, 0, 1]), torch.LongTensor([1, 0, 2]),
              torch.LongTensor([0, 2, 1]), torch.LongTensor([2, 1, 0]))

    _, _ = [x.eval() for x in (embedding, cls_head)]

    kway = dataloader.kway
    kshot = dataloader.kshot
    kquery = dataloader.kquery
    batch_size = dataloader.batch_size

    nkshot = kway * kshot
    nkquery = kway * kquery

    loss_list = []
    accu_list = []
    for i, batch in enumerate(dataloader(epoch), 0):
        data_shot, label_shot, data_query, label_query = [x.to(device) for x in batch]

        data_shot = torch.cat([data_shot[:, :, order] for order in orders], dim=0)
        label_shot = label_shot.repeat(6, 1)

        data_query = torch.cat([data_query[:, :, order] for order in orders], dim=0)

        with torch.no_grad():
            data_shot_query = torch.cat([data_shot.view([-1] + list(data_shot.shape[-3:])),
                                         data_query.view([-1] + list(data_query.shape[-3:]))], dim=0)
            emb_shot_query = embedding(data_shot_query)
            emb_shot = emb_shot_query[:6 * batch_size * nkshot].view(6 * batch_size, nkshot, -1)
            emb_query = emb_shot_query[6 * batch_size * nkshot:].view(6 * batch_size, nkquery, -1)

        logit_query = cls_head(emb_query, emb_shot, label_shot, kway, kshot)
        #print('dffs')
        #logit_query = torch.softmax(logit_query, dim=2)
        logit_query = logit_query.view(6, batch_size, -1, kway).mean(dim=0)

        with torch.no_grad():
            loss = lossf(logit_query.view(batch_size * nkquery, -1),
                         label_query.view(batch_size * nkquery), reduction='none')
            loss = loss.view(batch_size, nkquery).mean(1)
            accu = (torch.argmax(logit_query, 2) == label_query).type(FloatTensor).mean(1)

        loss_list.append(loss)
        accu_list.append(accu)

    loss_list = torch.cat(loss_list)
    accu_list = torch.cat(accu_list)

    with torch.no_grad():
        loss_mean = loss_list.mean(0).item()
        loss_std = loss_list.std(0).item()
        loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss_list))

        accu_mean = accu_list.mean(0).item()
        accu_std = accu_list.std(0).item()
        accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu_list))

    _, _ = [x.train() for x in (embedding, cls_head)]
    return loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
'''


class MetaStatistics(object):
    colors = ['firebrick', 'forestgreen']
    lstyles = ['-', '-']
    def __init__(self, lossf, loadids, file, val_intrain='', eval_mode=None):
        self.lossf = lossf
        self.file = file
        self.mode = 'accu'
        self.form = {'loss': [], 'lstd': [], 'lci95': [], 'accu': [],
                     'astd': [], 'aci95': []}
        self.val_intrain = val_intrain
        self.eval_mode = eval_mode

        self.extra_repr = '[{epoch}] loadid: {loadid}, '
        for key in list(self.form.keys()):
            if key[-4:] == '_all':
                raise ValueError("There is '_all' in the end of " + key + '!')
            self.extra_repr += key + ': {' + key + ':.4}, '
            self.form.setdefault(key + '_all', 0.)
        self.extra_repr = self.extra_repr[:-2]
        self.form.setdefault('num', 0)
        self.form.setdefault('start_epoch', 0)
        self.form.setdefault('outList', [])
        self.buf = dict(zip(loadids, [copy.deepcopy(self.form) for _ in range(len(loadids))]))

    def eval_batch(self, out, label, loss, loadid):
        ob = self.buf[loadid]
        ob['loss_all'] += loss.data.item() * label.size(0)
        if self.mode == 'accu':
            _, pred = torch.max(out, dim=1)
            ob['accu_all'] += (pred == label).sum().data.item()
        ob['num'] += label.size(0)

    def liquidate(self, epoch, loadid):
        ob = self.buf[loadid]
        di = {'epoch': epoch, 'loadid': loadid}
        for key in ob.keys():
            if key[-4:] == '_all' and key not in ['lstd', 'lci95', 'astd', 'aci95']:
                di[key[:-4]] = ob[key] / ob['num']
                ob[key[:-4]].append(di[key[:-4]])
                ob[key] = 0.
        ob['num'] = 0
        printl((self.extra_repr).format(**di))

    def add_loadid(self, loadid, start_epoch=0):
        self.buf[loadid] = copy.deepcopy(self.form)
        self.buf[loadid]['start_epoch'] = start_epoch

    def clear_loadid(self, loadid):
        return self.buf.pop(loadid, None)

    def eval_total(self, epoch, embedding, cls_head, dataloader, loadid):
        ob = self.buf[loadid]
        di = {'epoch': epoch, 'loadid': loadid}
        if self.mode == 'accu':
            loss, lstds, lci95, accu, astds, aci95, logit = \
                meta_caculate(epoch, embedding, cls_head, dataloader, self.lossf, self.eval_mode=='ensembles')
            ob['loss'].append(loss)
            ob['lstd'].append(lstds)
            ob['lci95'].append(lci95)
            ob['accu'].append(accu)
            ob['astd'].append(astds)
            ob['aci95'].append(aci95)
            if self.eval_mode=='ensembles': ob['outList'].append(logit)
        for key in ob.keys():
            if key[-4:] == '_all':
                di[key[:-4]] = ob[key[:-4]][-1]
        printl((self.extra_repr).format(**di))
        return di

    def eval_extra(self, epoch, embedding, cls_head, dataloader, loadid):
        di = {'epoch': epoch, 'loadid': loadid}
        if self.mode == 'accu':
            loss, lstds, lci95, accu, astds, aci95, _ = \
                meta_caculate(0, embedding, cls_head, dataloader, self.lossf)
            di['loss'] = loss
            di['lstd'] = lstds
            di['lci95'] = lci95
            di['accu'] = accu
            di['astd'] = astds
            di['aci95'] = aci95
        printl((self.extra_repr).format(**di))
        return di

    def save_buf(self):
        buf = copy.copy(self.buf)
        buf['val_intrain'] = self.val_intrain
        pickle.dump(buf, open(self.file + '.pkl', 'wb'))

    def load_buf(self, epoch):
        buf = pickle.load(open(self.file + '.pkl', 'rb'))
        self.val_intrain = buf['val_intrain']
        buf.pop('val_intrain', None)
        buf.pop('start_epoch', None)
        self.buf = buf

        for loadid in self.buf.keys():
            self.buf[loadid]['start_epoch'] = 0
        if epoch != -1:
            for loadid in self.buf.keys():
                ob = self.buf[loadid]
                for key in ob.keys():
                    if key[-4:] == '_all':
                        ob[key[:-4]] = ob[key[:-4]][:epoch]

    def plt(self):
        if self.mode == 'accu':
            y_valus = ['loss', 'accu']
        colors = self.colors[:len(self.buf.keys())]
        lstyles = self.lstyles[:len(self.buf.keys())]
        for y_valu in y_valus:
            plt_curves(self.file + '-' + y_valu + '.pdf', 1,
                       curves=[np.array(self.buf[loadid][y_valu]) for loadid in self.buf.keys()],
                       loadids=self.buf.keys(), colors=colors, lstyles=lstyles, x_valu='Epoch',
                       y_valu=y_valu, y_lim=None)

    def get_best(self, loadid, save_intrl):
        ob = self.buf[loadid]
        if self.mode == 'accu':
            accu = np.array(ob['accu'])
            best = len(accu) - 1 - np.argmax(accu[::-1])
            di = {'loadid': loadid}

            di['loss'] = ob['loss'][best]
            di['lstd'] = ob['lstd'][best]
            di['lci95'] = ob['lci95'][best]
            di['accu'] = ob['accu'][best]
            di['astd'] = ob['astd'][best]
            di['aci95'] = ob['aci95'][best]

            best = (best + 1) * save_intrl + ob['start_epoch']
            di['epoch'] = 'best=' + str(best)

        printl((self.extra_repr).format(**di))
        return best

    def get_last(self, loadid, save_intrl):
        ob = self.buf[loadid]
        return (len(ob['loss']) + 1) * save_intrl + ob['start_epoch']

    '''
    @torch.no_grad()
    def get_best_ensemble1(self, loadid, save_intrl, dataloader, activator, lossf, whole=True):
        kway = dataloader.kway
        kquery = dataloader.kquery
        batch_size = dataloader.batch_size
        nkquery = kway * kquery

        label = []
        for i, batch in enumerate(dataloader(epoch=0), 0):
            data_shot, label_shot, data_query, label_query = [x for x in batch]
            label.append(label_query.view(batch_size * nkquery))
        label = torch.cat(label, dim=0).to(device)

        ob = self.buf[loadid]
        outList = ob['outList']
        accu_list = np.array(ob['accu'])
        down_order = np.argsort(accu_list)[::-1]
        best = down_order[0]
        accu_best = accu_list[best]

        if self.mode == 'accu':
            di = {'loadid': loadid}

            di['loss'], di['lstd'], di['lci95'], di['accu'], di['astd'], di['aci95'] = \
                ob['loss'][best], ob['lstd'][best], ob['lci95'][best], ob['accu'][best], \
                ob['astd'][best], ob['aci95'][best]

            di['epoch'] = 'best=' + str((best + 1) * save_intrl + ob['start_epoch'])

        if whole:
            ensemble_list = down_order
            outList = torch.cat([out.unsqueeze(dim=0) for out in outList], dim=0).to(device)
            y = activator(outList.view(-1, kway)).view(len(ensemble_list), -1, kway).mean(0)

            # caluate and print ensemble results
            loss = lossf(y, label, reduction='none')
            loss = loss.view(-1, nkquery).mean(1)

            accu = (torch.argmax(y.view(-1, nkquery, kway), 2) ==
                    label.view(-1, nkquery)).type(FloatTensor).mean(1)
            accu_mean = accu.mean(0).item()
            accu_std = accu.std(0).item()
            accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu))

            loss_mean = loss.mean(0).item()
            loss_std = loss.std(0).item()
            loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss))
            # print result

            di_e = {'loadid': loadid}
            di_e['loss'], di_e['lstd'], di_e['lci95'], di_e['accu'], di_e['astd'], di_e['aci95'] = \
                loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
            di_e['epoch'] = 'ensemble' + str(len(ensemble_list))
        else:
            di_e = copy.deepcopy(di)
            di_e['epoch'] = 'ensemble1'
            printl((self.extra_repr).format(**di_e))
            ensemble_list = [best]
            y_total = activator(outList[best].to(device))
            for n, order in enumerate(down_order[1:], 2):
                y_h = y_total + activator(outList[order].to(device))
                y = y_h / n

                accu = (torch.argmax(y.view(-1, nkquery, kway), 2) ==
                        label.view(-1, nkquery)).type(FloatTensor).mean(1)
                accu_mean = accu.mean(0).item()

                if accu_mean >= accu_best:
                    # add to list
                    ensemble_list.append(order)
                    y_total = y_h
                    accu_best = accu_mean

                    # caluate results
                    loss = lossf(y, label, reduction='none')
                    loss = loss.view(-1, nkquery).mean(1)

                    accu_std = accu.std(0).item()
                    accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu))

                    loss_mean = loss.mean(0).item()
                    loss_std = loss.std(0).item()
                    loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss))

                    # print result
                    di_e = {'loadid': loadid}
                    di_e['loss'], di_e['lstd'], di_e['lci95'], di_e['accu'], di_e['astd'], di_e['aci95'] = \
                        loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
                    di_e['epoch'] = 'ensemble' + str(len(ensemble_list))

                    printl((self.extra_repr).format(**di_e))
            printl('')

        ensemble_list = [(order + 1) * save_intrl + ob['start_epoch'] for order in ensemble_list]
        ob['max_epoch'] = max(ensemble_list)

        printl('There are '+str(len(ensemble_list)) +
               ' models in the ensemble, ' + str(ensemble_list) + '.')
        printl((self.extra_repr).format(**di))
        printl((self.extra_repr).format(**di_e))
        return ensemble_list
    '''

    @torch.no_grad()
    def get_best_ensemble(self, loadid, save_intrl, dataloader, activator, lossf, whole=True):
        kway = dataloader.kway
        kquery = dataloader.kquery
        batch_size = dataloader.batch_size
        nkquery = kway * kquery

        label = []
        for i, batch in enumerate(dataloader(epoch=0), 0):
            data_shot, label_shot, data_query, label_query = [x for x in batch]
            label.append(label_query.view(batch_size * nkquery))
        label = torch.cat(label, dim=0).to(device)

        ob = self.buf[loadid]
        outList = ob['outList']
        accu_list = np.array(ob['accu'])
        down_order = np.argsort(accu_list)[::-1]
        best = down_order[0]

        if self.mode == 'accu':
            di = {'loadid': loadid}

            di['loss'], di['lstd'], di['lci95'], di['accu'], di['astd'], di['aci95'] = \
                ob['loss'][best], ob['lstd'][best], ob['lci95'][best], ob['accu'][best], \
                ob['astd'][best], ob['aci95'][best]

            best = best + 1
            di['epoch'] = 'best=' + str(best * save_intrl + ob['start_epoch'])

        if whole:
            ensemble_list = down_order
            outList = torch.cat([out.unsqueeze(dim=0) for out in outList], dim=0).to(device)
            y = activator(outList.view(-1, kway)).view(len(ensemble_list), -1, kway).mean(0)

            # caluate and print ensemble results
            loss = lossf(y, label, reduction='none')
            loss = loss.view(-1, nkquery).mean(1)

            accu = (torch.argmax(y.view(-1, nkquery, kway), 2) ==
                    label.view(-1, nkquery)).type(FloatTensor).mean(1)
            accu_mean = accu.mean(0).item()
            accu_std = accu.std(0).item()
            accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu))

            loss_mean = loss.mean(0).item()
            loss_std = loss.std(0).item()
            loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss))
            # print result

            di_e = {'loadid': loadid}
            di_e['loss'], di_e['lstd'], di_e['lci95'], di_e['accu'], di_e['astd'], di_e['aci95'] = \
                loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
            di_e['epoch'] = 'ensemble' + str(len(ensemble_list))
        else:
            di_e = {'loadid': loadid}
            len_model = len(outList)
            ensemble_list = []
            accu_e_best = 0
            y_total = 0
            for n, order in enumerate(range(len_model - 1, -1, -1), 1):
                y_total = y_total + activator(outList[order].to(device))
                y = y_total / n

                accu = (torch.argmax(y.view(-1, nkquery, kway), 2) ==
                        label.view(-1, nkquery)).type(FloatTensor).mean(1)
                accu_mean = accu.mean(0).item()
                ensemble_list.append(order)

                # caluate results
                loss = lossf(y, label, reduction='none')
                loss = loss.view(-1, nkquery).mean(1)

                accu_std = accu.std(0).item()
                accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu))

                loss_mean = loss.mean(0).item()
                loss_std = loss.std(0).item()
                loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss))

                # print result
                di_e['loss'], di_e['lstd'], di_e['lci95'], di_e['accu'], di_e['astd'], di_e['aci95'] = \
                    loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
                di_e['epoch'] = 'ensemble' + str(len(ensemble_list))
                printl((self.extra_repr).format(**di_e))

                if accu_mean >= accu_e_best:
                    # add to list
                    ensemble_list_best = copy.deepcopy(ensemble_list)
                    accu_e_best = accu_mean
                    di_e_best = copy.deepcopy(di_e)
            printl('')
            ensemble_list = ensemble_list_best
            di_e = di_e_best

        ensemble_list = [(order + 1) * save_intrl + ob['start_epoch'] for order in ensemble_list]
        ob['max_epoch'] = max(ensemble_list)

        printl('There are '+str(len(ensemble_list)) +
               ' models in the ensemble, ' + str(ensemble_list) + '.')
        printl((self.extra_repr).format(**di))
        printl((self.extra_repr).format(**di_e))
        return best, ensemble_list

    @torch.no_grad()
    def eval_extra_ensemble(self, loadid, model_list, load_model, dataloader, activator, lossf, best=None):
        kway = dataloader.kway
        kquery = dataloader.kquery
        batch_size = dataloader.batch_size
        nkquery = kway * kquery

        if best is None: best = model_list[0]

        # calculate best results and print
        embedding, cls_head = load_model(best)
        di = {'epoch': 'best='+str(best), 'loadid': loadid}
        if self.mode == 'accu':
            loss, lstds, lci95, accu, astds, aci95, logit = \
                meta_caculate(0, embedding, cls_head, dataloader, self.lossf, outList=True)
            di['loss'], di['lstd'], di['lci95'], di['accu'], di['astd'], di['aci95'] = \
                loss, lstds, lci95, accu, astds, aci95
        printl((self.extra_repr).format(**di))

        label = []
        for i, batch in enumerate(dataloader(epoch=0), 0):
            data_shot, label_shot, data_query, label_query = [x for x in batch]
            label.append(label_query.view(batch_size * nkquery))
        label = torch.cat(label, dim=0).to(device)

        # ensemble output
        if best in model_list:
            best_i = [i for i in range(len(model_list)) if model_list[i] == best]
            assert(len(best_i) == 1, 'There are same models in the list!')
            best_i = best_i[0]
            outList = [logit]
            m_list = model_list[:best_i] + model_list[best_i+1:]
        else:
            outList = []
            m_list = model_list
        for index in m_list:
            embedding, cls_head = load_model(index)
            loss, lstds, lci95, accu, astds, aci95, logit = \
                meta_caculate(0, embedding, cls_head, dataloader, self.lossf, outList=True)
            outList.append(logit)

        outList = torch.cat([out.unsqueeze(dim=0) for out in outList], dim=0).to(device)
        y = activator(outList.view(-1, kway)).view(len(model_list), -1, kway).mean(0)

        # caluate and print ensemble results
        loss = lossf(y, label, reduction='none')
        loss = loss.view(-1, nkquery).mean(1)

        accu = (torch.argmax(y.view(-1, nkquery, kway), 2) ==
                label.view(-1, nkquery)).type(FloatTensor).mean(1)
        accu_mean = accu.mean(0).item()
        accu_std = accu.std(0).item()
        accu_ci95 = 1.96 * accu_std / np.sqrt(len(accu))

        loss_mean = loss.mean(0).item()
        loss_std = loss.std(0).item()
        loss_ci95 = 1.96 * loss_std / np.sqrt(len(loss))
        # print result

        di_e = {'loadid': loadid}
        di_e['loss'], di_e['lstd'], di_e['lci95'], di_e['accu'], di_e['astd'], di_e['aci95'] = \
            loss_mean, loss_std, loss_ci95, accu_mean, accu_std, accu_ci95
        di_e['epoch'] = 'ensemble' + str(len(model_list))

        printl((self.extra_repr).format(**di_e))
        return di_e
