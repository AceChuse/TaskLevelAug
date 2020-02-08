#!/usr/bin/python3.6

import os

allclasses = False
re_val = False

dataset = 'miniImageNet'
kway = 5
trshot = 15
trquery = 6

vquery = 15
teshot = 1
tequery = 15

#task_aug = ' --task_aug Rot90'
task_aug = ''
rot90_p = 0.25
feat_aug = 'norm'

start_epoch = 0
epochs = 21 if allclasses else 60
epoch_size = 8000
batch_size = 8
parall_num = 2
allclasses = ' --allclasses' if allclasses else ''
re_val = ' --re_val' if re_val else ''
# lossf = 'slce_loss'
# eps = 0.1
lossf = 'cross_entropy'
eps = 0.

optim = 'SGD'
lr_sche = 'lambda_epoch'
lr = 0.1
wd = 5e-4
embedding = 'ResNet12'
head = 'R2D2'
seed = 0


python = 'python3'

vshot = 1
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=train'+allclasses+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+''
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here1!')

teshot = vshot = 5
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=ens_test'+allclasses+re_val+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here2!')

teshot = vshot = 1
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=ens_test'+allclasses+re_val+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here3!')


vshot = 5
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=final'+allclasses+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here1!')

teshot = vshot = 5
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=ens_testac'+allclasses+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+''
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+''
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here2!')

teshot = vshot = 1
if os.system(python + ' meta_e.py --dataset='+dataset+' --mode=ens_testac'+allclasses+' --epochs='+ str(epochs)+
             task_aug+' --rot90_p='+str(rot90_p)+' --feat_aug='+feat_aug+
             ' --start_epoch='+str(start_epoch)+' -es='+str(epoch_size)+' -b='+str(batch_size)+' -pn='+str(parall_num)+
             ' --lossf='+lossf+' --eps='+str(eps)+' --optim='+optim+' --lr_sche='+lr_sche+' --lr='+str(lr)+' --wd='+str(wd)+
             ' --embedding='+embedding+' --head='+head+' --kway='+str(kway)+' --trshot='+str(trshot)+
             ' --trquery='+str(trquery)+' --vshot='+str(vshot)+' --vquery='+str(vquery)+' --teshot='+str(teshot)+
             ' --tequery='+str(tequery)+' --seed='+str(seed)):
    raise ValueError('Here3!')