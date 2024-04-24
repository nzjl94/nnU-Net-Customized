import json
import pickle
import nibabel as nib
import numpy as np
from time import time
import argparse
import importlib

from torch import nn
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from augmentation.generators import prepare_generators

from training.utils import *
from training.loss import *
from training.generic_UNet import Robust_UNet


class DecorLoss(nn.Module):
    def __init__(self):
        super(DecorLoss, self).__init__()

    def forward(self, data):
        data1 = data.view(data.shape[0],data.shape[1]*data.shape[2]*data.shape[3]*data.shape[4])
        data2 = torch.transpose(data1,0,-1) 
        mean = torch.mean(data1)

        cov = torch.matmul((data2-mean),(data1-mean))

        frob_norm = torch.mean(torch.square(cov))
        eye = torch.eye(cov.shape[0], dtype=cov.dtype)
        if data.is_cuda:
            eye = eye.cuda()
        else:
            eye = eye.cpu()

        corr_diag_sqr = torch.mean(torch.square(cov * eye))
        loss = 0.5 * (frob_norm - corr_diag_sqr)

        return loss


def run_iteration(tr_gen,network,optimizer,loss,amp_grad_scaler,fp16=True,do_backprop=True, weight_recon = 1.0,weight_latent = 1.,weight_decor = 1.0):

    data_dict = next(tr_gen)
    data = data_dict['data'].detach().cpu().numpy()[:,:3,:,:,:]
    target = data_dict['data'].detach().cpu().numpy()[:,3,:,:,:]
    target = np.reshape(target,(target.shape[0],1,target.shape[1],target.shape[2],target.shape[3]))

    data = maybe_to_torch(data)
    target = maybe_to_torch(target)

    if torch.cuda.is_available():
        data = data.cuda(0, non_blocking=True)
        target = target.cuda(0, non_blocking=True)

    optimizer.zero_grad()

    loss_LAT = torch.nn.MSELoss()
    loss_RECON = torch.nn.MSELoss()
    loss_DECOR = DecorLoss()

    if fp16:
        with autocast():
            output,reconstruction,latent,latent_ae = network(data,target)
            del data
            l1 = loss(output, target)
            l2 = weight_recon*loss_RECON(reconstruction,target)
            l3 = weight_latent*loss_LAT(latent,latent_ae)
            l4 = weight_decor*loss_DECOR(latent_ae)
            #print(l1,l2,l3,l4)
            l = l1 + l2 + l3 + l4

        if do_backprop:
            amp_grad_scaler.scale(l).backward()
            amp_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
    else:
        output,reconstruction,latent,latent_ae = network(data,target)
        del data
        l1 = loss(output, target)
        l2 = weight_recon*loss_RECON(reconstruction,target)
        l3 = weight_latent*loss_LAT(latent,latent_ae)
        l4 = weight_decor*loss_DECOR(latent_ae)
        #print(l1,l2,l3,l4)
        l = l1 + l2 + l3 + l4

        if do_backprop:
            l.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            optimizer.step()


    del target

    return l.detach().cpu().numpy(),l1.detach().cpu().numpy(),l2.detach().cpu().numpy(),l3.detach().cpu().numpy(),l4.detach().cpu().numpy()


def save_checkpoint(fname, network, optimizer, amp_grad_scaler,epoch,lr,all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4, best_epoch, best_loss):

    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    print("saving checkpoint...",flush=True)

    save_this = {
        'epoch': epoch + 1,
        'learning_rate': lr,
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_grad_scaler': amp_grad_scaler.state_dict(),
        'plot_stuff': (all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4),
        'best_stuff' : (best_epoch, best_loss)}

    torch.save(save_this, fname)
    print("saving done",flush=True)


########################################################################
##############    READ CONFIG FILES                    #################
########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
args = parser.parse_args()
config_train = args.config


with open(config_train, 'r') as config_file:
    configs = json.load(config_file)

with open(configs['plans_file_path'], 'rb') as plans_file:
    plans = pickle.load(plans_file)

########################################################################
##########                    PREPARE NETWORK             ##############
########################################################################

network_type = configs['network_type']
assert network_type in ['2d', '3d_lowres', '3d_fullres', '3d_cascade_fullres'], "Incorrect network type!"

possible_stages = list(plans['plans_per_stage'].keys())

if (network_type == '3d_cascade_fullres' or network_type == "3d_lowres") and len(possible_stages) == 1:
    raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. Run 3d_fullres.")

if network_type == '2d' or network_type == "3d_lowres":
    stage = 0
else:
    stage = possible_stages[-1]

if (network_type == '2d' or len(possible_stages) > 1) and not network_type == '3d_lowres':
    batch_dice = True
else:
    batch_dice = False

batch_size = plans['plans_per_stage'][stage]['batch_size']

if '3d' in network_type:
    conv_op = nn.Conv3d
    dropout_op = nn.Dropout3d
    norm_op = nn.InstanceNorm3d
else:
    conv_op = nn.Conv2d
    dropout_op = nn.Dropout2d
    norm_op = nn.InstanceNorm2d

norm_op_kwargs = {'eps': 1e-5, 'affine': True}
dropout_op_kwargs = {'p': 0, 'inplace': True}
net_nonlin = nn.LeakyReLU
net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}


stage_plans = plans['plans_per_stage'][stage]
patch_size = np.array(stage_plans['patch_size']).astype(int)
net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']
base_num_features = plans['base_num_features']

num_input_channels = plans['num_modalities'] -1   ## plik dawki jest zwracany przez generator w danych
num_input_channels_ae = 1
num_classes = 1 

if 'conv_per_stage' in plans.keys():
    conv_per_stage = plans['conv_per_stage']
else:
    conv_per_stage = 2

deep_supervision = False

network = Robust_UNet(num_input_channels,num_input_channels_ae, base_num_features, num_classes,
                      len(net_num_pool_op_kernel_sizes),
                      conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                      dropout_op_kwargs,
                      net_nonlin, net_nonlin_kwargs, deep_supervision, False, lambda x: x, InitWeights_He(1e-2),
                      net_num_pool_op_kernel_sizes, net_conv_kernel_sizes, False, True, True)

if torch.cuda.is_available():
    network.cuda()
network.inference_apply_nonlin = softmax_helper

########################################################################
##########                    PREPARE OPTIMIZER           ##############
########################################################################

initial_lr = configs['initial_lr']
momentum = configs['momentum']
nesterov = configs['nesterov']
weight_decay = 3e-5
optimizer = torch.optim.SGD(network.parameters(), initial_lr, weight_decay=weight_decay,
                                         momentum=momentum, nesterov=nesterov)

#######################################################################
####                   PREPARE GENERATORS      ########################
#######################################################################

deep_supervision_scales = None
weights = None

tr_gen, val_gen = prepare_generators(config_train,deep_supervision_scales)

########################################################################
#########                TRAINING CONFIG                 ###############
########################################################################

loss = torch.nn.MSELoss()

fp16 = True
amp_grad_scaler = GradScaler()

numOfEpochs = configs["numOfEpochs"]
tr_batches_per_epoch = configs["tr_batches_per_epoch"]
val_batches_per_epoch = configs["val_batches_per_epoch"]
checkpoint_frequency = configs["checkpoint_frequency"]
outputDir = configs['output_folder']
fold = configs['fold']

all_tr_losses = []
all_val_losses = []

all_tr_losses1 = []
all_val_losses1 = []
all_tr_losses2 = []
all_val_losses2 = []
all_tr_losses3 = []
all_val_losses3 = []
all_tr_losses4 = []
all_val_losses4 = []


startEpoch = 0

bestValLoss = 1e30
bestEpoch = 0

########################################################################
#########         LOAD MODEL IF NECESSARY                ###############
########################################################################

if configs['continue_training']:
    print('loading model from',configs['checkpoint'],flush=True)
    fname = configs['checkpoint']
    saved_model = torch.load(fname, map_location=torch.device('cpu'))

    startEpoch = saved_model['epoch']
    initial_lr = saved_model['learning_rate']
    all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4 = saved_model['plot_stuff']
    bestEpoch,bestValLoss = saved_model['best_stuff']

    amp_grad_scaler.load_state_dict(saved_model['amp_grad_scaler'])
    optimizer.load_state_dict(saved_model['optimizer_state_dict'])
    network.load_state_dict(saved_model['state_dict'])

    print('model loaded',flush=True)



########################################################################
##########              TRAINING LOOP                     ##############
########################################################################

weight_recon = configs['weight_recon']
weight_latent = configs['weight_latent']
weight_decor = configs['weight_decor']

for epoch in range(startEpoch,numOfEpochs):

    network.train()

    print('epoch ',epoch,network.training,flush=True)
    lr = poly_lr(epoch, numOfEpochs, initial_lr, 0.9)
    optimizer.param_groups[0]['lr'] = lr

    epoch_start_time = time()
    train_losses_epoch = []
    train_losses1_epoch = []
    train_losses2_epoch = []
    train_losses3_epoch = []
    train_losses4_epoch = []

    for batchNo in range(tr_batches_per_epoch):
        if batchNo%10 == 0:
            print('#',end='', flush=True)
        l,l1,l2,l3,l4 = run_iteration(tr_gen,network,optimizer,loss,amp_grad_scaler,fp16=True,do_backprop=True,weight_recon = weight_recon, weight_latent = weight_latent, weight_decor = weight_decor)
        train_losses_epoch.append(l)
        train_losses1_epoch.append(l1)
        train_losses2_epoch.append(l2)
        train_losses3_epoch.append(l3)
        train_losses4_epoch.append(l4)
    print('\n',flush=True)
    all_tr_losses.append(np.mean(train_losses_epoch))
    all_tr_losses1.append(np.mean(train_losses1_epoch))
    all_tr_losses2.append(np.mean(train_losses2_epoch))
    all_tr_losses3.append(np.mean(train_losses3_epoch))
    all_tr_losses4.append(np.mean(train_losses4_epoch))

    with torch.no_grad():
        network.eval()
        val_losses = []
        val_losses1 = []
        val_losses2 = []
        val_losses3 = []
        val_losses4 = []
        for _ in range(val_batches_per_epoch):
            print('>',end='', flush=True)
            l,l1,l2,l3,l4 = run_iteration(val_gen,network,optimizer,loss,amp_grad_scaler,fp16=True,do_backprop=False, weight_recon = weight_recon, weight_latent = weight_latent, weight_decor = weight_decor)
            val_losses.append(l)
            val_losses1.append(l1)
            val_losses2.append(l2)
            val_losses3.append(l3)
            val_losses4.append(l4)
        print('\n',flush=True)

        all_val_losses.append(np.mean(val_losses))
        all_val_losses1.append(np.mean(val_losses1))
        all_val_losses2.append(np.mean(val_losses2))
        all_val_losses3.append(np.mean(val_losses3))
        all_val_losses4.append(np.mean(val_losses4))

    epoch_end_time = time()

    print("epoch: ",epoch,", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],', this epoch took: ',epoch_end_time-epoch_start_time, 's ',all_val_losses1[-1],all_val_losses2[-1],all_val_losses3[-1],all_val_losses4[-1],flush=True)

    f = open(configs['log_file'],'at')
    print("epoch: ",epoch,", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],', this epoch took: ',epoch_end_time-epoch_start_time, 's ',all_val_losses1[-1],all_val_losses2[-1],all_val_losses3[-1],all_val_losses4[-1],file = f)
    f.close()

    if all_val_losses[-1] < bestValLoss:
        bestValLoss = all_val_losses[-1]
        bestEpoch = epoch
        fname = outputDir + '/fold_' + str(fold) + '_model_best.model'
        save_checkpoint(fname, network, optimizer, amp_grad_scaler,epoch,lr,all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4, bestEpoch, bestValLoss)

    if epoch%checkpoint_frequency == checkpoint_frequency-1:
        fname = outputDir + '/fold_' + str(fold) + '_model_latest.model'
        save_checkpoint(fname, network, optimizer, amp_grad_scaler,epoch,lr,all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4, bestEpoch, bestValLoss)

fname = outputDir + '/fold_' + str(fold) + '_model_final.model'
save_checkpoint(fname, network, optimizer, amp_grad_scaler,epoch,lr,all_tr_losses, all_tr_losses1,all_tr_losses2,all_tr_losses3,all_tr_losses4,all_val_losses,all_val_losses1,all_val_losses2,all_val_losses3,all_val_losses4, bestEpoch, bestValLoss)


