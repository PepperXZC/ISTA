from statistics import mode
from ISTA import ISTAnet
from vxm import networks, layers
import torch
import generator
import os
import random
import argparse
import time
import numpy as np
import torch
from tqdm import tqdm
import voxelmorph as vxm  # nopep8
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

# import voxelmorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

class MSE:
    """
    Sigma-weighted mean squared error for image reconstruction.
    """

    def __init__(self, image_sigma=144*192*160):
        self.image_sigma = image_sigma

    def mse(self, y_true, y_pred):
        return torch.square(y_true - y_pred)

    def loss(self, y_true, y_pred, reduce='mean'):
        # compute mse
        mse = self.mse(y_true, y_pred)
        # reduce
        if reduce == 'mean':
            mse = torch.mean(mse)
        elif reduce == 'max':
            mse = torch.max(mse)
        elif reduce is not None:
            raise ValueError(f'Unknown MSE reduction type: {reduce}')
        # loss
        return 1.0 / (self.image_sigma ** 2) * mse

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', help='line-seperated list of training files',
    default=r"../dataset/file_list.txt")
# parser.add_argument('--img-prefix', default=r"/home/ISTA/dataset",help='optional input image file prefix')
parser.add_argument('--img-prefix', default=r"C:\Users\Administrator\ISTA\dataset",help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--model-dir', default='models/',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=30,
                    help='frequency of model saves (default: 100)')
parser.add_argument('--load-model1', help='optional model file to initialize with')
parser.add_argument('--load-model2', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=104,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=0.01,
                    help='weight of deformation loss (default: 0.01)')
args = parser.parse_args()

bidir = args.bidir

train_files = vxm.py.utils.read_file_list(args.img_list, prefix=args.img_prefix,
                                          suffix=args.img_suffix)
assert len(train_files) > 0, 'Could not find any training data.'

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

def distance(vector1,vector2, P, N):
    return torch.sqrt(torch.square(vector2-vector1).sum()) / (P * N)  # pow()是自带函数

# distance = nn.PairwiseDistance(p=2)

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(args.atlas, np_var='vol',
                                      add_batch_axis=True, add_feat_axis=add_feat_axis)
    dataset = generator.scan_to_atlas(train_files, atlas,
                                             batch_size=args.batch_size, bidir=args.bidir,
                                             add_feat_axis=add_feat_axis)
else:
    # scan-to-scan generator
    dataset = generator.scan_to_scan(
        train_files, batch_size=args.batch_size, bidir=args.bidir, add_feat_axis=add_feat_axis)

# extract shape from sampled input
[index_1, index_2], inshape = next(dataset)
in_shape = inshape[0][0].shape[1:-1]
N = in_shape[0] * in_shape[1] * in_shape[2]


# 模型文件存放
print(torch.randn(3))
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# 获取gpu信息
gpus = args.gpu.split(',')
nb_gpus = len(gpus)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

torch.backends.cudnn.deterministic = not args.cudnn_nondet

enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

vxm_loss_func = MSE().loss # 或NCC
weights = [1, 0.01] #后一项是lambda

# x_history = {}
# for i in range(30): # 目前假定数据集共30张图片
#     x_history[i] = 0

if args.load_model1 and args.load_model2:
    # load initial model (if specified)
    model = networks.VxmDense.load(args.load_model1, device)
    mini_model = ISTAnet.EPN_Net.load(args.load_model2, device)
else:
    # otherwise configure new model
    model = networks.VxmDense(
        inshape=in_shape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize
    )
    mini_model = ISTAnet.EPN_Net(inshape=in_shape,f_function=vxm_loss_func, iteration_S=50)

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save
    mini_model = torch.nn.DataParallel(mini_model)
    mini_model.save = mini_model.module.save
model.to(device)
mini_model.to(device)

model.train()
mini_model.train()

optimizer_vxm = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer_epn = torch.optim.Adam(mini_model.parameters(), lr=args.lr)
transformer = layers.SpatialTransformer(in_shape, device=device)

# writer = SummaryWriter()
# input_f = torch.rand(in_shape)
# input_m = torch.rand(in_shape)
# writer.add_graph(mini_model, x_history, input_f, input_m)

# writer.close()


print("start")
first_loss_list = []
second_loss_list = []
for epoch in range(args.initial_epoch, args.epochs):

    # save model checkpoint
    # if epoch % 20 == 0:
    #     model.save(os.path.join(model_dir, '%04d.pt' % epoch))

    epoch_loss = []
    epoch_total_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):
    # for step in range(args.steps_per_epoch):

        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        [index_1, index_2], (inputs, y_true) = next(dataset) # 以m图为主
        # x_history[index_1]
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs] # return [1, 1, 144, 192, 160] * 2, m, f
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true] # return [1, 1, 144, 192, 160] * 2, f, 0
        # [1, 1, 144, 192, 160]; [1, 3, 144, 192, 160]
        # y_true中后一组数为0
        for data in inputs:
            data.requires_grad_(True)
        for data in y_true:
            data.requires_grad_(True)
        # run inputs through the model to produce a warped image and flow field
        optimizer_vxm.zero_grad()
        y_pred = model(*inputs)  # return [1, 1, 144, 192, 160] * 2: m + phi, phi
        # m + phi: [1, 1, 144, 192, 160]; phi: [1, 3, 144, 192, 160]
        first_loss = vxm_loss_func(y_true[0],y_pred[0])
        for data in y_pred:
            data.requires_grad_(True)
        
        # 应该是
        # phi的思路不改，增加训练epoch数大小
        optimizer_epn.zero_grad()
        phi_pred = mini_model(y_pred[1],y_true[0],inputs[0])
        x_pred = transformer(inputs[1],phi_pred)

        y_temp = y_true[0].detach()
        y_temp.requires_grad_(True)
        x_temp = x_pred.detach()
        x_temp.requires_grad_(True)

        second_loss = distance(x_temp, y_temp, args.steps_per_epoch, N)

        loss_list = [first_loss, second_loss]

        first_loss.backward()
        optimizer_vxm.step()
        second_loss.backward()
        optimizer_epn.step()
        

        # calculate total loss
        # loss = 0
        # loss_list = []
        # for n, loss_function in enumerate(losses): #1: f, m+phi; 2: 0, phi
        #     curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
        #     loss_list.append(curr_loss.item())
        #     loss += curr_loss

        

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    time_info = '%.4f sec/step' % np.mean(epoch_step_time)
    losses_info = ', '.join(['%.4e' % f for f in loss_list])
    loss_info = 'first loss:{num1}, second loss:{num2}'.format(num1=first_loss, num2=second_loss)
    # f = open("../output/logs.txt", 'w')
    # print(' - '.join((epoch_info, time_info, loss_info)), flush=True)
    print("{nums1} {nums2} {nums3}\n".format(nums1=epoch, nums2=first_loss, nums3=second_loss))


    # f.write("epoch:{num1}, first loss:{num2}, second loss:{num3}\n".format(num1=epoch, num2=first_loss, num3=second_loss))
    # f.close()

    if epoch % 10 == 0:
        model.save(os.path.join(model_dir + "vxm/", '%04d.pt' % epoch))
        mini_model.save(os.path.join(model_dir + "epn/", '%04d.pt' % epoch))

# final model save



