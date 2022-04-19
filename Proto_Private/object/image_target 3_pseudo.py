import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.optimize import linear_sum_assignment
from tools.utils import AverageMeter, ProgressMeter

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # else:
  #   normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  # else:
  #   normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
  return  transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args): 
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=True)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    print(len(dsets["test"]))
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs*3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
    
def dist(X, Y):
    # X: n1*d Y: n2*d
    # XX.t(): (n1, n2)  YY: (n1, n2) dist：(n1, n2)
    X = X.detach()
    Y = Y.detach()
    xx = X.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X @ Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy

    return dist    

def normal_dist(X, Y):
    # X: n1*d Y: n2*d
    # XX.t(): (n1, n2)  YY: (n1, n2) dist：(n1, n2)
    norm_X = X.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    norm_Y = Y.pow(2).sum(1).pow(0.5).detach().unsqueeze(1)
    X = X * (1/norm_X)
    Y = Y * (1/norm_Y)
    
    xx = X.pow(2).sum(1).repeat(Y.shape[0], 1)
    xy = X @ Y.t()
    yy = Y.pow(2).sum(1).repeat(X.shape[0], 1)
    dist = xx.t() + yy - 2 * xy

    return dist
best_acc1 = []
best_acc_list = []
def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
  
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()
    proto_criterion = loss.ProtoLoss(nav_t=args.nav_t, num_classes=args.class_num).cuda()


    modelpath = args.output_dir_src + '/source_F.pt'   
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'   
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'    
    netC.load_state_dict(torch.load(modelpath))
    # netC.eval()
    
    
    # for k, v in netC.named_parameters():
    #     v.requires_grad = False

    param_group = []
    for k, v in netF.named_parameters():
        if args.lr_decay1 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
        else:
            v.requires_grad = False
    for k, v in netB.named_parameters():
        if args.lr_decay2 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        else:
            v.requires_grad = False
    for k, v in netC.named_parameters():
        if args.lr_decay3 > 0:
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay3}]
        else:
            v.requires_grad = False
        

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    clc_losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        max_iter,
        [clc_losses],
        prefix="Epoch: [{}]".format(args.max_epoch))
    #interval_iter = max_iter // args.interval
    
    iter_num = 0
    best_acc = 0
    
    iter_test = iter(dset_loaders["target"])
    while iter_num < max_iter:
        
        if iter_num % 10 == 0:
            
            start_train = True
            with torch.no_grad():
                iter_target = iter(dset_loaders["target"])
                for _ in range(len(dset_loaders["target"])):
                    data = iter_target.next()
                    inputs = data[0]
                    
                    inputs = inputs.cuda()
                    feas = netB(netF(inputs))

                    if start_train:
                        all_fea = feas.float()
                        start_train = False
                    else:
                        all_fea = torch.cat((all_fea, feas.float()), 0)
                
                
                init_centroid = netC.fc.weight.data.clone().detach()   
                # print(init_centroid.shape)
                kmodel = KMeans(n_clusters=args.class_num,
                                init=init_centroid.cpu(),
                                n_init=1,
                                max_iter=300,
                                tol=1e-4
                                ).fit(all_fea.cpu().numpy())         
                centroid_target = torch.from_numpy(kmodel.cluster_centers_).cuda()
                # print(kmodel.n_iter_)
                
                dis = dist(centroid_target, init_centroid.cuda()).cpu()
                #dis = normal_dist(centroid_target, init_centroid)
                row, col = linear_sum_assignment(dis)
                match_matrix = torch.zeros(dis.shape).cuda()
                match_matrix[row, col] = 1
                # print(match_matrix)
                
                
        try:
            inputs_test, pred, _ = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, pred,_ = iter_test.next()

        

        
        inputs_test = inputs_test.cuda()
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        
        
        
        features_test = netB(netF(inputs_test))
        outputs_test = netC(features_test)
        
        classifier_loss = torch.tensor(0.0).cuda()
        
        
        if args.cls_par > 0:
            # target pseudo-labeled by source prototype 
            print('three values')
            output_test = nn.Softmax(dim=1)(outputs_test/100)
            pred_value2, predict2 = torch.max(output_test, 1)
            # print(pred_value2, predict2) 
            print(torch.sum(predict2==pred.cuda())/args.batch_size)
            # target pseudo-labeled by target centroid
           
            output_test1 = torch.matmul(features_test, centroid_target.t())
            output_test1 = torch.matmul(output_test1, match_matrix)
            
            output_test1 = nn.Softmax(dim=1)(output_test1/1)
            pred_value1, predict1 = torch.max(output_test1, 1)
            # print(pred_value1, predict1) 
            print(torch.sum(predict1==pred.cuda())/args.batch_size)
            # print(pred)

            # selective pseudo_labeling 

            pred_value = torch.cat((pred_value1.unsqueeze(1), pred_value2.unsqueeze(1)), 1)
            pred_value, _ = torch.max(pred_value, 1)
            
            predict1[pred_value1 < pred_value2] = 0
            predict2[pred_value1 > pred_value2] = 0
            predict = predict1 + predict2
            print(torch.sum(predict==pred.cuda())/args.batch_size)
            
            #print(pred_value, predict)
            
            
            
            all_predict = torch.zeros(1, args.class_num).cuda()
            all_label = torch.zeros(1, 1).long().squeeze(1).cuda()
            
            start_test = True
            
            threshold = args.init_threshold + iter_num * 2 * 1e-3
            if threshold > 0.995:
                threshold = 0.995
            
            for i in range(pred_value.shape[0]):
                if pred_value[i] > threshold and start_test == True:   
                    
                    all_predict = outputs_test[i].unsqueeze(0).float()
                    all_label = predict[i].unsqueeze(0)
                    start_test = False
                    continue
                
                elif pred_value[i] > threshold:
                    all_predict = torch.cat((all_predict, outputs_test[i].unsqueeze(0).float()), 0)
                    all_label = torch.cat((all_label, predict[i].unsqueeze(0)), 0)
            print(all_predict.shape[0])
            classifier_loss += F.cross_entropy(all_predict, all_label) * args.cls_par
            
        
        if args.ent:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(loss.Entropy(softmax_out))
            if args.gent:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * args.ent_par
            classifier_loss += im_loss
        
        if args.proto_par > 0.0:
            prototypes = netC.fc.weight.data.clone()
            proto_loss = proto_criterion(prototypes, features_test)
            classifier_loss += args.proto_par*proto_loss
            
        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        
        iter_num += 1
        if iter_num % 10 == 0:
            progress.display(iter_num)
            
        

        if iter_num % 20 == 0 or iter_num == max_iter:
            netF.eval()
            netB.eval()
            netC.eval()
            
            # acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            # log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)
            
            acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, True)
            log_str = '\nTraining: {}, Task: {},Iter:{}/{}; Accuracy = {:.2f}%'.format(args.trte, args.name, iter_num, max_iter, acc) + '\n' + acc_list
            if best_acc < acc:
               best_acc = acc
               best_acc_list = acc_list
            
            print(log_str+'\n')
            netF.train()
            netB.train()
            netC.train()

    best_acc1.append(best_acc)
    print('{}_best_acc = {}'.format(args.name, best_acc)) 
    print(best_acc_list)
    return netF, netB, netC

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='7', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int  , default=2, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='VISDA', choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2019, help="random seed")
 
    parser.add_argument('--gent', dest='gent', action='store_true')
    parser.add_argument('--ent', dest='ent', action='store_true')
    parser.add_argument('--init_threshold', type=int, default=0.8)
    parser.add_argument('--cls_par', type=float, default=0.1)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--lr_decay3', type=float, default=1.0)
    
    parser.add_argument('--lr_decay_nav', type=float, default=1.0)
    parser.add_argument('--nav_t', type=float, default=1.0)
    parser.add_argument('--proto_par', type=float, default=1.0)
    parser.add_argument('--trte', type=str, default='full', choices=['full', 'val'])



    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])  
    parser.add_argument('--output', type=str, default='data/')
    parser.add_argument('--output_src', type=str, default='data/')
    parser.add_argument('--da', type=str, default='uda', choices=['uda', 'pda'])
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    if args.dset == 'officeHome':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office31':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31
    if args.dset == 'VISDA':
        names = ['train', 'validation']
        args.class_num = 12
        
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    folder = '/data1/zym/VisDA-2017/'
    args.source_dset_path = folder + names[args.s] + '/' + 'image_list.txt'
    args.t_dset_path = folder + names[args.t] + '/' + 'image_list.txt'
    args.test_dset_path = folder + names[args.t] + '/' + 'image_list.txt'
    # folder = '/data3/Stone/Proto_DA-master/data/'
    
    # args.t_dset_path = folder +  args.dset + '/' + 'image_list' + '/' + names[args.t] + '.txt'
    # args.test_dset_path = folder + args.dset + '/' + 'image_list' + '/' + names[args.t] + '.txt'
    args.name = names[args.s][0].upper()+names[args.t][0].upper()
    
    
    args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
    train_target(args)
print(best_acc1)
        
        #args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper()+names[args.t][0].upper())
        

        # if not osp.exists(args.output_dir):
        #     os.system('mkdir -p ' + args.output_dir)
        # if not osp.exists(args.output_dir):
        #     os.mkdir(args.output_dir)

        
       
