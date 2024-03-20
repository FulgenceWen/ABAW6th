import datetime
import os
import time
import argparse
import numpy as np
from omegaconf import OmegaConf
from thop import profile
import logging
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from toolkit.utils.loss import *
from toolkit.utils.metric import *
from toolkit.utils.functions import *
from toolkit.models import get_models
from toolkit.dataloader import get_dataloaders
import matplotlib.pyplot as plt
from datetime import datetime


def output_txt(prediction_dict, save_path):
    pseudo_label_file = '/data/wenzhuofan/Data/ABAW/Aff-Wild2/predictions.txt'
    pseudo_labels_df = pd.read_csv(pseudo_label_file, header=0)

    merged_predictions = {}
    for video_name, preds_list in prediction_dict.items():
        video_prefix = video_name.split('_')[:-1]  # 获取视频名称前缀
        video_prefix ='_'.join(video_prefix)
        if video_prefix not in merged_predictions:
            merged_predictions[video_prefix] = preds_list
        else:
            merged_predictions[video_prefix] = np.concatenate((merged_predictions[video_prefix], preds_list), axis=0)

    filled_labels = []
    last_preds = None
    for index, row in pseudo_labels_df.iterrows():
        video_name = row[0].split('/')[0]  # 获取斜杠前面的部分作为视频名称
        img_id= int(row[0].split('/')[1][:-4])-1
        if video_name not in merged_predictions:
            print(f"Video {video_name} not found in predictions")
            continue
        preds = merged_predictions[video_name]
        if img_id>=len(preds):
            print(f"Video {video_name} Image {img_id} not found in predictions, use last preds")
            output_row = "{},{:.6f},{:.6f}\n".format(row[0], np.clip(last_preds[0], -1, 1), np.clip(last_preds[1], -1, 1))  # 替换伪标签
            filled_labels.append(output_row)
            continue
        output_row = "{},{:.6f},{:.6f}\n".format(row[0], np.clip(preds[img_id][0],-1,1), np.clip(preds[img_id][1],-1,1))  # 替换伪标签
        last_preds = preds[img_id]
        filled_labels.append(output_row)

    # 添加列名
    header = "image_location,valence,arousal\n"
    filled_labels.insert(0, header)

    return filled_labels


def plot_selected_preds_labels(arousal_preds, valence_preds, arousal_labels, valence_labels, num_batches=4, data_type='train',epoch=0):
    """
    绘制指定批次数量的 arousal 和 valence 的 preds 和 labels 曲线。

    Args:
        arousal_preds (ndarray): arousal 的预测值.
        valence_preds (ndarray): valence 的预测值.
        arousal_labels (ndarray): arousal 的标签值.
        valence_labels (ndarray): valence 的标签值.
        num_batches (int): 要绘制的批次数量，默认为 4.
    """
    # 设置图形大小
    plt.figure(figsize=(15, 10))

    for i in range(num_batches):
        # 绘制 arousal 的 preds 和 labels 曲线
        plt.subplot(num_batches, 2, i * 2 + 1)
        plt.plot(arousal_preds[i].flatten(), label='Predictions')
        plt.plot(arousal_labels[i].flatten(), label='Labels')
        plt.title(f'{data_type} {epoch} Arousal Predictions and Labels - Batch {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

        # 绘制 valence 的 preds 和 labels 曲线
        plt.subplot(num_batches, 2, i * 2 + 2)
        plt.plot(valence_preds[i].flatten(), label='Predictions')
        plt.plot(valence_labels[i].flatten(), label='Labels')
        plt.title(f'{data_type} {epoch} Valence Predictions and Labels - Batch {i + 1}')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()
# 调用示例
# plot_selected_preds_labels(arousal_preds, valence_preds, arousal_labels, valence_labels, num_batches=4)


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, epoch, optimizer=None, train=False,test=False):
    
    vidnames = []
    val_preds, val_labels = [], []
    emo_probs, emo_labels = [], []
    losses = []

    assert not train or optimizer!=None
    config.train = train # 将 train 设置为全局变量影响后面 dataloader 信息
    if train:
        model.train()
    else:
        model.eval()
    
    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()
        
        # read data + cuda
        batch, emos, vals, bnames = data
        vidnames += bnames
        if test and batch['videos'].shape[0] < len(args.gpu):
                for key in batch: batch[key] = batch[key].repeat(len(args.gpu),1, 1)
        for key in batch: batch[key] = batch[key].cuda()
        emos = emos.cuda()
        vals = vals.cuda()

        # forward process
        # start_time = time.time()
        features, emos_out, vals_out, interloss = model(batch)
        if test:
            vals_out = vals_out.view(len(args.gpu), -1,vals_out.shape[-2], vals_out.shape[-1])
            vals_out=vals_out.mean(dim=0)
        # duration = time.time() - start_time
        # macs, params = profile(model, inputs=(batch, ))
        # print(f"MACs: {macs}, Parameters: {params}, Duration: {duration}; bsize: {len(bnames)}")

        # loss calculation
        loss = interloss
        if args.output_dim1 != 0:
            loss = loss + cls_loss(emos_out, emos)
            emo_probs.append(emos_out.data.cpu().numpy())
            emo_labels.append(emos.data.cpu().numpy())
        if args.output_dim2 != 0:
            if not test:
                loss = loss + reg_loss(vals_out, vals)
            val_preds.append(vals_out.data.cpu().numpy())
            val_labels.append(vals.data.cpu().numpy())
        if len(args.gpu) > 1:
            try:
                loss = loss.mean()
            except:
                loss = loss.float().mean()
        losses.append(loss.data.cpu().numpy())
        
        # optimize params
        if train:
            loss.backward()
            if args.local_rank != -1 or len(args.gpu) > 1:
                if model.module.model.grad_clip != -1:
                    torch.nn.utils.clip_grad_value_([param for param in model.module.parameters() if param.requires_grad],
                                                    model.module.model.grad_clip)
            elif model.model.grad_clip != -1:
                    torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad],
                                                    model.model.grad_clip)
            optimizer.step()
        
        # print
        if (iter+1) % args.print_iters == 0:
            print (f'process on {iter+1}|{len(dataloader)}, meanloss: {np.mean(losses)}')

        ## whether save models [only train process consider save models]
        if train and args.savemodel and (iter+1) % args.save_iters == 0: # [time consuming]
            save_path = f"{save_modelroot}/{prefix_name}_epoch:{'%02d' %(epoch)}_iter:{'%06d' %(iter+1)}_meanloss:{np.mean(losses)}_{name_time}"
            model.model.model.save_pretrained(save_path)

    if emo_probs  != []: emo_probs  = np.concatenate(emo_probs)
    if emo_labels != []: emo_labels = np.concatenate(emo_labels)
    if val_preds  != []: val_preds  = np.concatenate(val_preds)
    if val_labels != []: val_labels = np.concatenate(val_labels)

    # arousal_preds = val_preds[:, :, 1]
    # valence_preds = val_preds[:, :, 0]
    # arousal_labels = val_labels[:, :, 1]
    # valence_labels = val_labels[:, :, 0]
    # plot_selected_preds_labels(arousal_preds, valence_preds, arousal_labels, valence_labels, num_batches=8, data_type='train' if train else 'val',epoch=epoch)

    if test:
        save_results = dict(
            val_preds = val_preds,
            names = vidnames,
        )
        return save_results

    results, _ = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
    save_results = dict(
        names = vidnames,
        loss  = np.mean(losses),
        **results,
    )
    return save_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Params for datasets
    parser.add_argument('--dataset', type=str, default=None, help='dataset')
    parser.add_argument('--train_dataset', type=str, default=None, help='train dataset') # for cross-corpus test  
    parser.add_argument('--test_dataset',  type=str, default=None, help='test dataset')  # for cross-corpus test
    parser.add_argument('--save_root', type=str, default='./saved', help='save prediction results and models')
    parser.add_argument('--debug', action='store_true', default=False, help='whether use debug to limit samples')
    parser.add_argument('--savemodel', action='store_true', default=False, help='whether to save model, default: False')
    parser.add_argument('--save_iters', type=int, default=1e8, help='save models per iters')

    # Params for feature inputs
    parser.add_argument('--audio_feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text_feature',  type=str, default=None, help='text feature name')
    parser.add_argument('--video_feature', type=str, default=None, help='video feature name')
    parser.add_argument('--feat_type',  type=str, default=None, help='feature type [utt, frm_align, frm_unalign]')
    parser.add_argument('--feat_scale', type=int, default=None, help='pre-compress input from [seqlen, dim] -> [seqlen/scale, dim]')
    # Params for raw inputs
    parser.add_argument('--e2e_name', type=str, default=None, help='e2e pretrained model names')
    parser.add_argument('--e2e_dim',  type=int, default=None, help='e2e pretrained model hidden size')

    # Params for model
    parser.add_argument('--n_classes', type=int, default=None, help='number of classes [defined by args.label_path]')
    parser.add_argument('--hyper_path', type=str, default=None, help='whether choose fixed hyper-params [default use hyperparam tuning]')
    parser.add_argument('--model', type=str, default=None, help='model name for training [mlp, attention, and others from MMSA toolkits]')

    # Params for training
    parser.add_argument('--lr', type=float, default=None, metavar='lr', help='set lr rate') # 如果是None, lr 作为了 hyper-params 通过 model-tune.yaml 调节
    parser.add_argument('--lr_adjust', type=str, default='case1', help='[case1, case2]. case1: uniform lr; case2: pretrain lr = 1/10 fc lr')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size [deal with OOM]')
    parser.add_argument('--num_workers', type=int, default=0, metavar='nw', help='number of workers')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--print_iters', type=int, default=1e8, help='print per-iteartion')
    parser.add_argument('--gpu', type=str, default='1,2,3', help='gpu id')
    parser.add_argument("--local_rank",default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.gpu = list(range(len(args.gpu.split(','))))
    if args.batch_size % len(args.gpu) != 0:
        raise ValueError(f'batch_size={args.batch_size} should be divided by len(gpu)={len(args.gpu)}')
    if args.local_rank == -1:
        if len(args.gpu) == 1:
            torch.cuda.set_device(args.gpu[0])
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 自定义日志文件名
    log_filename = f"{args.video_feature}_{current_time}.log"
    log_path='/data/wenzhuofan/Code/ABAW6th/MERBench/log'
    logging.basicConfig(filename=os.path.join(log_path,log_filename), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(args)

    print ('====== Params Pre-analysis =======')
    ## accelate: pre-compress for ['frm_align', 'frm_unalign']
    if args.feat_type == 'utt':
        args.feat_scale = 1
    elif args.feat_type == 'frm_align':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        # args.feat_scale = 6
        args.feat_scale = 1
    elif args.feat_type == 'frm_unalign':
        assert args.audio_feature.endswith('FRA')
        assert args.text_feature.endswith('FRA')
        assert args.video_feature.endswith('FRA')
        args.feat_scale = 12

    ## define store folder
    if args.train_dataset is not None:
        args.save_root = f'{args.save_root}-cross'
    whole_features = [args.audio_feature, args.text_feature, args.video_feature]
    whole_features = [item for item in whole_features if item is not None]
    if len(set(whole_features)) == 0:
        args.save_root = f'{args.save_root}-others'
    elif len(set(whole_features)) == 1:
        args.save_root = f'{args.save_root}-unimodal'
    elif len(set(whole_features)) == 2:
        args.save_root = f'{args.save_root}-bimodal'
    elif len(set(whole_features)) == 3:
        args.save_root = f'{args.save_root}-trimodal'

    ## generate model_config
    if args.hyper_path is None:
        model_config = OmegaConf.load('toolkit/model-tune.yaml')[args.model]
        model_config = func_random_select(model_config)
    else:
        model_config = OmegaConf.load(args.hyper_path)[args.model]
    config.dataset = args.dataset
    args = merge_args_config(args, model_config) # merge params
    print('args: ', args)

    ## save root
    save_resroot  = os.path.join(args.save_root, 'result')
    save_modelroot  = os.path.join(args.save_root, 'model')
    if not os.path.exists(save_resroot):  os.makedirs(save_resroot)
    if not os.path.exists(save_modelroot): os.makedirs(save_modelroot)
    # gain prefix_name
    feature_name = "+".join(sorted(list(set(whole_features)))) # sort to avoid random order
    model_name = f'{args.model}+{args.feat_type}+{args.e2e_name}'
    prefix_name = f'features:{feature_name}_dataset:{args.dataset}_model:{model_name}'
    if args.train_dataset is not None:
        assert args.test_dataset is not None
        prefix_name += f'_train:{args.train_dataset}_test:{args.test_dataset}'


    print ('====== Reading Data =======')
    dataloader_class = get_dataloaders(args) # (MER2023 + e2e + e2e_name)
    train_loaders, eval_loaders, test_loaders = dataloader_class.get_loaders()
    assert len(train_loaders) == len(eval_loaders)
    print (f'train&val folder:{len(train_loaders)}; test sets:{len(test_loaders)}')
    args.audio_dim, args.text_dim, args.video_dim = train_loaders[0].dataset.get_featdim()
    args.seq_len = train_loaders[0].dataset.get_seqlen()


    print ('====== Training and Evaluation =======')
    folder_save = [] # store best results for each folder
    folder_duration = []
    for ii in range(len(train_loaders)):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        eval_loader  = eval_loaders[ii]
        start_time = name_time = time.time()

        print (f'Step1: build model (each folder has its own model)')
        model = get_models(args)
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl", init_method='env://')

            model.to(device)
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        elif len(args.gpu) > 1:
            model = torch.nn.DataParallel(model, device_ids=args.gpu)
            model.cuda()
        else:
            model.cuda()

        reg_loss = CCCLoss().cuda()
        cls_loss = CELoss().cuda()

        if args.lr_adjust == 'case1':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
        elif args.lr_adjust == 'case2':
            assert args.model == 'e2e_model', 'lr_adjust=case2 only support for e2e_model'
            print ('set different learning rates for different layers')
            optimizer = optim.Adam([{'params': model.model.pretrain_model.parameters(), 'lr': args.lr/10},
                                    {'params': model.model.encoder.parameters(),        'lr': args.lr},
                                    {'params': model.model.fc_out_1.parameters(),       'lr': args.lr},
                                    {'params': model.model.fc_out_2.parameters(),       'lr': args.lr},
                                    ], lr=args.lr, weight_decay=args.l2)

        print (f'Step2: training (multiple epoches)')
        whole_store = []
        whole_metrics = []
        for epoch in range(args.epochs):

            epoch_store = {}

            ## training and validation
            train_results = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, epoch=epoch, optimizer=optimizer, train=True )
            eval_results  = train_or_eval_model(args, model, reg_loss, cls_loss, eval_loader,  epoch=epoch, optimizer=None,      train=False)
            func_update_storage(inputs=eval_results, prefix='eval', outputs=epoch_store)

            ## use args.metric_name to determine best_index
            train_metric = gain_metric_from_results(train_results, args.metric_name)
            eval_metric  = gain_metric_from_results(eval_results,  args.metric_name)
            whole_metrics.append(eval_metric)
            print ('epoch:%d; metric:%s; train results:%.4f; eval results:%.4f' %(epoch+1, args.metric_name, train_metric, eval_metric))
            logging.info('epoch:%d; metric:%s; train results:%.4f; eval results:%.4f' %(epoch+1, args.metric_name, train_metric, eval_metric))
            # testing and saving
            for jj, test_loader in enumerate(test_loaders):
                test_results = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, epoch=epoch, optimizer=None, train=False, test=True)
                func_update_storage(inputs=test_results, prefix=f'test{jj+1}', outputs=epoch_store)
            
            ## saving
            whole_store.append(epoch_store)


        print (f'Step3: saving and testing on the {ii+1} folder')
        best_index = np.argmax(np.array(whole_metrics))
        folder_save.append(whole_store[best_index])
        end_time = time.time()
        duration = end_time - start_time
        folder_duration.append(duration)
        print (f'>>>>> Finish: training on the {ii+1}-th folder, best_index: {best_index}, duration: {duration} >>>>>')
        # clear memory
        del model
        del optimizer
        torch.cuda.empty_cache()


    print ('====== Prediction and Saving =======')
    args.duration = np.sum(folder_duration) # store duration
    cv_result = gain_cv_results(folder_save)
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    # save_path = f'{save_resroot}/args/cv_{prefix_name}_{cv_result}_{name_time}.npz'
    save_path = f'{save_resroot}/args/cv_{prefix_name}_{formatted_time}.npz'
    print (f'save results in {save_path}')
    np.savez_compressed(save_path,
                        args)

    ## store test1|test2|test3 results
    for jj in range(len(test_loaders)):
        # emo_labels, emo_probs = average_folder_for_emos(folder_save, f'test{jj+1}')
        # val_labels, val_preds = average_folder_for_vals(folder_save, f'test{jj+1}')
        # _, test_result = dataloader_class.calculate_results(emo_probs, emo_labels, val_preds, val_labels)
        #save_path = f'{save_resroot}/test{jj+1}_{prefix_name}_{test_result}_{name_time}.npz'

        save_path = f'{save_resroot}/preds/test{jj + 1}_{prefix_name}_{formatted_time}.txt'
        prediction_dict = dict(zip( folder_save[0]['test1_names'],folder_save[0]['test1_val_preds']))
        # 将填充后的结果写入文件
        preds_to_txt=output_txt(prediction_dict, save_path)
        with open(save_path, 'w') as f:
            f.writelines(preds_to_txt)
        print (f'save results in {save_path}')


