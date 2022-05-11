import torch
from Dataset import Spcc_Dataset
from model import SPCC
from eval import get_recall,get_precision,get_ndcg,get_novelty
import numpy as np
from utils.Utils import get_logger,setup_seed
import options

def test(dataset,model,logger,writer,epoch,args):
    
    model.eval()
    train_user_item=dataset.train_user_item
    test_user_item=dataset.test_user_item
    item_hitRate=dataset.item_hitRate
    item_num=dataset.item_num
    user_num=dataset.user_num

    recall_list=[]
    precision_list=[]
    ndcg_list=[]
    novelty_list=[]
    rec_set=set()

    user_id=np.arange(user_num)
    batch_size=int(args.test.batch_size)
    batch_num=user_num//batch_size
    
    no_test=0 # 记录测试item为空的用户个数，有4个
    count=0 # 记录测试item不为空的用户个数
    # generate all user's pred_scores
    with torch.no_grad():
        for i in range(batch_num+1):
            batch_user=user_id[batch_size*i:batch_size*(i+1)]

            P_score,S_score=model.test(torch.from_numpy(batch_user).long().to(args.basic.device))
            
            alpha=float(args.test.alpha)
            pred_scores=(alpha*P_score + (1-alpha)*S_score).detach().cpu() #[bs,item_num]

            for j in range(len(batch_user)):
                user=batch_user[j]
                scores=pred_scores[j]
                train_item=np.array(list(train_user_item[user]))
                gt_id=np.array(list(test_user_item.get(user,set()))) # 这里用dict.get()方法，如果不存在这个user，那就返回空的set()
                if len(gt_id)==0:
                    no_test+=1
                    continue

                # 通过mask去除训练中的正样本
                mask=torch.from_numpy(np.where(np.isin(np.arange(item_num),train_item),-1e8,1))
                scores=torch.mul(mask,scores)
                
                # # 另一种去除训练中正样本的方式, 这种直接批量索引的方式是numpy等数组类型特有的
                # scores[train_item]=-1e8

                topk_scores,topk_id=torch.topk(scores,args.test.k)
                topk_id=topk_id.numpy()

                userwise_recall=get_recall(topk_id,gt_id)
                userwise_precision=get_precision(topk_id,gt_id)
                userwise_ndcg=get_ndcg(topk_id,gt_id)
                userwise_novelty=get_novelty(topk_id,item_hitRate)
                
                recall_list.append(userwise_recall)
                precision_list.append(userwise_precision)
                ndcg_list.append(userwise_ndcg)
                novelty_list.append(userwise_novelty)
                for item in topk_id: 
                    rec_set.add(item)

                count+=1
            # print("user:",i)

    assert count+no_test==user_num

    # recall=np.mean(np.array(recall_list))
    # precision=np.mean(np.array(precision_list))
    # ndcg=np.mean(np.array(ndcg_list))
    # novelty=np.mean(np.array(novelty_list))
    recall=np.sum(recall_list)/count
    precision=np.sum(precision_list)/count
    ndcg=np.sum(ndcg_list)/count
    novelty=np.sum(novelty_list)/count
    coverage=len(rec_set)/item_num
    
    if writer!=None:
        writer.add_scalar('metric/recall@%d'%(args.test.k),recall,epoch)
        writer.add_scalar('metric/precision@%d'%(args.test.k),precision,epoch)
        writer.add_scalar('metric/ndcg@%d'%(args.test.k),ndcg,epoch)
        writer.add_scalar('metric/coverage@%d'%(args.test.k),coverage,epoch)
        writer.add_scalar('metric/novelty@%d'%(args.test.k),novelty,epoch)


    if logger!=None:
        logger.info("---------------Test recall & precision & ndcg & coverage-----------------")
        logger.info("epoch:{},test recall:{:.4f},precision:{:.4f}, ndcg:{:.4f}, coverage:{:.4f}, novelty:{:.4f}".format(epoch,recall,precision,ndcg,coverage,novelty))
    print("---------------Test recall & precision & ndcg & coverage-----------------")
    print("epoch:{},test recall:{:.4f},precision:{:.4f}, ndcg:{:.4f}, coverage:{:.4f}, novelty:{:.4f}".format(epoch,recall,precision,ndcg,coverage,novelty))


if __name__=="__main__":
    args=options._process_args_over_yaml()   # load train、test dataset、Cur_dataset
    # setup_seed
    setup_seed(args.basic.seed)

    # load train dataset、Cur_dataset
    dataset=Spcc_Dataset(args)

    # load model
    model = SPCC(dataset.user_num, 
                dataset.item_num,
                dataset.train_user_item,
                dataset.UserItemNet,
                args)

    # load model parameters
    model_name='SPCC_movie_nocur_0'
    test_epoch=99
    print('model_name:',model_name)
    model.load_state_dict(torch.load(r'./ckpt/' + model_name + '/Epoch_' + str(test_epoch) + '.pkl'))

    model.to(args.basic.device)

    logger=None
    writer=None
    model._get_relation()
    test(dataset,model,logger,writer,test_epoch,args)

    # python test.py --config_yaml 'configs/SPCC_movie.yaml' --test.alpha 0.85 --Personal.use_mlp True --test.post_process 'none' --test.k 20
    # CUDA_VISIBLE_DEVICES=2 python test.py --config_yaml 'configs/SPCC_music.yaml' --test.alpha 0.99 --Personal.use_mlp True --test.post_process 'sigmoid' --test.k 20