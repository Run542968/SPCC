import torch
from Dataset import Spcc_Dataset
from model import SPCC
from eval import get_recall,get_precision,get_ndcg,get_novelty
import numpy as np
from utils.Utils import get_logger,setup_seed
import options

def popularity(dataset,args):
    
    test_user_item=dataset.test_user_item
    item_hitRate=dataset.item_hitRate
    sorted_hitRate=sorted(item_hitRate.items(),key=lambda x:x[1],reverse=True)
    pred_item_id=[x[0] for x in sorted_hitRate]
    topk_id=pred_item_id[:args.test.k]
    item_num=dataset.item_num
    user_num=dataset.user_num

    recall_list=[]
    precision_list=[]
    ndcg_list=[]
    novelty_list=[]
    rec_set=set()

    user_id=np.arange(user_num)
    no_test=0 # 记录测试item为空的用户个数，有4个
    count=0 # 记录测试item不为空的用户个数
    for user in user_id:

        gt_id=np.array(list(test_user_item.get(user,set())))
        if len(gt_id)==0:
            no_test+=1
            continue

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
    assert count+no_test==user_num

    recall=np.sum(recall_list)/count
    precision=np.sum(precision_list)/count
    ndcg=np.sum(ndcg_list)/count
    novelty=np.sum(novelty_list)/count
    coverage=len(rec_set)/item_num

    print("---------------Test recall & precision & ndcg & coverage-----------------")
    print("test recall:{:.4f},precision:{:.4f}, ndcg:{:.4f}, coverage:{:.4f}, novelty:{:.4f}".format(recall,precision,ndcg,coverage,novelty))


if __name__=="__main__":
    args=options._process_args_over_yaml()   # load train、test dataset、Cur_dataset
    # setup_seed
    setup_seed(args.basic.seed)

    # load train dataset、Cur_dataset
    dataset=Spcc_Dataset(args)

    popularity(dataset,args)

    # python Popularity.py --config_yaml 'configs/SPCC_music.yaml' --test.k 20