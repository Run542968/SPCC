import torch
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
import numpy as np
import math

def eval_auc_acc(SPCC_model,inputs,args):
    SPCC_model.eval()
    inputs=torch.from_numpy(inputs).to(args.basic.device)
    user_idx,item_idx=inputs[:,0],inputs[:,1]
    labels,hitRate_labels=inputs[:,2],inputs[:,3]
    social_scores,personal_scores,cur_scores,pred_scores,item_scores,sti=SPCC_model(user_idx,item_idx)
    
    #计算精确度和AUC
    labels = labels.to("cpu").detach().numpy()
    pred_scores = torch.sigmoid(pred_scores).to("cpu").detach().numpy()
    auc=roc_auc_score(y_true=labels,y_score=pred_scores)
    predictions=[1 if i>=0.5 else 0 for i in pred_scores]
    acc=np.mean(np.equal(predictions,labels))

    n_hit=np.sum(np.equal(predictions,labels))
    n_total=len(labels)
    return auc,acc,n_hit,n_total

def eval_auc_acc_v2(pred_scores,labels):

    #计算batch数据的精确度和AUC
    labels = labels.to("cpu").detach().numpy()
    pred_scores = torch.sigmoid(pred_scores).to("cpu").detach().numpy()
    auc=roc_auc_score(y_true=labels,y_score=pred_scores)
    predictions=[1 if i>=0.5 else 0 for i in pred_scores]
    acc=np.mean(np.equal(predictions,labels))

    n_hit=np.sum(np.equal(predictions,labels))
    n_total=len(labels)
    return auc,acc,n_hit,n_total


# # 这个输入有问题，因为gt_id和topk_id不一定一样长
# def get_ndcg(topk_id,gt_id): 

#     def getDCG(id_list):
#         return np.sum(
#             np.divide(np.power(2, id_list) - 1, np.log2(np.arange(id_list.shape[0], dtype=np.float32) + 2)),
#             dtype=np.float32)

#     idealist = np.ones_like(gt_id)
#     idcg=getDCG(idealist)
#     rank_list=np.array([1 if (id in gt_id) else 0 for id in topk_id])
#     dcg=getDCG(rank_list)
#     # print("%.2f %.2f" % (dcg,idcg))
#     ndcg=dcg/idcg
#     return ndcg


def get_ndcg(topk_id, gt_id):
    '''
    Following 2021 SIGIR EDUA
    '''
    def getDCG(topk_id, gt_id):
        dcg = 0.0
        for i in range(len(topk_id)):
            item = topk_id[i]
            if item in gt_id:
                dcg += 1.0 / math.log(i + 2)
        return  dcg

    def getIDCG(topk_id, gt_id):
        idcg = 0.0
        i = 0
        for item in topk_id:
            if item in gt_id:
                idcg += 1.0 / math.log(i + 2)
                i += 1
        return idcg

    dcg = getDCG(topk_id, gt_id)
    idcg = getIDCG(topk_id, gt_id)
    if idcg == 0:
        return 0
    return dcg / idcg

def get_precision(topk_id,gt_id):
    TP=0
    for id in topk_id:
        if id in gt_id:
            TP+=1
    precision=TP/len(topk_id)
    return precision

def get_recall(topk_id,gt_id):
    TP=0
    for id in topk_id:
        if id in gt_id:
            TP+=1
    # gd_len=len(topk_id) if len(topk_id)<=(len(test_data)/2) else (len(test_data)/2)
    recall=TP/len(gt_id)
    return recall
    
def get_novelty(topk_id,item_hitRate):
    novel_sum=0
    for id in topk_id:
        novel=-np.log(item_hitRate[id])
        novel_sum+=novel
    novelty=novel_sum/len(topk_id)
    return novelty