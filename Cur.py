import numpy as np
import torch

def Cur_mining(model,dataset,args):
    '''
    求参与训练的样本的median和mean,不包括测试集中item
    '''
    user_num=dataset.user_num
    item_num=dataset.item_num

    item_set=set(np.arange(item_num))
    test_user_item=dataset.test_user_item
    para_lookup=torch.zeros((user_num,2)).to(args.basic.device) # median,mean

    model.eval()
    with torch.no_grad(): # 禁止计算梯度
        for user in range(user_num):
            train_item=np.array(list(item_set-test_user_item.get(user,set()))) # 遇到没有测试集数据的user，设置为空set
            user_id=np.array(user)
            sti,median,std=model._for_cur(user_id,train_item)
            para_lookup[user][0],para_lookup[user][1]=median,std

    print(f"para_lookup[0]:{para_lookup[0]}")
    model.para_lookup=para_lookup
    model.Cur_mining=True

