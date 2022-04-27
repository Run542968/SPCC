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
    para_lookup=torch.zeros((user_num,3)).to(args.basic.device) # median,mean,std

    model.eval()
    with torch.no_grad(): # 禁止计算梯度
        for user in range(user_num):
            train_item=np.array(list(item_set-test_user_item.get(user,set()))) # 遇到没有测试集数据的user，设置为空set
            user_id=np.array(user)
            sti,median,mean,std=model._for_cur(user_id,train_item)
            para_lookup[user][0],para_lookup[user][1],para_lookup[user][2]=median,mean,std   

    model.para_lookup=para_lookup
    model.Cur_mining=True
    
    ## save para_lookup
    cur_para_path = './ckpt/'+args.basic.model_name+'/'+'Cur_para.pkl'
    para_lookup_np = para_lookup.cpu().numpy()
    np.save(cur_para_path,para_lookup_np)

