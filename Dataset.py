from torch.utils.data import Dataset, DataLoader
import torch
import scipy.sparse as sp
import os
import numpy as np
import options
import pickle

class Spcc_Dataset(Dataset):
    def __init__(self,args):
        super(Spcc_Dataset,self).__init__()
        self.args=args
        self.dataset_name=args.basic.dataset_name
        self.device=args.basic.device

        self.train_user_item,self.test_user_item,self.item_hitRate=self._get_data()
        self.user_num,self.item_num=self._compute_item_user_num(self.train_user_item,self.item_hitRate)
        self.train_interaction_num=self._compute_train_interaction_num(self.train_user_item)
        self.UserItemNet=self._generate_UserItemNet()
        self._train_sampling() # 需要先采样一遍，用于__len__方法
        # self.csr_adj_tatrix=self._get_adj_matrix()

    
    def __getitem__(self,index):
        
        data=self.train_data_np
        
        return data[index]

    def __len__(self):
        length=self.train_data_np.shape[0]

        return length
    
        
    def _get_data(self):
        print('Loading data')

        train_user_item=np.load(self.args.basic.data_path+self.dataset_name+'/train_user_item.npy', allow_pickle=True).item()
        test_user_item=np.load(self.args.basic.data_path+self.dataset_name+'/test_user_item.npy', allow_pickle=True).item() 
        item_hitRate=np.load(self.args.basic.data_path+self.dataset_name+'/item_hitRate.npy', allow_pickle=True).item()    

        return train_user_item,test_user_item,item_hitRate

    def _get_adj_matrix(self):
        print('Reading csr_matrix file')
        file=self.args.basic.data_path+self.dataset_name+'/csr_'+self.dataset_name+'_matrix.npz'
        csr_adj_matrix=sp.load_npz(file)
        return csr_adj_matrix


    def _compute_train_interaction_num(self,train_user_item):

        # 统计一下训练集有多少交互
        train_interaction_num=0
        for user in train_user_item.keys():
            train_interaction_num+=len(train_user_item[user])

        print(f"train_interaction_num: {train_interaction_num}")

        return train_interaction_num

    def _compute_item_user_num(self,train_user_item,item_hitRate):
        '''
        这里能够用item_hitRate计算item_num的前提是构造数据集的时候保证训练集中的每个物品出现过(和用户有交互), 每个用户都有交互物品
        '''
        user_num,item_num=len(train_user_item.keys()),len(item_hitRate.keys())
        print(f"user_num: {user_num}, item_num: {item_num}")
        return user_num,item_num

    def _train_sampling(self):
        '''
        NCF是使用CrossEntropy, 所以数据格式构造改一下
        '''
        # print(f"Epoch negative data sampling ...")

        # # 和LightGCN稍有差别，差别在于对user的采样方式
        # item_set=set(range(user_nums))
        # train_data=[]
        # for user in train_user_item.keys():
        #     for p_item in range(len(train_user_item[user])):
        #         n_item=np.random.choice(list(item_set-train_user_item[user]))
        #         row_data=[user,p_item,n_item]
        #         train_data.append(row_data)

        # 和LightGCN完全一样
        train_data=[]
        users = np.random.randint(0, self.user_num, self.train_interaction_num)
        for user in users:
            posForUser = list(self.train_user_item[user]) # 得到这个用户的所有正样本list
            if len(posForUser) == 0: # 如果这个用户没有正样本，跳过
                continue
            posindex = np.random.randint(0, len(posForUser)) # 从这个用户的所有正样本随机抽一个
            p_item = posForUser[posindex]
            while True:
                n_item = np.random.randint(0, self.item_num) # 从所有item中随机抽一个
                if n_item in posForUser: # 如果抽中的在正样本集中，继续抽，知道抽出一个负样本
                    continue
                else:
                    break
            train_data.append([user, p_item, n_item]) # 添加到训练集

        # # Following LightGCN, For CrossEntropy format
        # train_data=[]
        # users = np.random.randint(0, self.user_num, self.train_interaction_num)
        # for user in users:
        #     posForUser = list(self.train_user_item[user]) # 得到这个用户的所有正样本list
        #     if len(posForUser) == 0: # 如果这个用户没有正样本，跳过
        #         continue
        #     posindex = np.random.randint(0, len(posForUser)) # 从这个用户的所有正样本随机抽一个
        #     p_item = posForUser[posindex]
        #     while True:
        #         n_item = np.random.randint(0, self.item_num) # 从所有item中随机抽一个
        #         if n_item in posForUser: # 如果抽中的在正样本集中，继续抽，知道抽出一个负样本
        #             continue
        #         else:
        #             break
        #     train_data.append([user, p_item, 1]) # 添加到训练集
        #     train_data.append([user, n_item, 0])

        self.train_data_np=np.array(train_data)

        # print(f"Ending epoch negative data sampling ...")

    def _generate_UserItemNet(self):
        print("Contructing UserItemNet for accelerating get relation...")
        UserItemNet=sp.dok_matrix((self.user_num,self.item_num),dtype=np.float32)
        for user in self.train_user_item.keys():
            for item in self.train_user_item[user]:
                UserItemNet[user,item]=1
        return UserItemNet.tocsr() # 转为csr矩阵




if __name__=="__main__":
    pass
