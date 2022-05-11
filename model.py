import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class SPCC(nn.Module):
	def __init__(self, user_num, item_num,train_user_item,UserItemNet,args):
		super(SPCC, self).__init__()
	
		self.args=args
		self.Personal_dim=args.Personal.dim
		self.Social_dim=args.Social.dim
		self.item_num=item_num
		self.user_num=user_num
		self.UserItemNet=UserItemNet

		# MF
		self.Puser_lookup = nn.Embedding(user_num, self.Personal_dim)
		self.Pitem_lookup = nn.Embedding(item_num, self.Personal_dim)
		self.Puser_mlp = nn.Sequential()
		self.Pitem_mlp = nn.Sequential()

		for i_cnt in range(self.args.Personal.layers):
			self.Puser_mlp.add_module('user_mlp_{}'.format(i_cnt),
									nn.Linear(self.Personal_dim,self.Personal_dim))
			# self.Puser_mlp.add_module('user_act_{}'.format(i_cnt),
			# 						nn.Tanh())
			self.Pitem_mlp.add_module('item_mlp_{}'.format(i_cnt),
									nn.Linear(self.Personal_dim,self.Personal_dim))
			# self.Pitem_mlp.add_module('item_act_{}'.format(i_cnt),
			# 						nn.Tanh())

		# get train_item_user dict
		self.train_user_item=train_user_item
		self.train_item_user=self._transform_dict(self.train_user_item)

		# modeling relation for social
		self.Sitem_lookup=nn.Embedding(self.item_num,self.Social_dim)	
		self.Suser_lookup=nn.Embedding(self.user_num,self.Social_dim)
		self.att=nn.Linear(self.Social_dim,self.Social_dim)

		# cur
		self.para_lookup=None
		self.Cur_mining=False
	
		self.loss_function=nn.BCEWithLogitsLoss()
		self.Huber_loss=nn.SmoothL1Loss()

	def _transform_dict(self,train_user_item):
		'''
		转换train_user_item -> train_item_user, 用于构造relation
		'''
		
		train_item_user=dict()
		
		for user in train_user_item.keys():
			for item in train_user_item[user]:
				if item not in train_item_user.keys():
					train_item_user[item]=set()
				train_item_user[item].add(user)
				
		return train_item_user

	def _get_relation_slow(self):
		'''
		初始for循环版本,没利用矩阵
		对于大数据集, 得使用这个版本
		'''
		train_item_user_dict=self.train_item_user
        # print("bagin _get_ralation")

		items_relation=torch.from_numpy(np.array([])).to(self.args.basic.device)
		for item in range(self.item_num):
			item_id=torch.from_numpy(np.array(item)).to(self.args.basic.device)#[1]
			# print("item_id:",item_id)
			users_id=torch.from_numpy(np.array(list(train_item_user_dict[item]))).to(self.args.basic.device)#[user_nums]
			# print("user_id:",users_id)
			item_embedding=self.Sitem_lookup(item_id).unsqueeze(0)#[1,dim]
			# print("item_embedding:",item_embedding)
			users_embedding=self.Suser_lookup(users_id)#[user_nums,dim]
			# print("users_embedding:",users_embedding)
			keys=self.att(users_embedding)#[user_nums,dim]

			if self.args.Social.sigmoid_norm:
				qk=torch.sigmoid(torch.mm(item_embedding,torch.transpose(keys,1,0))) #[item_num,dim]x[dim,user_num]->[item_num,user_num] #NOTE:这里其实存在一个归一化和保证非负性的问题，考虑使用sigmoid，因为softmax会平均包括负样本的所有信息
			else:
				qk=torch.mm(item_embedding,torch.transpose(keys,1,0))

			relation=torch.mm(qk,users_embedding)#[1,dim]
			items_relation=torch.cat((items_relation,relation),dim=0)

		self.items_relation=items_relation

	def _get_relation_median(self):
		'''
		for循环 + 利用矩阵
		对于大数据集, 使用串行太慢了, 得使用这个版本
		由于UserItemNet太大, 所以采用这种分batch的方法, 但是又太占内存！！！
		'''

		relation_batch=self.args.Social.relation_batch
		UserItemNet_mask=np.transpose(self.UserItemNet.toarray()) # [user_num,item_num]->[item_num,user_num] 图太大，需要分开搞
		# print("UserItemNet_mask.shape: ",UserItemNet_mask.shape)
		# print("UserItemNet_mask.type: ",type(UserItemNet_mask))

		items_relation=torch.from_numpy(np.array([])).to(self.args.basic.device)
		relation_batch_num=self.item_num//relation_batch
		# print(f"relation_batch_num: {relation_batch_num}")
		item_id=np.arange(self.item_num)

		for i in range(relation_batch_num + 1):
			batch_item=item_id[relation_batch*i : relation_batch*(i+1)]
			batch_UserItemNet=torch.from_numpy(UserItemNet_mask[batch_item,:]).to(self.args.basic.device) # 这里batch_item得在cpu上
			# print(f"batch_UserItemNet.shape:{batch_UserItemNet.shape}")

			batch_item=torch.from_numpy(batch_item).to(self.args.basic.device) #[bs,]
			# print(f"batch_item.shape:{batch_item.shape}")
			all_user_id=torch.arange(self.user_num).to(self.args.basic.device)

			all_user_embedding=self.Suser_lookup(all_user_id)#[user_num,dim]
			batch_item_embedding=self.Sitem_lookup(batch_item)#[bs,dim]

			keys=self.att(all_user_embedding) # [user_num,dim]->[user_num,dim]
			# print(f"keys.shape:{keys.shape}")
			if self.args.Social.sigmoid_norm:
				qk=torch.sigmoid(torch.mm(batch_item_embedding,torch.transpose(keys,1,0))) #[bs,dim]x[dim,user_num]->[bs,user_num] #NOTE:这里其实存在一个归一化和保证非负性的问题，考虑使用sigmoid，因为softmax会平均包括负样本的所有信息
			else:
				qk=torch.mm(batch_item_embedding,torch.transpose(keys,1,0))
			
			masked_qk=torch.mul(qk,batch_UserItemNet) #[bs,user_num] 把负样本设置为0
			batch_items_relation=torch.mm(masked_qk,all_user_embedding) #[bs,user_num]x[user_num,dim]->[bs,dim]
			# print(f"batch_item_relation.shape:{batch_items_relation.shape}")
			items_relation=torch.cat([items_relation,batch_items_relation],dim=0) #final:[item_num,dim]

		self.items_relation=items_relation

	def _get_relation_fast(self):
		'''
		改良版本,利用矩阵和mask加速
		对于小数据集比较适用, 大数据集显存会不够
		'''
		all_user_id=torch.arange(self.user_num).to(self.args.basic.device)
		all_item_id=torch.arange(self.item_num).to(self.args.basic.device)
		all_user_embedding=self.Suser_lookup(all_user_id)
		all_item_embedding=self.Sitem_lookup(all_item_id)
		UserItemNet_mask=torch.from_numpy(self.UserItemNet.toarray()).to(self.args.basic.device) # [user_num,item_num]
		keys=self.att(all_user_embedding) # [user_num,dim]->[user_num,dim]
		
		if self.args.Social.sigmoid_norm:
			qk=torch.sigmoid(torch.mm(all_item_embedding,torch.transpose(keys,1,0))) #[item_num,dim]x[dim,user_num]->[item_num,user_num] #NOTE:这里其实存在一个归一化和保证非负性的问题，考虑使用sigmoid，因为softmax会平均包括负样本的所有信息
		else:
			qk=torch.mm(all_item_embedding,torch.transpose(keys,1,0))

		masked_qk=torch.mul(qk,torch.transpose(UserItemNet_mask,1,0)) #[item_num,user_num] 把负样本设置为0
		items_relation=torch.mm(masked_qk,all_user_embedding) #[item_num,user_num]x[user_num,dim]->[item_num,dim]

		self.items_relation=items_relation

	def _get_relation_fast_mean(self):
		'''
		改良版本,利用矩阵和mask加速
		对于小数据集比较适用, 大数据集显存会不够
		不适用attention, 使用mean聚合
		'''
		all_user_id=torch.arange(self.user_num).to(self.args.basic.device)
		all_item_id=torch.arange(self.item_num).to(self.args.basic.device)
		all_user_embedding=self.Suser_lookup(all_user_id)
		all_item_embedding=self.Sitem_lookup(all_item_id)
		UserItemNet_mask=torch.from_numpy(self.UserItemNet.toarray()).to(self.args.basic.device) # [user_num,item_num]
		
		items_relation=torch.mm(torch.transpose(UserItemNet_mask,1,0),all_user_embedding) #[item_num,user_num]x[user_num,dim]->[item_num,dim]
		count_nonzero=torch.count_nonzero(torch.transpose(UserItemNet_mask,1,0),dim=1).reshape(-1,1) #[item_num,1]
		mean_items_relation=items_relation/count_nonzero #[item_num,dim]


		self.items_relation=mean_items_relation

	def _get_relation(self):
		'''
		根据不同规模的数据集, 选择不同的relation generation方式
		'''
		if self.args.basic.dataset_name=='movie':
			# self._get_relation_fast_mean()
			self._get_relation_fast()
		elif self.args.basic.dataset_name=='book':
			self._get_relation_slow()
			# self._get_relation_median()
		elif self.args.basic.dataset_name=='music':
			self._get_relation_fast()
		else:
			raise AssertionError("Don't define this method. ")


	def forward(self, user, p_item,n_item):

		# MF
		Puser_embedding=self.Puser_lookup(user) #[bs,dim]
		Pp_item_embedding=self.Pitem_lookup(p_item)
		Pn_item_embedding=self.Pitem_lookup(n_item)

		if self.args.Personal.use_mlp:
			Puser_embedding=self.Puser_mlp(Puser_embedding)
			Pp_item_embedding=self.Pitem_mlp(Pp_item_embedding)
			Pn_item_embedding=self.Pitem_mlp(Pn_item_embedding)
			Pp_scores=torch.sum(torch.mul(Puser_embedding,Pp_item_embedding),dim=1) #[bs]
			Pn_scores=torch.sum(torch.mul(Puser_embedding,Pn_item_embedding),dim=1) #[bs]
		else:
			Pp_scores=torch.sum(torch.mul(Puser_embedding,Pp_item_embedding),dim=1) #[bs]
			Pn_scores=torch.sum(torch.mul(Puser_embedding,Pn_item_embedding),dim=1) #[bs]

		
		#------------social-----------
		self._get_relation()
		Sp_item_embedding=self.Sitem_lookup(p_item) #[bs,dim]
		Sn_item_embedding=self.Sitem_lookup(n_item)

		p_item_relation=self.items_relation[p_item]
		n_item_relation=self.items_relation[n_item]
		Suser_embedding=self.Suser_lookup(user)

		p_head_embedding=Suser_embedding + p_item_relation #[bs,dim]
		n_head_embedding=Suser_embedding + n_item_relation

		Sp_scores=torch.sum(torch.mul(p_head_embedding,Sp_item_embedding),dim=1)#[bs,dim]->[bs]
		Sn_scores=torch.sum(torch.mul(n_head_embedding,Sn_item_embedding),dim=1)
		
		return Pp_scores,Pn_scores,Sp_scores,Sn_scores,user
	
	def criterion(self,output):

		def Cur_compute(sti, median, slope_r, slope_p, offset,bias):
			R=1/(1+torch.exp(-slope_r*(sti+(offset-median))))
			P=1/(1+torch.exp(slope_p*(sti-(median+offset))))
			Cur_scores=R+P+bias
			return Cur_scores
		
		Pp_scores,Pn_scores,Sp_scores,Sn_scores,user_id=output #[bs]
		
		if self.Cur_mining:

			pp_scores,pn_scores,sp_scores,sn_scores = Pp_scores.sigmoid(),Pn_scores.sigmoid(),Sp_scores.sigmoid(),Sn_scores.sigmoid()

			p_sti=torch.abs(pp_scores-sp_scores) #[bs]
			n_sti=torch.abs(pn_scores-sn_scores)

			paras=self.para_lookup[user_id]#[bs,2]

			median,mean,std=paras[:,0],paras[:,1],paras[:,2]

			alpha=self.args.Cur.alpha
			bias=self.args.Cur.bias

			if self.args.Cur.pivot_mode=='median':
				p_Cur=Cur_compute(p_sti,median,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
				n_Cur=Cur_compute(n_sti,median,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			elif self.args.Cur.pivot_mode=='mean':
				p_Cur=Cur_compute(p_sti,mean,10*torch.exp(-alpha*(mean)),10*torch.exp(-alpha*(1-mean)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
				n_Cur=Cur_compute(n_sti,mean,10*torch.exp(-alpha*(mean)),10*torch.exp(-alpha*(1-mean)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			elif self.args.Cur.pivot_model=='fusion':
				p_Cur=Cur_compute(p_sti,mean,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
				n_Cur=Cur_compute(n_sti,mean,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			else:
				raise AssertionError("Dont't define this pivot_model. ")
			
			p_Cur=torch.clamp(p_Cur,min=0) # 剔除小于0的项, 这样bias可以设置为-1, 保证Cur in [0,1]
			n_Cur=torch.clamp(n_Cur,min=0)
			Cur=(p_Cur+n_Cur)/2

			if self.args.Cur.fusion=='v1':
				# fuse curiosity into NCF version 1
				loss_personal=torch.mean(Cur*F.softplus(Pn_scores-Pp_scores))
				loss_social=torch.mean(Cur*F.softplus(Sn_scores-Sp_scores))
				loss=self.args.losses.Cur_v1.P_weight*loss_personal + self.args.losses.Cur_v1.S_weight*loss_social
			else:
				raise AssertionError("Dont't define this Cur_fusion_model. ")
		else:
			if self.args.basic.criterion_mode=='v1':
				loss_personal=torch.mean(F.softplus(Pn_scores-Pp_scores))
				loss_social=torch.mean(F.softplus(Sn_scores-Sp_scores))
				loss=self.args.losses.v1.P_weight*loss_personal + self.args.losses.v1.S_weight*loss_social
			else:
				raise AssertionError("Dont't define this NCF_criterion mode. ")
		return loss
	
	def test(self,batch_user_id): #[bs]
		all_items_id=torch.arange(self.item_num).to(self.args.basic.device) #[item_num]

		# MF
		all_Pitem_embedding=self.Pitem_lookup(all_items_id) #[item_num,dim]
		batch_Puser_embedding=self.Puser_lookup(batch_user_id) #[bs,dim]
		
		if self.args.Personal.use_mlp:
			batch_Puser_embedding=self.Puser_mlp(batch_Puser_embedding)
			all_Pitem_embedding=self.Pitem_mlp(all_Pitem_embedding)

			P_score=torch.mm(batch_Puser_embedding,torch.transpose(all_Pitem_embedding,1,0)) #[bs,dim]x[dim,item_num]->[bs,item_num]
		else:
			P_score=torch.mm(batch_Puser_embedding,torch.transpose(all_Pitem_embedding,1,0)) #[bs,dim]x[dim,item_num]->[bs,item_num]

		#--------------Social-----------------
		Suser_embedding=self.Suser_lookup(batch_user_id).unsqueeze(1)#[bs,dim]->[bs,1,dim]
		items_relation=self.items_relation[all_items_id]#[items_num,dim]
		Sitem_embedding=self.Sitem_lookup(all_items_id)#[items_num,dim]

		head_embedding=Suser_embedding + items_relation #[bs,1,dim]+[items_num,dim]->[bs,items_num,dim] 
		head_embedding=head_embedding.float()

		S_scores=torch.sum(torch.mul(head_embedding,Sitem_embedding),dim=2) #[bs,itmes_num,dim]*[items_num,dim]->[bs,items_num,dim]->[bs,items_num]

		# post process
		if self.args.test.post_process=='sigmoid':
			return P_score.sigmoid(),S_scores.sigmoid() #[bs,dim]x[dim,item_num]->[bs,item_num]
		elif self.args.test.post_process=='none':
			return P_score,S_scores
		else:
			raise AssertionError("Don't define this mode. ")

	
	def _for_cur(self,user_id,items_id): #user_id=[1],items_id=[n]
		items_id=torch.from_numpy(items_id).to(self.args.basic.device) #[n]
		user_id=torch.from_numpy(user_id).to(self.args.basic.device)

		# MF
		Pitem_embedding=self.Pitem_lookup(items_id)
		Puser_embedding=self.Puser_lookup(user_id)

		if self.args.Personal.use_mlp:
			Puser_embedding=self.Puser_mlp(Puser_embedding)
			Pitem_embedding=self.Pitem_mlp(Pitem_embedding)
			P_score=torch.sum(torch.mul(Puser_embedding,Pitem_embedding),dim=1).sigmoid() #[1,dim]*[n,dim]->[n]

		else:
			P_score=torch.sum(torch.mul(Puser_embedding,Pitem_embedding),dim=1).sigmoid() #[1,dim]*[n,dim]->[n]

		#--------------Social-----------------
		Suser_embedding=self.Suser_lookup(user_id).unsqueeze(0)#[dim]->[1,dim]
		items_relation=self.items_relation[items_id]#[n,dim]
		Sitem_embedding=self.Sitem_lookup(items_id)#[n,dim]

		head_embedding=Suser_embedding+items_relation #[1,dim]+[n,dim]->[n,dim] 
		head_embedding=head_embedding.float()

		# scores_s=torch.sigmoid(torch.mm(head_embedding,torch.transpose(items_embedding,1,0)))#[bs,dim]x[dim,items_num]->[bs,items_num]
		S_score=torch.sum(torch.mul(head_embedding,Sitem_embedding),dim=1).sigmoid() #[n,dim]*[n,dim]->[n,dim]->[n]

		sti=torch.abs(P_score-S_score)
		median=torch.median(sti)
		mean=torch.mean(sti)
		std=torch.std(sti)
		return sti,median,mean,std #[n]
