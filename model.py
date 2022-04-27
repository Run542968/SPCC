import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class NCF(nn.Module):
	def __init__(self, user_num, item_num,train_user_item,UserItemNet,args,GMF_model=None, MLP_model=None):
		super(NCF, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""		
		self.args=args
		self.dropout = args.NCF.dropout
		self.model = args.NCF.mode
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model
		self.factor_num=args.NCF.dim
		self.layers_num=args.NCF.layers_num
		self.item_num=item_num
		self.user_num=user_num
		self.UserItemNet=UserItemNet

		# self.train_user_item=train_user_item
		# self.train_item_user=self._transform_dict(self.train_user_item)

		self.embed_user_GMF = nn.Embedding(user_num, self.factor_num)
		self.embed_item_GMF = nn.Embedding(item_num, self.factor_num)
		self.embed_user_MLP = nn.Embedding(
				user_num, self.factor_num * (2 ** (self.layers_num - 1)))
		self.embed_item_MLP = nn.Embedding(
				item_num, self.factor_num * (2 ** (self.layers_num - 1)))

		MLP_modules = []
		for i in range(self.layers_num):
			input_size = self.factor_num * (2 ** (self.layers_num - i))
			MLP_modules.append(nn.Dropout(p=self.dropout))
			MLP_modules.append(nn.Linear(input_size, input_size//2))
			MLP_modules.append(nn.ReLU())
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = self.factor_num 
		else:
			predict_size = self.factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)
		self.loss_function=nn.BCEWithLogitsLoss()

		# modeling relation for social
		self.item_rel_lookup=nn.Embedding(self.item_num,self.factor_num)	
		self.user_rel_lookup=nn.Embedding(self.user_num,self.factor_num)
		self.att=nn.Linear(self.factor_num,self.factor_num)

		# cur
		self.para_lookup=None
		self.Cur_mining=False
		self.Huber_loss=nn.SmoothL1Loss()

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)
    
	# def _transform_dict(self,train_user_item):
		
	# 	train_item_user=dict()
		
	# 	for user in train_user_item.keys():
	# 		for item in train_user_item[user]:
	# 			if item not in train_item_user.keys():
	# 				train_item_user[item]=set()
	# 			train_item_user[item].add(user)
				
	# 	return train_item_user

	# def _get_relation(self):
	# 	'''
	# 	初始for循环版本,没利用矩阵
	# 	'''
	# 	train_item_user_dict=self.train_item_user
    #     # print("bagin _get_ralation")

	# 	items_relation=torch.from_numpy(np.array([])).to(self.args.basic.device)
	# 	for item in range(self.item_num):
	# 		item_id=torch.from_numpy(np.array(item)).to(self.args.basic.device)#[1]
	# 		# print("item_id:",item_id)
	# 		users_id=torch.from_numpy(np.array(list(train_item_user_dict[item]))).to(self.args.basic.device)#[user_nums]
	# 		# print("user_id:",users_id)
	# 		item_embedding=self.item_rel_lookup(item_id).unsqueeze(0)#[1,dim]
	# 		# print("item_embedding:",item_embedding)
	# 		users_embedding=self.user_rel_lookup(users_id)#[user_nums,dim]
	# 		# print("users_embedding:",users_embedding)
	# 		keys=self.att(users_embedding)#[user_nums,dim]

	# 		qk=torch.mm(item_embedding,torch.transpose(keys,1,0))#[1,user_nums]
	# 		relation=torch.mm(qk,users_embedding)#[1,dim]
	# 		items_relation=torch.cat((items_relation,relation),dim=0)

	# 	self.items_relation=items_relation.to(self.args.basic.device)
	
	def _get_relation(self):
		'''
		改良版本,利用矩阵和mask
		'''
		all_user_id=torch.arange(self.user_num).to(self.args.basic.device)
		all_item_id=torch.arange(self.item_num).to(self.args.basic.device)
		all_user_embedding=self.user_rel_lookup(all_user_id)
		all_item_embedding=self.item_rel_lookup(all_item_id)
		UserItemNet_mask=torch.from_numpy(self.UserItemNet.toarray()).to(self.args.basic.device) # [user_num,item_num]
		keys=self.att(all_user_embedding) # [user_num,dim]->[user_num,dim]
		
		if self.args.Social.sigmoid_norm:
			qk=torch.sigmoid(torch.mm(all_item_embedding,torch.transpose(keys,1,0))) #[item_num,dim]x[dim,user_num]->[item_num,user_num] #NOTE:这里其实存在一个归一化和保证非负性的问题，考虑使用sigmoid，因为softmax会平均包括负样本的所有信息
		else:
			qk=torch.mm(all_item_embedding,torch.transpose(keys,1,0))

		masked_qk=torch.mul(qk,torch.transpose(UserItemNet_mask,1,0)) #[item_num,user_num] 把负样本设置为0
		items_relation=torch.mm(masked_qk,all_user_embedding) #[item_num,user_num]x[user_num,dim]->[item_num,dim]

		self.items_relation=items_relation

	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF #[bs,factor]
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user) #[bs,4*factor]
			embed_item_MLP = self.embed_item_MLP(item) #[bs,4*factor]
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1) #[bs,8*factor]
			output_MLP = self.MLP_layers(interaction) #[bs,8*factor]->[bs,factor]

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1) #[bs,2*factor]

		prediction = self.predict_layer(concat).squeeze() #[bs,1]->[bs]



		#------------social-----------
		self._get_relation()
		item_embedding=self.item_rel_lookup(item) #[bs,dim]
		item_relation=self.items_relation[item]
		user_embedding=self.user_rel_lookup(user)

		head_embedding=user_embedding+item_relation #[bs,dim]
		rel_scores=torch.sum(torch.mul(head_embedding,item_embedding),dim=1)#[bs,dim]->[bs]
		
		return prediction,rel_scores,user
	
	def criterion(self,output,label):

		def Cur_compute(sti, median, slope_r, slope_p, offset,bias):
			R=1/(1+torch.exp(-slope_r*(sti+(offset-median))))
			P=1/(1+torch.exp(slope_p*(sti-(median+offset))))
			Cur_scores=R+P+bias
			return Cur_scores
		
		prediction,rel_scores,user_id=output #[bs]
		
		if self.Cur_mining:

			prediction=prediction.sigmoid()
			rel_scores=rel_scores.sigmoid()

			sti=torch.abs(prediction-rel_scores) #[bs]
			paras=self.para_lookup[user_id]#[bs,2]

			median,mean,std=paras[:,0],paras[:,1],paras[:2]

			alpha=self.args.Cur.alpha
			bias=self.args.Cur.bias

			if self.args.Cur.pivot_mode=='median':
				Cur=Cur_compute(sti,median,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			elif self.args.Cur.pivot_mode=='mean':
				Cur=Cur_compute(sti,mean,10*torch.exp(-alpha*(mean)),10*torch.exp(-alpha*(1-mean)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			elif self.args.Cur.pivot_model=='fusion':
				Cur=Cur_compute(sti,mean,10*torch.exp(-alpha*(median)),10*torch.exp(-alpha*(1-median)),0.5-std,bias).detach()# 从梯度图上摘出去，这只是个权重，不参与更新
			else:
				assert("Dont't define this pivot_model. ")
			
			Cur=torch.clamp(Cur,min=0) # 剔除小于0的项, 这样bias可以设置为-1, 保证Cur in [0,1]

			if self.args.Cur.fusion=='v1':
				# fuse curiosity into NCF version 1
				loss_ncf=torch.mean(-Cur*(label*torch.log(prediction+1e-10) + (1-label)*torch.log(1-prediction+1e-10)))
				loss_rel=torch.mean(-Cur*(label*torch.log(rel_scores+1e-10) + (1-label)*torch.log(1-rel_scores+1e-10)))
				loss=self.args.losses.v1.ncf_weight*loss_ncf + self.args.losses.v1.social_weight*loss_rel
			elif self.args.Cur.fusion=='v2':
				# fuse curiosity into NCF version 2
				loss_acc = self.loss_function((prediction+rel_scores)/2, label)
				loss_cur = torch.mean(-Cur*torch.log((prediction+rel_scores)/2+1e-10))
				loss=self.args.losses.v2.acc_weight*loss_acc + self.args.losses.v2.cur_weight*loss_cur
			elif self.args.Cur.fusion=='v3':
				# fuse curiosity into NCF version 3
				loss_ncf=torch.mean(-(label*torch.log(prediction+1e-10) + (1-label)*torch.log(1-prediction+1e-10)))
				loss_rel=torch.mean(-(label*torch.log(rel_scores+1e-10) + (1-label)*torch.log(1-rel_scores+1e-10)))
				loss_acc = self.args.losses.v3.ncf_weight*loss_ncf + self.args.losses.v3.social_weight*loss_rel
				loss_cur = self.Huber_loss((prediction+rel_scores)/2,Cur)
				loss=self.args.losses.v3.acc_weight*loss_acc + self.args.losses.v3.cur_weight*loss_cur
			elif self.args.Cur.fusion=='v4':
				# fuse curiosity into NCF version 4
				loss_ncf=torch.mean(-(label*torch.log(prediction+1e-10) + (1-label)*torch.log(1-prediction+1e-10)))
				loss_rel=torch.mean(-(label*torch.log(rel_scores+1e-10) + (1-label)*torch.log(1-rel_scores+1e-10)))
				loss_acc = self.args.losses.v4.ncf_weight*loss_ncf + self.args.losses.v4.social_weight*loss_rel
				loss_cur = self.args.losses.v4.ncf_huber*self.Huber_loss(prediction,Cur) + self.args.losses.v4.social_huber*self.Huber_loss(rel_scores,Cur)
				loss=self.args.losses.v4.acc_weight*loss_acc + self.args.losses.v4.cur_weight*loss_cur
			else:
				assert("Dont't define this Cur_fusion_model. ")
		else:
			if self.args.NCF.criterion=='v1':
				loss_ncf = self.loss_function(prediction, label)
				loss_rel = self.loss_function(rel_scores,label)
				loss=self.args.losses.v1.ncf_weight*loss_ncf + self.args.losses.v1.social_weight*loss_rel
			elif self.args.NCF.criterion=='v2':
				loss = self.loss_function((prediction+rel_scores)/2, label)
			else:
				assert("Dont't define this NCF_criterion mode. ")
		return loss
	
	def test(self,batch_user_id): #[bs]
		all_items_id=torch.arange(self.item_num).to(self.args.basic.device) #[item_num]
		
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(batch_user_id).unsqueeze(1) #[bs,factor]->[bs,1,factor]
			embed_item_GMF = self.embed_item_GMF(all_items_id) #[item_num,factor]
			output_GMF = embed_user_GMF * embed_item_GMF #[bs,item_num,factor]
		
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(batch_user_id) #[bs,4*factor]
			embed_item_MLP = self.embed_item_MLP(all_items_id) #[item_num,4*factor]

			embed_user_MLP=embed_user_MLP.unsqueeze(1).repeat((1,len(all_items_id),1)) #[bs,item_num,4*factor]
			embed_item_MLP=embed_item_MLP.unsqueeze(0).repeat((len(batch_user_id),1,1)) #[bs,item_num,4*factor]

			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1) #[bs,item_num,8*factor]
			output_MLP = self.MLP_layers(interaction) #[bs,item_num,8*factor]->[bs,item_num,factor]

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1) #[bs,item_num,2*factor]

		prediction = self.predict_layer(concat).squeeze() #[bs,item_num,2*factor]->[bs,item_num,1]->[bs,item_num]
		


		#--------------Social-----------------
		users_embedding=self.user_rel_lookup(batch_user_id).unsqueeze(1)#[bs,dim]->[bs,1,dim]
		items_relation=self.items_relation[all_items_id]#[items_num,dim]
		items_embedding=self.item_rel_lookup(all_items_id)#[items_num,dim]

		head_embedding=users_embedding+items_relation #[bs,1,dim]+[items_num,dim]->[bs,items_num,dim] 
		head_embedding=head_embedding.float()

		# scores_s=torch.sigmoid(torch.mm(head_embedding,torch.transpose(items_embedding,1,0)))#[bs,dim]x[dim,items_num]->[bs,items_num]
		scores_s=torch.sum(torch.mul(head_embedding,items_embedding),dim=2) #[bs,itmes_num,dim]*[items_num,dim]->[bs,items_num,dim]->[bs,items_num]

		return prediction,scores_s #[bs]
	
	def _for_cur(self,user_id,items_id): #user_id=[1],items_id=[n]
		items_id=torch.from_numpy(items_id).to(self.args.basic.device) #[n]
		user_id=torch.from_numpy(user_id).to(self.args.basic.device)

		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user_id).unsqueeze(0) #[factor]->[1,factor]
			# print(f"embed_user_GMF.shape:{embed_user_GMF.shape}")
			embed_item_GMF = self.embed_item_GMF(items_id) #[n,factor]
			output_GMF = embed_user_GMF * embed_item_GMF #[n,factor]
			# print(f"output_GMF.shape:{output_GMF.shape}")
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user_id).unsqueeze(0) #[4*factor]->[1,4*factor]
			embed_item_MLP = self.embed_item_MLP(items_id) #[n,4*factor]

			embed_user_MLP=embed_user_MLP.repeat((len(items_id),1)) #[n,4*factor]

			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1) #[n,8*factor]
			# print(f"interaction.shape:{interaction.shape}")
			output_MLP = self.MLP_layers(interaction) #[n,8*factor]->[n,factor]

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1) #[n,2*factor]

		prediction = self.predict_layer(concat).squeeze().sigmoid() #[n,2*factor]->[n,1]->[n]
		


		#--------------Social-----------------
		users_embedding=self.user_rel_lookup(user_id).unsqueeze(0)#[dim]->[1,dim]
		items_relation=self.items_relation[items_id]#[n,dim]
		items_embedding=self.item_rel_lookup(items_id)#[n,dim]

		head_embedding=users_embedding+items_relation #[1,dim]+[n,dim]->[n,dim] 
		head_embedding=head_embedding.float()

		# scores_s=torch.sigmoid(torch.mm(head_embedding,torch.transpose(items_embedding,1,0)))#[bs,dim]x[dim,items_num]->[bs,items_num]
		scores_s=torch.sum(torch.mul(head_embedding,items_embedding),dim=1).sigmoid() #[n,dim]*[n,dim]->[n,dim]->[n]

		sti=torch.abs(prediction-scores_s)
		median=torch.median(sti)
		mean=torch.mean(sti)
		std=torch.std(sti)
		return sti,median,mean,std #[n]
