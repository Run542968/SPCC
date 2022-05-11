import torch
import os
from model import SPCC
import options
from Dataset import Spcc_Dataset
from torch.utils.data import DataLoader
from train import train
from test import test
from utils.Utils import get_logger,setup_seed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import shutil
from Cur import Cur_mining

if __name__=="__main__":
   args=options._process_args_over_yaml()

   # contruct logger
   if not os.path.exists('./logs/'):
      os.makedirs('./logs/')
   if os.path.exists('./logs/' + args.basic.model_name+'.log'):
      os.remove('./logs/' + args.basic.model_name+'.log')
   logger=get_logger('./logs/',args.basic.model_name)

   # contruct summary
   if os.path.exists('./summary/' + args.basic.model_name):
      shutil.rmtree('./summary/' + args.basic.model_name)
   writer = SummaryWriter('./summary/' + args.basic.model_name)

   # contruct ckpt file
   if os.path.exists('./ckpt/' + args.basic.model_name):
      shutil.rmtree('./ckpt/' + args.basic.model_name)
   if not os.path.exists('./ckpt/'+args.basic.model_name+'/'):
      os.makedirs('./ckpt/'+args.basic.model_name+'/')

   # setup_seed
   setup_seed(args.basic.seed)
   print('=============seed: {}, pid: {}============='.format(args.basic.seed,os.getpid()))
   
   logger.info('=============seed: {}, pid: {}============='.format(args.basic.seed,os.getpid()))
   logger.info(args)

   # load train dataset、Cur_dataset
   dataset=Spcc_Dataset(args)
   train_loader=DataLoader(
                           dataset=dataset,
                           batch_size=args.basic.batch_size,
                           shuffle=True,
                           num_workers=args.basic.num_workers,
                           drop_last=False)

   # load model
   model = SPCC(dataset.user_num, 
               dataset.item_num,
               dataset.train_user_item,
               dataset.UserItemNet,
               args)
   model=model.to(args.basic.device)
   logger.info(model)


   # load optimizer
   optimizer=torch.optim.Adam(model.parameters(),lr=args.optim.lr)

   # model._get_relation()
   # test(dataset,model,logger,writer,0,args)
   # Cur_mining(model,dataset,args)

   # train and eval
   for epoch in tqdm(range(args.basic.epoch_num)):

      logger.info("-------------------------Training------------------------------")
      train(train_loader,model,optimizer,logger,writer,epoch,args)
        
      if (epoch+1) % args.test.interval == 0:
         logger.info("-------------------------Testing------------------------------")
         model._get_relation() # get relation with updated parameters
         test(dataset,model,logger,writer,epoch,args)
      
      if args.Cur.mining_mode=='v1':
         if (epoch+1) % args.Cur.mine_interval == 0: # 指定interval挖掘,之后的epoch都用这次挖掘的结果
            logger.info("-------------------------Cur Mining------------------------------") 
            Cur_mining(model,dataset,args)
      elif args.Cur.mining_mode=='v2':
         if (epoch+1) >= args.Cur.mine_interval: # 大于某个setting后，每个epoch挖掘一次
            logger.info("-------------------------Cur Mining------------------------------") 
            Cur_mining(model,dataset,args)
      
      # save model and cur_model
      ckpt_path = './ckpt/'+args.basic.model_name+'/'+'Epoch_'+str(epoch)+'.pkl'
      torch.save(model.state_dict(), ckpt_path)

