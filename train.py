
import torch

def train(train_loader,model,optimizer,logger,writer,epoch,args):
    
    # change mode
    model.train()
    # negative sampling before training
    train_loader.dataset._train_sampling()

    for i ,batch_data in enumerate(train_loader):
        user_id,p_item,n_item=batch_data[:,0],batch_data[:,1],batch_data[:,2]
        output=model(user_id.to(args.basic.device),
                    p_item.to(args.basic.device),
                    n_item.to(args.basic.device))                                                                         
        loss=model.criterion(output)

        # optimization
        optimizer.zero_grad()
        # backforward
        loss.backward()
        # gradient update
        optimizer.step()

        
        loss=loss.cpu().detach().numpy()
        logger.info("Epoch:{}-batch:{},spcc_total_loss_dict:{}".format(epoch,i,{"spcc_total_loss_dict":loss}))

        writer.add_scalar('loss/spcc_total_loss_dict',loss, epoch*len(train_loader)+i)


