import argparse
import yaml
from attrdict import AttrDict
parser = argparse.ArgumentParser(description='SPCC')
config_blocks=['basic','optim','NCF','Cur','Social','test','losses']

parser.add_argument('--config_yaml',type=str,default='configs/SPCC_NCF_lightGCN.yaml',metavar='FILE')

# basic setting
parser.add_argument('--seed', type=int, default=0,help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=8192,help='batch_size')
parser.add_argument('--epoch_num', type=int, default=50,help='epoch_num')
parser.add_argument('--dataset_name', type=str, default='movie',help='dataset name')

parser.add_argument('--model-name', default='NCF', help='name to save model')

# # Cur setting
parser.add_argument('--Cur.mine_interval',type=int,default=80)
parser.add_argument('--Cur.fusion',type=str,default='v1',help='the mode of fusing curiosity')

# optim setting
parser.add_argument('--optim.lr',type=float,default=0.001,help='the lr of optimizer')


# NCF setting
parser.add_argument('--NCF.mode',type=str,default='NeuMF-end',help='the mode of NCF')
parser.add_argument('--NCF.criterion',type=str,default='v1',help='the mode of loss')


# losses setting
# parser.add_argument('--losses.cur_weight',type=float,default=1,help="the weight of cur loss")
# parser.add_argument('--losses.Social.bpr_weight',type=float,default=1,help="the weight of Social loss")
parser.add_argument('--losses.v4.cur_weight',type=float,default=1,help="the weight of cur loss")


# for testing
parser.add_argument('--test.k', type=int,default=20)
parser.add_argument('--test.interval',type=int,default=1)

def _process_args_over_yaml():
    args=vars(parser.parse_args())
    # default config
    with open(args['config_yaml'], 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    for key in config_blocks:
        if key not in cfg.keys():
            cfg[key]={}

    for (key,value) in args.items():
        key_splits=key.split('.')
        if len(key_splits)==1:
            cfg['basic'][key]=value
        elif len(key_splits)>1 and key_splits[0] in config_blocks:
            tmp_values=[cfg]
            for sp in key_splits[:-1]:
                try:
                    tmp_values.append(tmp_values[-1][sp])
                except ValueError:
                    raise ValueError('Wrong key')
            # to the last but one dict
            tmp_value=value
            for i in range(len(key_splits)-1,-1,-1):
                tmp_values[i][key_splits[i]]=tmp_value
                tmp_value=tmp_values[i]
            cfg=tmp_value
        else:
            raise ValueError('Unexpected Error', key,value)

    cfg = AttrDict(cfg)
    return cfg

if __name__=='__main__':
    args=_process_args_over_yaml()
    print(type(float(args.losses.GCN.reg_weight)))