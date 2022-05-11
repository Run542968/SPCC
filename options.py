import argparse
import yaml
from attrdict import AttrDict
parser = argparse.ArgumentParser(description='SPCC')
config_blocks=['basic','optim','Personal','Cur','Social','test','losses']

parser.add_argument('--config_yaml',type=str,default='configs/SPCC_movie.yaml',metavar='FILE')

# basic setting
parser.add_argument('--seed', type=int, default=0,help='random seed (default: 0)')
parser.add_argument('--batch_size', type=int, default=8192,help='batch_size')
parser.add_argument('--epoch_num', type=int, default=100,help='epoch_num')
parser.add_argument('--model-name', default='SPCC', help='name to save model')

# # Cur setting
parser.add_argument('--Cur.mine_interval',type=int,default=80)
parser.add_argument('--Cur.fusion',type=str,default='v1',help='the mode of fusing curiosity')
parser.add_argument('--Cur.pivot_mode',type=str,default='mean',help='the mode of pivot setting')


# optim setting
parser.add_argument('--optim.lr',type=float,default=0.02,help='the lr of optimizer')


# Personal setting
parser.add_argument('--Personal.dim',type=int,default=32)
parser.add_argument('--Personal.use_mlp',type=lambda x: x.lower() == 'true',default=False)

# Social setting
parser.add_argument('--Social.dim',type=int,default=32)

# losses setting
parser.add_argument('--losses.v1.P_weight',type=float,default=1)
parser.add_argument('--losses.v1.S_weight',type=float,default=1)

# for testing
parser.add_argument('--test.k', type=int,default=20)
parser.add_argument('--test.interval',type=int,default=1)
parser.add_argument('--test.batch_size',type=int,default=200)
parser.add_argument('--test.alpha',type=float,default=0.5,help="control the rate about the fusion of Ssocre and Pscore")
parser.add_argument('--test.post_process',type=str,default='sigmoid')


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
    print(args.Personal.use_mlp)