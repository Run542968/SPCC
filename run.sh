## test NCF + relation module
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_0 --dataset_name 'movie' --batch_size 16384 --test.interval 1 --optim.lr 0.02
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_1 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.002

# Tanh -> ReLU
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_2 --dataset_name 'movie' --batch_size 16384 --test.interval 1 --optim.lr 0.02
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_3 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.002
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_4 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.05 --epoch_num 100
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_5 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100

# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_13 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); no sigmoid normalization
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_14 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_15 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.01 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_16 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.03 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_17 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.005 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_18 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.001 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_19 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.1 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,per_scores+soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_20 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.05 --epoch_num 100 --Cur.mine_interval 300
# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,(per_scores+soc_scores)/2); sigmoid normalization
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_21 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 300 --NCF.criterion 'v2'


## add Cur_mining(中值越靠近左侧，左侧的斜率越小)
# 只在第一个epoch mining一次,后面一直用这次挖掘过的信息做权重,bias=-1
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_6 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100
# 第80个epoch mining一次,后面一直用这次挖掘过的信息做权重,bias=-1
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_7 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100
# 每个epoch mining一次,bias=-1
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_8 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100

## test acceleration in _get_relation()
# sigmoid normalization,第80个epoch mining一次,后面一直用这次挖掘过的信息做权重,bias=0
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_9 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100
# no normalization,第80个epoch mining一次,后面一直用这次挖掘过的信息做权重,bias=0
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_10 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100

## update Cur_mining(中值靠近左侧，左侧斜率越大)
# bias=0; offset=0.5-std; no sigmoid normalization; fix testing about _get_relation(); bi-sigmoid testing; cur_mining when epoch equals 80, and use this result for later
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_11 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100
# bias=0; offset=0.5-std; sigmoid normalization; fix testing about _get_relation(); bi-sigmoid testing; cur_mining when epoch equals 80, and use this result for later
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_12 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100

# BCE(y,per_scores)+BCE(y,soc_scores) -> BCE(y,(per_scores+soc_scores)/2); sigmoid normalization
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_22 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v2' --Cur.fusion 'v4'
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization
CUDA_VISIBLE_DEVICES=5 python main.py --model-name NCF_Cur_23 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.001
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=5 python main.py --model-name NCF_Cur_24 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.001
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=6 python main.py --model-name NCF_Cur_25 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.00001
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_26 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.01
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_27 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.1
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=4 python main.py --model-name NCF_Cur_28 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.1
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=4 python main.py --model-name NCF_Cur_29 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.001 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.1

## 把median改为mean
# BCE(y,per_scores)+BCE(y,soc_scores); sigmoid normalization; fix a bug of loss_BCE after cur mining(Cur_23少了负号)
CUDA_VISIBLE_DEVICES=7 python main.py --model-name NCF_Cur_30 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 0.1 --Cur.pivot_mode 'mean'
CUDA_VISIBLE_DEVICES=6 python main.py --model-name NCF_Cur_31 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v4' --losses.v4.cur_weight 1 --Cur.pivot_mode 'mean'


## 改一下cur_mining,以上都是在80epoch挖掘一次,然后不挖掘,现在改成80epoch后都挖掘一次