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

## update Cur_mining(中值靠近左侧，左侧斜率越大), 这以下用cur进行re-weighting
# bias=0; offset=0.5-std; no sigmoid normalization; fix testing about _get_relation(); bi-sigmoid testing; cur_mining when epoch equals 80, and use this result for later
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_11 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100
# bias=0; offset=0.5-std; sigmoid normalization; fix testing about _get_relation(); bi-sigmoid testing; cur_mining when epoch equals 80, and use this result for later
CUDA_VISIBLE_DEVICES=0 python main.py --model-name NCF_Cur_12 --dataset_name 'movie'  --batch_size 16384 --test.interval 1 --optim.lr 0.02 --epoch_num 100

## 下面是把Cur作为回归标签
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

## 新增了一种v5的好奇心融合方式, 设置一个阈值把Cur作为伪标签
# --Cur.fusion 'v5'; --losses.v5.cur_weight 0.01
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_32 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v5' --losses.v5.cur_weight 0.01 --Cur.pivot_mode 'mean'
# --Cur.fusion 'v5'; --losses.v5.cur_weight 0.001
CUDA_VISIBLE_DEVICES=4 python main.py --model-name NCF_Cur_33 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v5' --losses.v5.cur_weight 0.001 --Cur.pivot_mode 'mean'
# --Cur.fusion 'v6'; --losses.v5.cur_weight 0.01
CUDA_VISIBLE_DEVICES=5 python main.py --model-name NCF_Cur_34 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v6' --losses.v6.cur_weight 0.01 --Cur.pivot_mode 'mean'
# --Cur.fusion 'v5'; --losses.v5.cur_weight 0.01; epoch_num 200
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_35 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v5' --losses.v5.cur_weight 0.01 --Cur.pivot_mode 'mean'

## 用Cur做re-weighting
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_36 --dataset_name 'movie'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 100 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'




### Amazon Book
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_Book_0 --dataset_name 'book'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean' --test.batch_size 20 --Social.relation_batch 100

### LastFM
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_music_0 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 80 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# 去掉Cur mining
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_music_1 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 800 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# 去掉Cur mining; lr=0.01
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_music_2 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.01 --epoch_num 200 --Cur.mine_interval 800 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# 去掉Cur mining; --losses.v1.social_weight=0.1
CUDA_VISIBLE_DEVICES=3 python main.py --model-name NCF_Cur_music_3 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 800 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean' --losses.v1.social_weight=0.1
# 去掉Cur mining; epoch_num=500; 看看模型的性能瓶颈在哪里
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_music_4 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 500 --Cur.mine_interval 800 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# Cur mining; --Cur.mine_interval 180
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_music_5 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 200 --Cur.mine_interval 180 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# Cur mining; --Cur.mine_interval 480; 看看性能瓶颈在哪里
CUDA_VISIBLE_DEVICES=1 python main.py --model-name NCF_Cur_music_6 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 500 --Cur.mine_interval 480 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
# Cur mining; --Cur.mine_interval 380; 看看性能瓶颈在哪里
CUDA_VISIBLE_DEVICES=2 python main.py --model-name NCF_Cur_music_7 --dataset_name 'music'  --batch_size 8192 --test.interval 1 --optim.lr 0.02 --epoch_num 500 --Cur.mine_interval 380 --NCF.criterion 'v1' --Cur.fusion 'v1' --Cur.pivot_mode 'mean'
