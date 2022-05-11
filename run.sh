### test SPCC
## movie
# base
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_movie_0 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --epoch_num 200
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_movie_1 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_movie_2 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.1
CUDA_VISIBLE_DEVICES=3 python main.py --model-name SPCC_movie_3 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.05
CUDA_VISIBLE_DEVICES=3 python main.py --model-name SPCC_movie_4 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.01
CUDA_VISIBLE_DEVICES=3 python main.py --model-name SPCC_movie_5 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.01 --losses.v1.S_weight 0.5
CUDA_VISIBLE_DEVICES=3 python main.py --model-name SPCC_movie_6 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.01 --losses.v1.S_weight 0.1
CUDA_VISIBLE_DEVICES=3 python main.py --model-name SPCC_movie_7 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.01 --losses.v1.S_weight 2
CUDA_VISIBLE_DEVICES=4 python main.py --model-name SPCC_movie_8 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --optim.lr 0.001 --batch_size 256
CUDA_VISIBLE_DEVICES=4 python main.py --model-name SPCC_movie_9 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --losses.v1.S_weight 0 --test.Sscore_weight 0 
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_movie_10 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --epoch_num 200 --Cur.mine_interval 180


## music
# base
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_music_0 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_music_1 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True --optim.lr 0.1
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_music_2 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True --optim.lr 0.15
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_music_3 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True --optim.lr 0.12
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_music_4 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 300 --Personal.use_mlp True --optim.lr 0.1 --Cur.mine_interval 180
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_music_5 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True --optim.lr 0.09 --Cur.mine_interval 180
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_music_6 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 200 --Personal.use_mlp True --optim.lr 0.05 --Cur.mine_interval 180
CUDA_VISIBLE_DEVICES=1 python main.py --model-name SPCC_music_7 --config_yaml 'configs/SPCC_music.yaml' --test.interval 1 --epoch_num 300 --Personal.use_mlp True --optim.lr 0.1 --Cur.mine_interval 280


## movie
# 用mean代替att
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_movie_mean_0 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True

# 不使用Cur mining
CUDA_VISIBLE_DEVICES=2 python main.py --model-name SPCC_movie_nocur_0 --config_yaml 'configs/SPCC_movie.yaml' --test.interval 1 --Personal.use_mlp True --Cur.mine_interval 5000


