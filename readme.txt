# 打包命令
cd SPCC
zip -r SPCC_backup_2021.12.29.zip ./* 

# 在Powershell将远程复制到本地
scp student@202.38.247.208:/home/student/dujiarun/SPCC/SPCC_backup_2021.12.29.zip D:\杜佳润\本科\软工三班\许可老师SRP项目\KG_Recommendation研究\SPCC_backup

# 在Powershell将本地复制到远程
scp -r D:\杜佳润\本科\软工三班\许可老师SRP项目\KG_Recommendation研究\SPCC_backup\SPCC_backup.zip student@202.38.247.208:/home/student/dujiarun/

# git命令
git status
git add -A #把所有修改/未追踪修改添加到暂存区
git status
git commit -m "comment" 
git push origin dev:dev


--------------------------
现在不用这样了，直接sftp插件完美替代


-----------------code版本记录-------------------
上一个版本：
    数据格式是(user,item) pairs，使用三层网络从sti学习出Cur，（第十六次组会前的版本）
2022.3.4：
    数据格式是(user:item,item,item)
    通过寻找user对应的item的sti中值作为Wundt的最高点，左右两侧设计一个网络学习：越靠近中值Cur越高
    模型仍然使用NCF和LightGCN
    效果很差，存在梯度爆炸问题，(user,item)pair太多了————放弃
2022.3.14:
    数据格式是(user,item,label,hitRate)
    通过寻找user对应的item的sti中值作为Wundt的最高点，左右两侧设计一个网络学习：越靠近中值Cur越高
    模型仍然使用NCF和LightGCN
2022.3.21:
    数据格式发生变化(v5): 训练负样本和训练正样本数目相同，并且测试的样本是该用户训练阶段没见过的数据
2022.4.1-(当前版本):
    使用BPR_loss,同时使用BPR的数据格式(final)