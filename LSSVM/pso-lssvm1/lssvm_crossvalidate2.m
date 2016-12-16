function [train_predict, model ]= lssvm_crossvalidate(train,train_out,test,test_out)
%% 该代码为基于lssvm—crossvalidate的预测算法
%% 清空环境变量
% clc
% close all
% clear

%%%% 训练数据预测数据提取及归一化
%% 清空环境
% clc
% clear

%% 导入训练数据和测试数据
%%训练数据输入、输出
% train = [3000 0.176 0.2 0.2 113.04;
%     6000 0.181 0.2 0.2 226.08;
%     9000 0.186 0.2 0.2 339.12;
%     12000 0.193 0.2 0.2 452.16;
%     15000 0.198 0.2 0.2 565.2;
%     18000 0.202 0.2 0.2 678.24;
%     21000 0.208 0.2 0.2 791.28;
%     24000 0.216 0.2 0.2 904.32;
%     22000 0.05 0.2 0.05 828.96;
%     22000 0.15 0.2 0.15 828.96;
%     22000 0.1 0.2 0.1 828.6;
%     22000 0.2 0.2 0.2 828.96;
%     22000 0.25 0.2 0.25 828.96;
%     22000 0.3 0.2 0.3 828.96;
%     22000 0.35 0.2 0.35 828.96;
%     22000 0.4 0.2 0.4 828.96;
%     22000 0.45 0.2 0.45 828.96];
% train_out = [5.387 2.824 3.414 3.051 2.515 3.312 3.084 4.137 1.519 1.284 2.413 2.012 1.815 2.512 2.914 3.224 5.671];
%%测试数据输入、输出
% test = [4500 0.179 0.2 0.2 169.56;
%     7500 0.184 0.2 0.2 282.6;
%     10500 0.189 0.2 0.2 395.64;
%     13500 0.195 0.2 0.2 508.68;
%     16500 0.2 0.2 0.2 621.72;
%     19500 0.205 0.2 0.2 734.76;
%     22500 0.212 0.2 0.2 847.8];
% test_out = [3.8237 2.9526 3.3824 2.8132 2.1938 2.816 3.0887];
%%归一化方法（利用mapminmax函数归一化）
% train = train';
% train_out = train_out';
% test = test';
% test_out = test_out';
[train_data,pstrain0] = mapminmax(train',0,1);
[test_data] = mapminmax('apply',test',pstrain0);
[train_result,pstrain1] = mapminmax(train_out,0,1);
[test_result] = mapminmax('apply',test_out,pstrain1);

train_data = train_data';
train_result=train_result';
test_data = test_data';

size(train_data)
size(test_data)
%% 建立lssvm模型
type='f'
gam=75;
% gam=[75 50];
sig2=10;
% sig2=[0.45 0.45];
kernel = 'RBF_kernel'
proprecess='function estimation'
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess);
% 交叉验证优化参数
costfun = 'crossvalidatelssvm';
costfun_args = {5,'mse'};  % the value should be an interger
optfun = 'gridsearch';
model = tunelssvm(model,optfun,costfun,costfun_args);   % 模型参数优化
model=trainlssvm(model)
%求出训练集和测试集的预测值
[train_predict_y,zt,model]=simlssvm(model,train_data)
[test_predict_y,zt,model]=simlssvm(model,test_data)
%预测数据反归一化
train_predict=mapminmax('reverse',train_predict_y',pstrain1)
test_predict=mapminmax('reverse',test_predict_y',pstrain1)
%计算均方差
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
besttestmse=testmse;

% figure(1)
% plot([1:6],test_out,'r-o',[1:6],test_predict,'b-*');
% 
figure(2)
plot([1:6],train_predict,'r-o',[1:6],train_out,'b-');
end