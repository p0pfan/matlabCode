function [train_ans,test_predict, model ]= lssvm_crossvalidate(train,train_out,test,test_o)
%% 该代码为基于lssvm—crossvalidate的预测算法

% train : 5 *14 一行表示一个样本
% train_result 5 *2 一行表示一个结果

%% 归一化
% 需要注意的是，这里进行归一化的时候，
% 是按列归一化，还是按行归一化？
[train_data,pstrain0] = mapminmax(train,0,1);
[test_data] = mapminmax('apply',test,pstrain0);
[train_result,pstrain1] = mapminmax(train_out',0,1);
[test_out] = mapminmax('apply',test_o(1,:)',pstrain1);

train_data = train_data';
train_result = train_result';
test_data = test_data';
test_out = test_out';

%% 建立lssvm模型
type='f'
% gam=75;
gam=[15 10];
% sig2=10;
sig2=[4.5 5];                                                    
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
train_ans = [train_predict test_predict];
%计算均方差
% trainmse=sum((train_predict-train_out).^2)/length(train_result)
% testmse=sum((test_predict-test_out).^2)/length(test_result) 
% besttestmse=testmse;

% test_predict = predict_left(model,test_data,train_predict,5,input);


% 
% figure(1)
% plot([1:6],test_out,'r-o',[1:6],test_predict,'b-*');
% % 
% figure(2)
% plot([1:5],train_predict,'r-o',[1:5],train_out,'b-');
end