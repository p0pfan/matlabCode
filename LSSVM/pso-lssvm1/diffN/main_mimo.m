clc
clear all

out__ = '************************************************************'

%% A : 表示浓度
%T : 温度
% 浓度的初始值
% A_std : A的输出方差 
% T_std : T的输出方差
% sample_time : 采样时间

A_0_init = 6.5;
A_0_init1 = 7.5;
A_std = 0.0275;
error = A_std;
T_std = 0.0592;
sample_time = 200;

%% 初始化输入浓度,输入温度
%A_input : sample_time * 1
%T_input : sample_time * 1
A_input = [ones(30,1) * A_0_init ;
            ones(sample_time - 30,1) * A_0_init1];

T_input = ones(sample_time,1)*3.5;

[t, A_measurement_,T_measurement_,truevalue_]  = measurement(A_std,T_std,sample_time);

temp = 2;
tmp =1;
for i = 1 : numel(t)
    if(t(i) == temp)
        A_measurement(tmp,:) = A_measurement_(i,:);
        T_measurement(tmp,:)  = T_measurement_(i,:);
        truevalue(tmp,:)  = truevalue_(i,:);
        temp = temp + 2;
        tmp = tmp +1;
    end
end

%% 训练模型的阶次等参数
mm = 5; 
nn = 2;
windows = 5;

tt = sample_time;
tt =100;

%% 训练结果的输出
% sample_time * output_num
svm_train_ans = zeros(tt,2);

%% 优化结果的存储及设置
% 以列向量进行存储
A_reconcile = zeros(tt, 1);
T_reconcile = zeros(tt,1);

% 初始值的设置
% 我们需要 1～5的真实值 和 6 ～ 10的预测值
A_reconcile(1:10,1) = truevalue(1:2*mm,1);
T_reconcile(1:10,1) = truevalue(1:2*mm,2);

%% STROE PREDICT VALUE
% 用来训练出模型
x_train_predict = truevalue(mm+1:2*mm,1); 
y_train_predict = truevalue(mm+1:2*mm,2);
    
out =zeros(tt,2);
%% START 
for i = 10 : tt
% for i = 10 :tt
    %% STORE TRIAN SAMPLE
    %训练模型的输入
    %每一行表示一个训练样本
    %列数表示一次训练输入的个数
    %5 * 7
    x_train_sample = zeros(windows,mm+nn);
    y_train_sample = zeros(windows,mm+nn);
    
    
    %% get train sample
    s = 0;
    for j = mm:-1:2
        [size(x_train_predict,1)-1-s 1 i-mm i-mm-s i-s-1 i-s-nn ]
        x_train_sample(j,:) =  [ x_train_predict(size(x_train_predict,1)-1-s : -1 :1, 1)' A_reconcile(i-mm : -1 :i-mm-s,1)' A_input(i-s-1 : -1 : i-s-nn , 1)'];
        y_train_sample(j,:) =  [ y_train_predict(size(y_train_predict,1)-1-s : -1 :1, 1)' T_reconcile(i-mm : -1 :i-mm-s,1)' T_input(i-s-1 : -1 : i-s-nn , 1)'];
        s = s+1;
    end
    [0 0 i-mm i-mm-s i-mm i-mm-nn+1]
    x_train_sample(1,:) =  [ A_reconcile(i-mm : -1 :i-mm-s,1)' A_input(i-mm : -1 : i-mm-nn+1 , 1)'];
    y_train_sample(1,:) =  [ T_reconcile(10-mm : -1 :10-mm-s,1)' T_input(i-mm : -1 : i-mm-nn+1 , 1)'];

    
    A_train_out = x_train_predict';% A_train_out 是列向量 
    T_train_out = y_train_predict';% T_train_out 是行向量
    
    train = [x_train_sample y_train_sample ]';
    
    %train_predict
    %获得下一个窗口的预测值
    if i ~= tt
        [test,test_out] = test_predit(x_train_predict,y_train_predict,A_input(i-1:-1:i-nn,1),T_input(i-1:-1:i-nn,1),...
                                    A_measurement(i+1, 1),T_measurement(i+1,1));
    else
        [test,test_out] = test_predit(x_train_predict,y_train_predict,A_input(i-1:-1:i-nn,1),T_input(i-1:-1:i-nn,1),...
                                    A_measurement(i, 1),T_measurement(i,1));
    end
    
    %train the model
    [tran_out,test_predict,model] =lssvm_crossvalidate(train,[A_train_out' T_train_out'] ,test,test_out);
%     [train_predict,model] =pso_lssvm(x_train_sample,train_out,test,test_out);
    out(i-4:i+1,:) = tran_out';
    disp('train finish')    
    

   
    %% START OPTIMIZATION

    A_sys_input = A_input(i - mm-nn +1 :i-1,1);
    T_sys_input = T_input(i - mm-nn +1 :i-1,1);
    A_init_opt_value = A_reconcile(i-mm-4:i-mm);
    T_init_opt_value = T_reconcile(i-mm-4:i-mm);

    A_init_matrix = x_train_predict';
    T_init_matrix = y_train_predict';
      
    % 整理model 的变量
    A_model.alpha = model.alpha(:,1);
    A_model.b = model.b(:,1);
    A_model.sigma =  model.gam(:,1);
    A_model.input_dim = 8;
   
    T_model.alpha = model.alpha(:,2);
    T_model.b = model.b(:,2);
    T_model.sigma = model.gam(:,2);
    T_model.input_dim = 8;
    
%   'Algorithm','sqp'
%,  'GradObj','on'
%   'LargeScale','on', 
%    A = eye(6)*-1;
%    b = ones(6,1)*min(xx_measurement)
    options = optimset('display','off');
    
   % x_train_sample : 用来充当非线性约束模型的参数
   % A_model：模型需要的参数
   % A_sys_input :系统的输入
   % A_init_opt_value ： 用来与y一起构建的参数 应该是前一时刻的校正值
    [y]= fmincon(@(y)optimization_func(y,A_measurement(i-4:i,1),A_std),A_init_matrix,...
       [],[],[],[],A_measurement(i-4:i)-ones(5,1)*error,A_measurement(i-4:i),...
       @(y)nonlcons_mimo_fmincon(y,A_init_opt_value,A_sys_input,A_model,x_train_sample),options)
   
    A_reconcile(i) = y(5);
   
    [y1]= fmincon(@(y1)optimization_func(y1,T_measurement(i-4:i,1),T_std),T_init_matrix,...
         [],[],[],[],truevalue(i-4:i,2)-ones(5,1)*error,truevalue(i-4:i,2)+ones(5,1)*0.01,...
         @(y1)nonlcons_mimo_fmincon(y1,T_init_opt_value,T_sys_input,T_model,y_train_sample),options)
     
    T_reconcile(i) = y1(5);
    
       %% 更新预测值
    x_train_predict = [y(2:5)';test_predict(1,1)];
    y_train_predict = [y1(2:5)';test_predict(2,1)];
    
end

%% 画图

% figure(2)
% plot([1:tt],A_svm_train_ans(1,1:tt),'b',[1:tt],A_measurement(1:tt,1),'g')
% 
% figure(3)
% plot([1:tt],T_svm_train_ans(1,1:tt),'b',[1:tt],T_measurement(1:tt,1),'g')

figure(1)
plot([1:tt],truevalue(1:tt,1),'r-',[1:tt],A_reconcile(1:tt,1),'b',[1:tt],A_measurement(1:tt,1),'g');

figure(2)
plot([1:tt],truevalue(1:tt,2),'r-',[1:tt],T_reconcile(1:tt,1),'b',[1:tt],T_measurement(1:tt,1),'g');

figure(3)
plot([1:tt],truevalue(1:tt,1),'r-',[1:tt],out(1:tt,1),'b',[1:tt],A_measurement(1:tt,1),'g');

figure(4)
plot([1:tt],truevalue(1:tt,2),'r-',[1:tt],out(1:tt,2),'b',[1:tt],T_measurement(1:tt,1),'g');