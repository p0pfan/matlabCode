function [train_predict, model ]= pso_lssvm(train,train_out,test,test_out)
% %% 清空环境
% clc
% clear
% 
% %% 导入训练数据和测试数据
% %%训练数据输入、输出
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
% %%测试数据输入、输出
% test = [4500 0.179 0.2 0.2 169.56;
%     7500 0.184 0.2 0.2 282.6;
%     10500 0.189 0.2 0.2 395.64;
%     13500 0.195 0.2 0.2 508.68;
%     16500 0.2 0.2 0.2 621.72;
%     19500 0.205 0.2 0.2 734.76;
%     22500 0.212 0.2 0.2 847.8];
% test_out = [3.8237 2.9526 3.3824 2.8132 2.1938 2.816 3.0887];

%%数据归一化
%%归一化方法1（利用libsvm工具箱函数归一化）
%[train_data,test_data]=scaleForSVM(train,test,0,1)
%[train_result,test_result,pstrain1]=scaleForSVM(train_out',test_out',0,1)

%%归一化方法2（利用mapminmax函数归一化）
[train_data,pstrain0] = mapminmax(train',0,1);
[test_data] = mapminmax('apply',test',pstrain0);
[train_result,pstrain1] = mapminmax(train_out,0,1);
[test_result] = mapminmax('apply',test_out,pstrain1);

train_data = train_data';
train_result=train_result';
test_data = test_data';
%% 粒子群参数初始化

%粒子群算法中的两个参数
c1 = 1.5; % c1 belongs to [0,2] c1:初始为1.5,pso参数局部搜索能力
c2 = 1.7; % c2 belongs to [0,2] c2:初始为1.7,pso参数全局搜索能力

maxgen=100; % 进化次数 
sizepop=30; % 种群规模

popcmax=10^(3);% popcmax:初始为1000,LSSVM 参数c的变化的最大值.
popcmin=10^(-1);% popcmin:初始为0.1,LSSVM 参数c的变化的最小值.
popgmax=10^(2);% popgmax:初始为1000,LSSVM 参数g的变化的最大值
popgmin=10^(-2);% popgmin:初始为0.01,LSSVM 参数c的变化的最小值.
k = 0.5; % k belongs to [0.1,1.0];
Vcmax = k*popcmax;%参数 c 迭代速度最大值
Vcmin = -Vcmax ;
Vgmax = k*popgmax;%参数 g 迭代速度最大值
Vgmin = -Vgmax ; 

eps = 10^(-8);

%%定义lssvm相关参数
type='f';
kernel = 'RBF_kernel';
proprecess='proprecess';
%% 产生初始粒子和速度
for i=1:sizepop
% 随机产生种群
pop(i,1) = (popcmax-popcmin)*rand(1,1)+popcmin ; % 初始种群
pop(i,2) = (popgmax-popgmin)*rand(1,1)+popgmin;
V(i,1)=Vcmax*rands(1,1); % 初始化速度
V(i,2)=Vgmax*rands(1,1);

% 计算初始适应度
gam=pop(i,1)
sig2=pop(i,2)
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess);
model=trainlssvm(model);
%求出训练集和测试集的预测值
[train_predict_y,zt,model]=simlssvm(model,train_data);
[test_predict_y,zt,model]=simlssvm(model,test_data);
%预测数据反归一化
train_predict=mapminmax('reverse',train_predict_y',pstrain1)%训练集预测值
test_predict=mapminmax('reverse',test_predict_y',pstrain1) %测试集预测值
%计算均方差
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
%fitness(i)=trainmse;%以训练集/测试集的预测值计算的均方差为适应度值
fitness(i)=testmse; 
end

% 找极值和极值点
[global_fitness bestindex]=min(fitness); % 全局极值
local_fitness=fitness; % 个体极值初始化 

global_x=pop(bestindex,:); % 全局极值点
local_x=pop; % 个体极值点初始化

% 每一代种群的平均适应度
avgfitness_gen = zeros(1,maxgen);

tic

%% 迭代寻优
for i=1:maxgen

for j=1:sizepop

%速度更新
wV = 1; % wV best belongs to [0.8,1.2]为速率更新公式中速度前面的弹性系数
V(j,:) = wV*V(j,:) + c1*rand*(local_x(j,:) - pop(j,:)) + c2*rand*(global_x - pop(j,:));
if V(j,1) > Vcmax %以下几个不等式是为了限定速度在最大最小之间
V(j,1) = Vcmax;
end
if V(j,1) < Vcmin
V(j,1) = Vcmin;
end
if V(j,2) > Vgmax
V(j,2) = Vgmax;
end
if V(j,2) < Vgmin
V(j,2) = Vgmin; %以上几个不等式是为了限定速度在最大最小之间
end

%种群更新
wP = 1; % wP:初始为1,种群更新公式中速度前面的弹性系数
pop(j,:)=pop(j,:)+wP*V(j,:);
if pop(j,1) > popcmax %以下几个不等式是为了限定 c 在最大最小之间
pop(j,1) = popcmax;
end
if pop(j,1) < popcmin
pop(j,1) = popcmin;
end
if pop(j,2) > popgmax %以下几个不等式是为了限定 g 在最大最小之间
pop(j,2) = popgmax;
end
if pop(j,2) < popgmin
pop(j,2) = popgmin;
end

% 自适应粒子变异
if rand>0.5
k=ceil(2*rand);%ceil 是向离它最近的大整数圆整

if k == 1
pop(j,k) = (20-1)*rand+1;
end
if k == 2
pop(j,k) = (popgmax-popgmin)*rand+popgmin;
end 


%%新粒子适应度值
gam=pop(j,1)
sig2=pop(j,2)
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess)
model=trainlssvm(model)
%求出训练集和测试集的预测值
[train_predict_y,zt,model]=simlssvm(model,train_data)
[test_predict_y,zt,model]=simlssvm(model,test_data)
%预测数据反归一化
train_predict=mapminmax('reverse',train_predict_y',pstrain1)%训练集预测值
test_predict=mapminmax('reverse',test_predict_y',pstrain1) %测试集预测值
%计算均方差
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
%fitness(i)=trainmse;%以训练集/测试集的预测值计算的均方差为适应度值
fitness(i)=testmse; 
end

%个体最优更新
if fitness(j) < local_fitness(j)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end

if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end 

%群体最优更新
if fitness(j) < global_fitness
global_x = pop(j,:);
global_fitness = fitness(j);
end

if abs( fitness(j)-global_fitness )<=eps && pop(j,1) < global_x(1)
global_x = pop(j,:);
global_fitness = fitness(j);
end
end
fit_gen(i)=global_fitness; 
avgfitness_gen(i) = sum(fitness)/sizepop;

end

toc

%% 结果分析
plot(fit_gen,'LineWidth',5);
title(['适应度曲线','(参数c1=',num2str(c1),',c2=',num2str(c2),',终止代数=',num2str(maxgen),')'],'FontSize',13);
xlabel('进化代数');ylabel('适应度');

bestc = global_x(1)
bestg = global_x(2)

gam=bestc
sig2=bestg
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess)
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

figure(1)
plot(test_out,'r-o');
hold on;
plot(test_predict,'b-*');
end





