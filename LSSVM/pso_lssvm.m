function [train_predict, model ]= pso_lssvm(train,train_out,test,test_out)
% %% ��ջ���
% clc
% clear
% 
% %% ����ѵ�����ݺͲ�������
% %%ѵ���������롢���
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
% %%�����������롢���
% test = [4500 0.179 0.2 0.2 169.56;
%     7500 0.184 0.2 0.2 282.6;
%     10500 0.189 0.2 0.2 395.64;
%     13500 0.195 0.2 0.2 508.68;
%     16500 0.2 0.2 0.2 621.72;
%     19500 0.205 0.2 0.2 734.76;
%     22500 0.212 0.2 0.2 847.8];
% test_out = [3.8237 2.9526 3.3824 2.8132 2.1938 2.816 3.0887];

%%���ݹ�һ��
%%��һ������1������libsvm�����亯����һ����
%[train_data,test_data]=scaleForSVM(train,test,0,1)
%[train_result,test_result,pstrain1]=scaleForSVM(train_out',test_out',0,1)

%%��һ������2������mapminmax������һ����
[train_data,pstrain0] = mapminmax(train',0,1);
[test_data] = mapminmax('apply',test',pstrain0);
[train_result,pstrain1] = mapminmax(train_out,0,1);
[test_result] = mapminmax('apply',test_out,pstrain1);

train_data = train_data';
train_result=train_result';
test_data = test_data';
%% ����Ⱥ������ʼ��

%����Ⱥ�㷨�е���������
c1 = 1.5; % c1 belongs to [0,2] c1:��ʼΪ1.5,pso�����ֲ���������
c2 = 1.7; % c2 belongs to [0,2] c2:��ʼΪ1.7,pso����ȫ����������

maxgen=100; % �������� 
sizepop=30; % ��Ⱥ��ģ

popcmax=10^(3);% popcmax:��ʼΪ1000,LSSVM ����c�ı仯�����ֵ.
popcmin=10^(-1);% popcmin:��ʼΪ0.1,LSSVM ����c�ı仯����Сֵ.
popgmax=10^(2);% popgmax:��ʼΪ1000,LSSVM ����g�ı仯�����ֵ
popgmin=10^(-2);% popgmin:��ʼΪ0.01,LSSVM ����c�ı仯����Сֵ.
k = 0.5; % k belongs to [0.1,1.0];
Vcmax = k*popcmax;%���� c �����ٶ����ֵ
Vcmin = -Vcmax ;
Vgmax = k*popgmax;%���� g �����ٶ����ֵ
Vgmin = -Vgmax ; 

eps = 10^(-8);

%%����lssvm��ز���
type='f';
kernel = 'RBF_kernel';
proprecess='proprecess';
%% ������ʼ���Ӻ��ٶ�
for i=1:sizepop
% ���������Ⱥ
pop(i,1) = (popcmax-popcmin)*rand(1,1)+popcmin ; % ��ʼ��Ⱥ
pop(i,2) = (popgmax-popgmin)*rand(1,1)+popgmin;
V(i,1)=Vcmax*rands(1,1); % ��ʼ���ٶ�
V(i,2)=Vgmax*rands(1,1);

% �����ʼ��Ӧ��
gam=pop(i,1)
sig2=pop(i,2)
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess);
model=trainlssvm(model);
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_data);
[test_predict_y,zt,model]=simlssvm(model,test_data);
%Ԥ�����ݷ���һ��
train_predict=mapminmax('reverse',train_predict_y',pstrain1)%ѵ����Ԥ��ֵ
test_predict=mapminmax('reverse',test_predict_y',pstrain1) %���Լ�Ԥ��ֵ
%���������
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
%fitness(i)=trainmse;%��ѵ����/���Լ���Ԥ��ֵ����ľ�����Ϊ��Ӧ��ֵ
fitness(i)=testmse; 
end

% �Ҽ�ֵ�ͼ�ֵ��
[global_fitness bestindex]=min(fitness); % ȫ�ּ�ֵ
local_fitness=fitness; % ���弫ֵ��ʼ�� 

global_x=pop(bestindex,:); % ȫ�ּ�ֵ��
local_x=pop; % ���弫ֵ���ʼ��

% ÿһ����Ⱥ��ƽ����Ӧ��
avgfitness_gen = zeros(1,maxgen);

tic

%% ����Ѱ��
for i=1:maxgen

for j=1:sizepop

%�ٶȸ���
wV = 1; % wV best belongs to [0.8,1.2]Ϊ���ʸ��¹�ʽ���ٶ�ǰ��ĵ���ϵ��
V(j,:) = wV*V(j,:) + c1*rand*(local_x(j,:) - pop(j,:)) + c2*rand*(global_x - pop(j,:));
if V(j,1) > Vcmax %���¼�������ʽ��Ϊ���޶��ٶ��������С֮��
V(j,1) = Vcmax;
end
if V(j,1) < Vcmin
V(j,1) = Vcmin;
end
if V(j,2) > Vgmax
V(j,2) = Vgmax;
end
if V(j,2) < Vgmin
V(j,2) = Vgmin; %���ϼ�������ʽ��Ϊ���޶��ٶ��������С֮��
end

%��Ⱥ����
wP = 1; % wP:��ʼΪ1,��Ⱥ���¹�ʽ���ٶ�ǰ��ĵ���ϵ��
pop(j,:)=pop(j,:)+wP*V(j,:);
if pop(j,1) > popcmax %���¼�������ʽ��Ϊ���޶� c �������С֮��
pop(j,1) = popcmax;
end
if pop(j,1) < popcmin
pop(j,1) = popcmin;
end
if pop(j,2) > popgmax %���¼�������ʽ��Ϊ���޶� g �������С֮��
pop(j,2) = popgmax;
end
if pop(j,2) < popgmin
pop(j,2) = popgmin;
end

% ����Ӧ���ӱ���
if rand>0.5
k=ceil(2*rand);%ceil ������������Ĵ�����Բ��

if k == 1
pop(j,k) = (20-1)*rand+1;
end
if k == 2
pop(j,k) = (popgmax-popgmin)*rand+popgmin;
end 


%%��������Ӧ��ֵ
gam=pop(j,1)
sig2=pop(j,2)
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess)
model=trainlssvm(model)
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_data)
[test_predict_y,zt,model]=simlssvm(model,test_data)
%Ԥ�����ݷ���һ��
train_predict=mapminmax('reverse',train_predict_y',pstrain1)%ѵ����Ԥ��ֵ
test_predict=mapminmax('reverse',test_predict_y',pstrain1) %���Լ�Ԥ��ֵ
%���������
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
%fitness(i)=trainmse;%��ѵ����/���Լ���Ԥ��ֵ����ľ�����Ϊ��Ӧ��ֵ
fitness(i)=testmse; 
end

%�������Ÿ���
if fitness(j) < local_fitness(j)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end

if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
local_x(j,:) = pop(j,:);
local_fitness(j) = fitness(j);
end 

%Ⱥ�����Ÿ���
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

%% �������
plot(fit_gen,'LineWidth',5);
title(['��Ӧ������','(����c1=',num2str(c1),',c2=',num2str(c2),',��ֹ����=',num2str(maxgen),')'],'FontSize',13);
xlabel('��������');ylabel('��Ӧ��');

bestc = global_x(1)
bestg = global_x(2)

gam=bestc
sig2=bestg
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess)
model=trainlssvm(model)
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_data)
[test_predict_y,zt,model]=simlssvm(model,test_data)
%Ԥ�����ݷ���һ��
train_predict=mapminmax('reverse',train_predict_y',pstrain1)
test_predict=mapminmax('reverse',test_predict_y',pstrain1)
%���������
trainmse=sum((train_predict-train_out).^2)/length(train_result)
testmse=sum((test_predict-test_out).^2)/length(test_result) 
besttestmse=testmse;

figure(1)
plot(test_out,'r-o');
hold on;
plot(test_predict,'b-*');
end





