function [test_predict, model ]= lssvm_crossvalidate(train,train_out,test,test_o,input)
%% �ô���Ϊ����lssvm��crossvalidate��Ԥ���㷨

% train : 5 *14 һ�б�ʾһ������
% train_result 5 *2 һ�б�ʾһ�����

train_data = train';
train_result = train_out;
test_data = test';
test_out = test_o(1,:);
%% ����lssvmģ��
type='f'
% gam=75;
gam=[75 10];
% sig2=10;
sig2=[0.45 5];
kernel = 'RBF_kernel'
proprecess='function estimation'
model=initlssvm(train_data,train_result,type,gam,sig2,kernel,proprecess);
% ������֤�Ż�����
costfun = 'crossvalidatelssvm';
costfun_args = {5,'mse'};  % the value should be an interger
optfun = 'simplex';
model = tunelssvm(model,optfun,costfun,costfun_args);   % ģ�Ͳ����Ż�
model=trainlssvm(model)
%���ѵ�����Ͳ��Լ���Ԥ��ֵ
[train_predict_y,zt,model]=simlssvm(model,train_data)
[test_predict_y,zt,model]=simlssvm(model,test_data)
% %Ԥ�����ݷ���һ��
% train_predict=mapminmax('reverse',train_predict_y',pstrain1)
% test_predict=mapminmax('reverse',test_predict_y',pstrain1)
train_predict = train_predict_y;
% test_predict = test_predict_y;
%���������
% trainmse=sum((train_predict-train_out).^2)/length(train_result)
% testmse=sum((test_predict-test_out).^2)/length(test_result) 
% besttestmse=testmse;

test_predict = predict_left(model,test_data,train_predict,5,input);




% 
% figure(1)
% plot([1:6],test_out,'r-o',[1:6],test_predict,'b-*');
% % 
% figure(2)
% plot([1:5],train_predict,'r-o',[1:5],train_out,'b-');
end