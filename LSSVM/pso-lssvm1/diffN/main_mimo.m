clc
clear all
   
out__ = '************************************************************'
%A : 表示浓度
%T : 温度
% 浓度的初始值
A_0_init = 6.5;
A_0_init1 = 7.5;
A_std = 0.0275;
T_std = 0.0592;
sample_time = 200;

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

A_reconciliation_value = [];
A_recon0 = 0.1531;
A_reconciliation_value(1 : 5,1) = truevalue(1:5,1);

T_reconciliation_value = [];
T_recon0 = 4.6091;
T_reconciliation_value(1 : 5,1) = truevalue(1:5,2);

mm = 5;
nn = 2;
tt = sample_time;
tt =100;
% for i = 10 :5: tt
for i = 10 :tt
    if(i ==20)
        ddd =   0;
    end
    %%%%
    x_train_sample = zeros(6,8);
    y_train_sample = zeros(6,8);
    s = 0;
    %get train sample
    for j = 6:-1:1
        start = i-1-s;
        if(i-mm-s == 0)
            x_train_sample(j,:) =  [ A_reconciliation_value(i-1-s: -1 :1 ,1)'  A_recon0 A_input(i-s:-1:i-s-2,1)'];
            y_train_sample(j,:) =  [ T_reconciliation_value(i-1-s: -1 :1 ,1)'  T_recon0 T_input(i-s:-1:i-s-2,1)'];

        else
            if(j==6)
                x_train_sample(j,:) =  [ A_measurement(i-1-s: -1 :i-mm ,1)' A_input(i-s:-1:i-s-2,1)'];
                y_train_sample(j,:) =  [ T_measurement(i-1-s: -1 :i-mm ,1)' T_input(i-s:-1:i-s-2,1)'];
            else
                 x_train_sample(j,:) =  [ A_measurement(i-1-s: -1 :i-mm ,1)'  A_reconciliation_value(i-mm-1:-1:i-mm-s)' A_input(i-s:-1:i-s-2,1)'];
                 y_train_sample(j,:) =  [ T_measurement(i-1-s: -1 :i-mm ,1)'  T_reconciliation_value(i-mm-1:-1:i-mm-s)' T_input(i-s:-1:i-s-2,1)'];
            end  
        end
        s = s+1;
    end
    % x_train_sample 6 samples 6 *8
     % y_train_sample 6 samples 6 *8
    
    
   A_train_out = A_measurement(i-5:i,1)';
    T_train_out = T_measurement(i-5:i,1)';
    train = [x_train_sample y_train_sample ]';
    

    %%%%
    %train_predict
    %It is useless at this time
    x6 = [ flipud(A_measurement(6:10,1))' ones(1,3)*6.5];
    x7 = [ flipud(A_measurement(7:11,1))' ones(1,3)*6.5];
    x8 = [ flipud(A_measurement(8:12,1))' ones(1,3)*6.5];
    x9 = [ flipud(A_measurement(9:13,1))' ones(1,3)*6.5];
    x10 = [ flipud(A_measurement(10:14,1))' ones(1,3)*6.5];
    x11 = [ flipud(A_measurement(11:15,1))' ones(1,3)*6.5];
    
    y6 = [ flipud(T_measurement(6:10,1))' ones(1,3)*6.5];
    y7 = [ flipud(T_measurement(7:11,1))' ones(1,3)*6.5];
    y8 = [ flipud(T_measurement(8:12,1))' ones(1,3)*6.5];
    y9 = [ flipud(T_measurement(9:13,1))' ones(1,3)*6.5];
    y10 = [ flipud(T_measurement(10:14,1))' ones(1,3)*6.5];
    y11 = [ flipud(T_measurement(11:15,1))' ones(1,3)*6.5];
    A_test = [x6;x7;x8;x9;x10;x11];
    T_test = [y6;y7;y8;y9;y10;y11];
    A_test_out = A_measurement(11:16,1)';
    T_test_out = T_measurement(11:16,1)';
    
    test = [A_test T_test]';

   test_out = [A_test_out' T_test_out']; 
    %%%%
    %%%%
    %train the model
    [train_predict,model] =lssvm_crossvalidate(train,[A_train_out' T_train_out'] ,test,test_out);
%     [train_predict,model] =pso_lssvm(x_train_sample,train_out,test,test_out);


    
%%%%
%%%%%%optimization'
    xx_measurement = train_predict(1,:);
    yy_measurement= train_predict(2,:);
    if (i - 10) == 0
       
        A_init_opt_value = [A_recon0 ; A_reconciliation_value(1:i - 5 - 1,1)];
         T_init_opt_value = [T_recon0 ; T_reconciliation_value(1:i - 5 - 1,1)]
    else
        A_init_opt_value = A_reconciliation_value(i - 10:i - 5-1,1);
        T_init_opt_value = T_reconciliation_value(i - 10:i - 5-1,1);
    end
    A_sys_input = A_input(i - mm -nn :i,1);
     T_sys_input = T_input(i - mm -nn :i,1);
    

    A_init_matrix = [A_init_opt_value;A_init_opt_value(end,1)]'
     
    
    T_init_matrix = [T_init_opt_value;T_init_opt_value(end,1)]'
    
    
    A_model.alpha = model.alpha(:,1);
    A_model.b = model.b(:,1);
    A_model.sigma =  model.gam(:,1);
    A_model.input_dim = 8;
    
  
    
    T_model.alpha = model.alpha(:,2);
    T_model.b = model.b(:,2);
    T_model.sigma = model.gam(:,2);
    T_model.input_dim = 8;
% 'Algorithm','sqp'
%,'GradObj','on'
% 'LargeScale','on', 
%    A = eye(6)*-1;
%     b = ones(6,1)*min(xx_measurement)
    options = optimset('display','off','Algorithm','interior-point');
   [y]= fmincon(@(y)optimization_func(y,A_measurement(i-5:i,1)',A_std),A_init_matrix,[],[],[],[],ones(6,1)*max(xx_measurement)-0.1,ones(6,1)*max(xx_measurement),@(y)nonlcons_mimo_fmincon(y,A_init_opt_value,A_sys_input,A_model,x_train_sample),options)
     A_reconciliation_value(i-5:i) = y(1 : 6);
     
     [y1]= fmincon(@(y1)optimization_func(y1,T_measurement(i-5:i,1)',T_std),T_init_matrix,[],[],[],[],ones(6,1)*min(yy_measurement),ones(6,1)*max(yy_measurement),@(y1)nonlcons_mimo_fmincon(y1,T_init_opt_value,T_sys_input,T_model,y_train_sample),options)
     T_reconciliation_value(i-5:i) = y1(1 : 6);
%     [Y1,fval,exitflag] =  lsqnonlin(@(Y1)lagrange_multiplier_mimo(Y1,xx_measurement,A_std,A_init_opt_value, ...
%         A_sys_input,A_model,x_train_sample),A_init_matrix,[],[],optimset('display','off','LargeScale','on','Algorithm', 'levenberg-marquardt'))
% 
%      fval1 = lagrange_multiplier_mimo(Y1,xx_measurement,A_std,A_init_opt_value, ... 
%          A_sys_input,A_model,x_train_sample)
%      A_reconciliation_value(i-5:i) = Y1(1 : 6);
     
%      alpha,b,sigma,input_dim
%      Y2 = fsolve(@(Y2)lagrange_multiplier_mimo(Y2,yy_measurement,T_std,T_init_opt_value,...
%          T_sys_input,T_model,y_train_sample),T_init_matrix,optimset('display','off','Algorithm ','levenberg-marquardt'))
%      fval2 = lagrange_multiplier_mimo(Y2,yy_measurement,T_std,T_init_opt_value,T_sys_input, ... 
%          T_model,y_train_sample)
%      T_reconciliation_value(i-5:i) = Y2(1 : 6);



    
end
figure(1)
plot([1:tt],truevalue(1:tt,1),'r-',[1:tt],A_reconciliation_value(1:tt,1),'b',[1:tt],A_measurement(1:tt,1),'g');

figure(2)
plot([1:tt],truevalue(1:tt,2),'r-',[1:tt],T_reconciliation_value(1:tt,1),'b',[1:tt],T_measurement(1:tt,1),'g');