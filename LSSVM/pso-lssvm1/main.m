clc
clear all
   
A_0_init = 6.5;
A_0_init1 = 7.5;
A_std = 0.0575;
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

mm = 5;
nn = 2;
tt = sample_time;
tt =100;
for i = 10 : 5: tt
% for i = 10 :tt
    if(i ==20)
        ddd =   0;
    end
    %%%%
    x_train_sample = zeros(6,8);
    s = 0;
    %get train sample
    for j = 6:-1:1
        start = i-1-s;
        if(i-mm-s == 0)
            x_train_sample(j,:) =  [ A_reconciliation_value(i-1-s: -1 :1 ,1)'  A_recon0 A_input(i-s:-1:i-s-2,1)'];
        else
            if(j==6)
                x_train_sample(j,:) =  [ A_measurement(i-1-s: -1 :i-mm ,1)' A_input(i-s:-1:i-s-2,1)'];
            else
                 x_train_sample(j,:) =  [ A_measurement(i-1-s: -1 :i-mm ,1)'  A_reconciliation_value(i-mm-1:-1:i-mm-s)' A_input(i-s:-1:i-s-2,1)'];
            end  
        end
        s = s+1;
    end
    
    
    
    train_out = A_measurement(i-5:i,1)';
    
    %%%%
    %train_predict
    %It is useless at this time
    x6 = [ flipud(A_measurement(6:10,1))' ones(1,3)*6.5];
    x7 = [ flipud(A_measurement(7:11,1))' ones(1,3)*6.5];
    x8 = [ flipud(A_measurement(8:12,1))' ones(1,3)*6.5];
    x9 = [ flipud(A_measurement(9:13,1))' ones(1,3)*6.5];
    x10 = [ flipud(A_measurement(10:14,1))' ones(1,3)*6.5];
    x11 = [ flipud(A_measurement(11:15,1))' ones(1,3)*6.5];
    test = [x6;x7;x8;x9;x10;x11];
    test_out = A_measurement(11:16,1)';
    %%%%
    %%%%
    %train the model
    [train_predict,model] =lssvm_crossvalidate(x_train_sample,train_out,test,test_out);
%     [train_predict,model] =pso_lssvm(x_train_sample,train_out,test,test_out);
    %%%%
    %optimization
%     yy_measurement = ;A_measurement()
   
    if (i - 10) == 0
       
        init_opt_value = [A_recon0 ; A_reconciliation_value(1:i - 5 - 1,1)]
    else
        init_opt_value = A_reconciliation_value(i - 10:i - 5-1,1);
    end
    sys_input = A_input(i - mm -nn :i,1);
    
%     options.Algorithm = 'levenberg-marquardt';
% %     init_matrix =  [yy_measurement 0 0 0 0 0 0]';
    init_matrix = [init_opt_value' 0 0 0 0 0 0]';
%     Y1 = fsolve(@(Y)lagrange_multiplier(Y,yy_measurement,A_std,init_opt_value,sys_input,model,x_train_sample),init_matrix,optimset('display','off' ,'Algorithm', 'levenberg-marquardt'))
%    
%     fval1 = lagrange_multiplier(Y1,yy_measurement,A_std,init_opt_value,sys_input,model,x_train_sample)
    options.Algorithm = 'levenberg-marquardt';
    Y1 = lsqnonlin(@(Y)lagrange_multiplier(Y,train_out,A_std,init_opt_value,sys_input,model,x_train_sample),init_matrix,[],[],options)
     fval1 = lagrange_multiplier(Y1,train_out,A_std,init_opt_value,sys_input,model,x_train_sample)
     A_reconciliation_value(i-5:i) = Y1(1 : 6);

% 
% %%%%%   
% 
%     lb = ones(6,1)*0;
%     ub = ones(6,1)*0.5;
%     options = optimset('Algorithm','active-set','TolCon',1e-006);
% %     if(i - 10 ==0)
% %          init_matrix =  [A_recon0;A_reconciliation_value(i-5:i-9)];
% %     else
% %          init_matrix =  A_reconciliation_value(i-5:i-9);
% % 
% %     end
% init_matrix =  yy_measurement';
%     [y,fval] = fmincon(@(y)optimization_func(y,yy_measurement,A_std),init_matrix,[],[],[],[],[],[],@(y)contraint_equality(y,init_opt_value,sys_input,model,x_train_sample),options);
%     A_reconciliation_value(i-5:i) = y(1 : 6);
% %%%%%

    
end
figure(1)
plot([1:tt],truevalue(1:tt,1),'r-',[1:tt],A_reconciliation_value(1:tt,1),'b',[1:tt],A_measurement(1:tt,1),'g');
% % get train sample
% x1 = [0.1531 flipud(A_measurement(1:4,1))' ones(1,3)*6.5];
% x2 = [ flipud(A_measurement(1:5,1))' ones(1,3)*6.5];
% x3 = [ flipud(A_measurement(2:6,1))' ones(1,3)*6.5];
% x4 = [ flipud(A_measurement(3:7,1))' ones(1,3)*6.5];
% x5 = [ flipud(A_measurement(4:8,1))' ones(1,3)*6.5];
% x6 = [ flipud(A_measurement(5:9,1))' ones(1,3)*6.5];
% train = [x1;x2;x3;x4;x5;x6];
% train_out = A_measurement(5:10,1)';
% 
% 
% x6 = [ flipud(A_measurement(6:10,1))' ones(1,3)*6.5];
% x7 = [ flipud(A_measurement(7:11,1))' ones(1,3)*6.5];
% x8 = [ flipud(A_measurement(8:12,1))' ones(1,3)*6.5];
% x9 = [ flipud(A_measurement(9:13,1))' ones(1,3)*6.5];
% x10 = [ flipud(A_measurement(10:14,1))' ones(1,3)*6.5];
% x11 = [ flipud(A_measurement(11:15,1))' ones(1,3)*6.5];
% test = [x6;x7;x8;x9;x10;x11];
% test_out = A_measurement(11:16,1)';
% 
% model =lssvm_crossvalidate(train,train_out,test,test_out);
%  
% 
% 
% 
% %optimization
% yy_measurement = A_measurement(5:10,1)';
% init_opt_value = [0.1531;truevalue(1:4,1)];
% %i-2 ---> i + 6 -2
% sys_input = ones(10,1)*6.5;
% % 
% % x0 = ones(6,1)*0.0;
% % lb = ones(6,1)*0.1;
% % ub = ones(6,1)*0.2;
% % options = optimset('Algorithm', 'interior-point','largescale','off');
% % [y,fval] = fmincon(@(y)optimization_func(y,yy_measurement,A_std), x0, [], [], [], [], lb, ub, @(y)contraint_equality(y,init_opt_value,sys_input,model,train),options)
% 
% % ,'TolFun',1e-12,'TolCon',1e-12,'TolX',1e-12
% init_matrix = [0.1 0.1 0.1 0.1 0.1 0.1 0 0 0 0 0 0]';
% Y1 = fsolve(@(Y)lagrange_multiplier(Y,yy_measurement,A_std,init_opt_value,sys_input,model,train),init_matrix,optimset('display','off'))
% 
% fval1 = lagrange_multiplier(Y1,yy_measurement,A_std,init_opt_value,sys_input,model,train)
% 

