function [t, A_measurement,T_measurement,truevalue] = measurement(A_std,T_std,sampletime)
    [t, truevalue] = true_value(sampletime);

    [t_row,t_col] = size(t);
    A_error = normrnd(0,A_std ,t_row,1);
%     size(A_error)
    T_error = normrnd(0,T_std ,t_row,1);
    
    A_measurement = truevalue(:,1)+A_error;
    
    T_measurement = truevalue(:,2)+T_error;
%     figure(1)
%     plot(t,truevalue(:,1),'-', t,(truevalue(:,1)+A_error),'o' );
%     figure(2)
%     plot(t,truevalue(:,2),'-', t,truevalue(:,2)+T_error, 'o');
    
end