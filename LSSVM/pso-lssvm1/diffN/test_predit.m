function [predict, predict_out ]= train_predit(A_predict_in,T_predict_in,A_input,T_input,A_measure,T_measure)
    % NOTE : the A_predict_in & T_predict_in is not fit to predict,
    %        you must to hande this
    %        they are all column vector
    
    % One more step you should change it to row vector
    
    x6 = [ flipud(A_predict_in)' A_input'];
    
    y6 = [ flipud(T_predict_in)' T_input'];
    
    A_test = [x6];
    T_test = [y6];
    A_test_out = A_measure(1,1);
    T_test_out = T_measure(1,1);
    
    predict = [A_test T_test]';

   predict_out = [A_test_out' T_test_out']; 
end