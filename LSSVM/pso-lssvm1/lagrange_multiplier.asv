function gamma = lagrange_multiplier(Y,measurement,var,init_opt_value,sys_input,model,train)
    y = Y(1:6);
    lamda = Y(7:end);
    f_part1 =  optimization_func(y,measurement,var);
    ceq = nonlcons(y,init_opt_value,sys_input,model,train);
    f = 0;
    for j = 1 : max(size(lamda))
        f =f +lamda(j)*ceq(j);
    end
    gamma  =  f_part1 + f;
    
end