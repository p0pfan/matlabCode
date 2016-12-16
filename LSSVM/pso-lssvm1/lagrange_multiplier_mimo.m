function gamma = lagrange_multiplier_mimo(Y,measurement,var,init_opt_value,sys_input,model,train)
    alpha = model.alpha;
    b = model.b;
    sigma = model.sigma;
    input_dim = model.input_dim;

    
    y = Y(1:6);
    lamda = Y(7:end);
    f_part1 =  optimization_func(y,measurement,var);
    ceq = nonlcons_mimo(y,init_opt_value,sys_input,alpha,b,sigma,input_dim,train);
    f = 0;
    for j = 1 : max(size(lamda))
        f =f +lamda(j)*ceq(j);
    end
    gamma  =  f_part1 + f;
    
end