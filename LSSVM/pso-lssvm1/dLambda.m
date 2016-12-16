function dLambda(Y,measurement,var,init_opt_value,sys_input,model,train)
    dLambda = nan(size(Y));

    h = 1e-3; % this is the step size used in the finite difference.
    for i=1:numel(Y)
       dY=zeros(size(Y));
       dY(i) = h;
       dLambda(i) = (func(Y+dY,measurement,var,init_opt_value,sys_input,model,train)-func(Y-dY,measurement,var,init_opt_value,sys_input,model,train))/(2*h);
    end
end