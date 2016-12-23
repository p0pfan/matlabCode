function [f, obj_gradient] = optimization_func(y,measurement,var)
    
    f =0;

    for i = 1 : 6
        f = f + 0.5 * (y(i) - measurement(1,i))^2/var;
    end
    
end
 