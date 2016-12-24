function [f] = optimization_func(y,measurement,var)
    
    f =0;

    for i = 1 : size(measurement,1)
        f = f + 0.5 * (y(i) - measurement(i,1))^2/var;
    end
    
end
 