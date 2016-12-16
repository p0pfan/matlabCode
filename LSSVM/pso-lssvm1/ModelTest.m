function output = ModelTest(model, train, input) 
    sigma = model.kernel_pars;
    b = model.b;
    alpha = model.alpha;
    
    
    for i = 1: 6
        kernal_matrix(i,1) =exp(-( sum((input - train(1:8 , i)).^2)/(2*sigma(1))));
       
    end
    output = alpha(:,1)' * kernal_matrix + b(1);
    
end