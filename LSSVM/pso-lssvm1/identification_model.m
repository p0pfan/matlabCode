% the inputX: y(k-1),y(k-2),y(k-3),y(k-4),y(k-5),u(k-1),u(k-2)
% train:  y(k-1),y(k-2),y(k-3),y(k-4),y(k-5),u(k-1),u(k-2)
% outpuet : number

function output = identification_model(model,inputX, train)

%     1. get RBF kernal
    alpha = model.alpha;
    b = model.b;
    sigma = model.kernel_pars;
    [row, col] = size(train);
    kernal_matrix = zeros(row,1);
%     size(inputX)
    for i = 1: row
%         train_size = size(train(i,:))
        kernal_matrix(i,1) =exp(-( sum((inputX - train(i , :)').^2)/(2*sigma)));
       
    end
    
%      kernal_matrix_size = size( kernal_matrix(i,1) )
    output = alpha' * kernal_matrix + b;
    
end