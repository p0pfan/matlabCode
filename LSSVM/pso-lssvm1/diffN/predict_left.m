function [ output ] = predict_left( model, test_data,train_predict,count,input )
%PREDICT_LEFT 预测值的计算
%   model : svm得到的模型
%   test_data : 需要预测的一行输入
    output(1) = simlssvm(model,test_data);
    for i = 2:count
        test_data1 = [output(1,i-1) input(,1)]
        test_data2 = [output(2,i)]
        output(i) = simlssvm(model,test_data)
    end
    


end

