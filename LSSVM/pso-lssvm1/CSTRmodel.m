function dydt= CSTRmodel(t,y,A_0)
    a = y(1);
    temperature = y(2);
%     A_0 = y(3);
    tt = A_0
    T_0 = 3.5;
    dydt1 = 0.01 * (A_0 - a) - 7.86 * 10^12 * exp(-140.9/temperature)*a;
    dydt2= 0.01 * (T_0 - temperature) + 2.12 * 10 ^12* exp(-140.9 / temperature) * a - 0.005 * (temperature - 3.4);
    dydt = [dydt1;dydt2];
end