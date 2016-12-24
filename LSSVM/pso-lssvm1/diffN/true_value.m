function [time, output] = true_value(sampletime)
  
    tspan = [0:0.5: 30];
    A_0_init = 6.5;
    A_init = 0.1531;
    T_init = 4.6091;
    [t,y]= ode45(@CSTRmodel,tspan,[A_init; T_init],odeset,A_0_init);
    
    tspan1 = [31:0.5:sampletime];
    A_0_init1 = 7.5;
    [zz,z]= ode45(@CSTRmodel,tspan1,[A_init;T_init],odeset,A_0_init1 );
    
    time = [t; zz];
    output = [y; z];

    
end