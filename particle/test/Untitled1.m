% clearvars -except TT CC;
clear all;
clc;
q=10;
V=1000;
Hr=-27000;
p=0.001;
Cp=1;
U=5*10^(-4);
Ar=10;
Tc=3.4;
k0=7.86*10^12;
Ea=14090;
T0=3.5;
C0=7.5;
Cr=10^(-6);
Tr=100;
C(1)=0.1531;
T(1)=4.6091;
dt=1;
for i=1:100
    k=k0*exp(-Ea/(T(i)*Tr));
    C(i+1)=C(i)+dt*(  q/V*(C0-C(i))-k*C(i)    );
    T(i+1)=T(i)+dt*(   q/V*(T0-T(i))-Hr*Cr*k*C(i) /(p*Cp*Tr)-U*Ar*(T(i)-Tc)/(p*Cp*V)   );
    
    A=[1-dt*q/V-dt*k                             -dt*C(i)*k*Ea/(Tr*T(i)^2)
        -dt*Hr*Cr*k/(p*Cp*Tr)      1-dt*q/V-dt*Hr*Cr*C(i)*k*Ea/(p*Cp*Tr^2*T(i)^2)-dt*U*Ar/(p*Cp*V)];
end
q1=0.005;
q2=0.05;
qq=[q1 0
    0 q2];
TT=normrnd(T,T*q1);
CC=normrnd(C,C*q2);


L1=60;
TT(L1)=TT(L1)*(1-10*q1);
CC(L1)=CC(L1)*(1+10*q2);

L2=40;
TT(L2)=TT(L2)*(1-10*q1);
CC(L2)=CC(L2)*(1+10*q2);
% CC= load('CC.txt')';
% TT= load('TT.txt')';

% save 'C:\Users\Administrator\Desktop\毕业论文程序\nonlinear_kalman_filter\save_TT.mat' 'TT' ;
% save 'C:\Users\Administrator\Desktop\毕业论文程序\nonlinear_kalman_filter\save_CC.mat' 'CC' ;
TT_t=load('save_TT.mat')';
CC_t=load('save_CC.mat')';
TT_t=struct2cell(TT_t);
CC_t=struct2cell(CC_t);
TT=cell2mat(TT_t);
CC=cell2mat(CC_t);

x=[ CC
    TT];
y=zeros(2,length(TT));

y(:,1)=x(:,1);
k=k0*exp(-Ea/(y(2,1)*Tr));
y(1,2)=y(1,1)+dt*(  q/V*(C0-y(1,1))-k*y(1,1)    );
y(2,2)=y(2,1)+dt*(   q/V*(T0-y(2,1))-Hr*Cr*k*y(1,1) /(p*Cp*Tr)-U*Ar*(y(2,1)-Tc)/(p*Cp*V)   );

P_k1=[(x(1,1)*q2)^2   0
    0      (x(2,1)*q1)^2];
Q=[0.0000000008 0
    0  0.00001];
% 卡尔曼滤波
A=[1-dt*q/V-dt*k                             -dt*y(1,1)*k*Ea/(Tr*y(2,1)^2)
    -dt*Hr*Cr*k/(p*Cp*Tr)      1-dt*q/V-dt*Hr*Cr*y(1,1)*k*Ea/(p*Cp*Tr^2*y(2,1)^2)-dt*U*Ar/(p*Cp*V)];
P_k=A*P_k1*A'+Q;
R=[(x(1,2)*q2)^2   0
    0      (x(2,2)*q1)^2];
H=[1 0
    0 1];
K_k=P_k*H'*inv(H*P_k*H'+R);
y(:,2)=y(:,2)+K_k*(x(:,2)-y(:,2));
I=[1 0
    0 1];

P_k=(I-K_k*H)*P_k;

for i=2:length(TT)-1
    k=k0*exp(-Ea/(y(2,i)*Tr));
    y(1,i+1)=y(1,i)+dt*(  q/V*(C0-y(1,i))-k*y(1,i)    );
    y(2,i+1)=y(2,i)+dt*(   q/V*(T0-y(2,i))-Hr*Cr*k*y(1,i) /(p*Cp*Tr)-U*Ar*(y(2,i)-Tc)/(p*Cp*V)   );
    
    A=[1-dt*q/V-dt*k                             -dt*y(1,i)*k*Ea/(Tr*y(2,i)^2)
        -dt*Hr*Cr*k/(p*Cp*Tr)      1-dt*q/V-dt*Hr*Cr*y(1,i)*k*Ea/(p*Cp*Tr^2*y(2,i)^2)-dt*U*Ar/(p*Cp*V)];
    
    P_k=A*P_k*A'+Q;
    
%     r(1)=abs(  (y(1,i+1)-x(1,i+1))./(y(1,i+1)*q2 ) );
%     r(2)=abs(  (y(2,i+1)-x(2,i+1))./(y(2,i+1)*q1 ) );
    
    R=[(x(1,2)*q2)^2   0
    0      (x(2,2)*q1)^2];
    K_k=P_k*H'*inv(H*P_k*H'+R);
    y(:,i+1)=y(:,i+1)+K_k*(x(:,i+1)-y(:,i+1));
    P_k=(I-K_k*H)*P_k;
    c=1.5;
    for j=1:4
        r(1)=abs(  (y(1,i+1)-x(1,i+1))./(y(1,i+1)*q2 ) );
        r(2)=abs(  (y(2,i+1)-x(2,i+1))./(y(2,i+1)*q1 ) );
        if r(1)<c
           R(1,1)=(x(1,i+1)*q2)^2;
       elseif r(1)>2*c
          R(1,1)=100000;
       else
          R(1,1)=((x(1,i+1)*q2)^2*c)/(2*c-r(1));
       end
    
       if r(2)<c
           R(2,2)=(x(2,i+1)*q1)^2;
       elseif r(2)>2*c
          R(2,2)=100000;
       else
          R(2,2)=((x(2,i+1)*q1)^2*c)/(2*c-r(2));
       end 
        R=[R(1,1)  0
         0    R(2,2)];  
      K_k=P_k*H'*inv(H*P_k*H'+R);
      y(:,i+1)=y(:,i+1)+K_k*(x(:,i+1)-y(:,i+1));
      P_k=(I-K_k*H)*P_k;
    end

%        Huber
%        if r(1)<c
%            R(1,1)=(x(1,i+1)*q2)^2;
%        else
%           R(1,1)=(x(1,i+1)*q2)^2*r(1)/c;
%        end
%     
%           if r(2)<c
%            R(2,2)=(x(2,i+1)*q1)^2;
%        else
%           R(2,2)=(x(2,i+1)*q1)^2*r(2)/c;
%           end
    
%        new method
%        if r(1)<c
%            R(1,1)=(x(1,i+1)*q2)^2;
%        elseif r(1)>2*c
%           R(1,1)=100000;
%        else
%           R(1,1)=((x(1,i+1)*q2)^2*c)/(2*c-r(1));
%        end
%     
%        if r(2)<c
%            R(2,2)=(x(2,i+1)*q1)^2;
%        elseif r(2)>2*c
%           R(2,2)=100000;
%        else
%           R(2,2)=((x(2,i+1)*q1)^2*c)/(2*c-r(2));
%        end    

% cauchy
%           R=[(x(1,i+1)*q2)^2*(c^2+r(1)^2)/(2*c^2)   0
%          0      (x(2,i+1)*q1)^2*(c^2+r(2)^2)/(2*c^2)];
     
%      R=[R(1,1)  0
%          0    R(2,2)];    
     
%         least squares
%     R=[(x(1,i+1)*q2)^2   0
%         0      (x(2,i+1)*q1)^2];
    
%     K_k=P_k*H'*inv(H*P_k*H'+R);
%     y(:,i+1)=y(:,i+1)+K_k*(x(:,i+1)-y(:,i+1));
%     P_k=(I-K_k*H)*P_k;
end
% save 'C:\Users\Administrator\Desktop\毕业论文程序\nonlinear_kalman_filter\save_LS_result.mat' 'y'
yls_t=load('save_LS_result.mat');
yls_t=struct2cell(yls_t);
yls=cell2mat(yls_t);

% c=load('C.txt');
plot(1:length(C),C,'k:',1:length(CC),CC,'k*',1:length(y(1,:)),y(1,:),'k.',1:length(yls(1,:)),yls(1,:),'k-')%, 1:length(c),c,'k'
% title('反应器出口对比浓度校正曲线')
xlabel('Time/s')
axis([1,101,0.04,0.18])
ylabel('  Concentration ')
legend('true value','measurements','New method','Kalman filtering')%,'reconciled value 2'
SSRE= sum(  abs(       (y(1,:)-C)./(C*q2)       ).^2   )
% grid on;
figure
% t=load('T.txt');
plot(1:length(T),T,'k:',1:length(TT),TT,'k*',1:length(y(2,:)),y(2,:),'k.',1:length(yls(2,:)),yls(2,:),'k-')%,1:length(t),t ,'k' 
%  title('反应器出口对比温度校正曲线')
axis([1,101,4.45,4.9])
xlabel('Time/s')
ylabel('Temperature')
legend('true value','measurements','New method','Kalman filtering')%,'reconciled value 2'
SSRE= sum(  abs(       (y(2,:)-T)./(T*q1)       ).^2   )
% grid on;
