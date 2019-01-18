%%误码率 10-6.三个数三个数的判决。
clear all
fcarrier=10e3;%10k
Tp=1/fcarrier;%（单个载波周期，针对bpsk则为一个比特的周期）单位s,倒数极为载波频率0.05
num = 16 ;%一个周期内的采样点数为20，由于三个数联合判决因此注意后面乘以3
fs=num/Tp;%单位？Hz采样率恰好是周期倒数的整数倍
nbei=1;%s最大速率的nbei分之一
bps=1/Tp/nbei;%由于是bpsk所以最快只能一个周期传1个bit，最大1/Tp，速率为
T3=3*nbei*Tp;%三个比特,因为需要几个载波周期才传一个比特
SNR=20;


N1=10;%样本数为N1；
n= [0:1:N1*T3*fs-1]';
% m=ceil(N/Ncomp); %取大取整
jumps=2.0*round(rand(3*N1,1))-1.0;
am=kron(jumps,ones(nbei*num,1));
% am=kron(jumps,ones(2,1));%此处不能用repmat，此用于测试
s1=[sin(2*pi/Tp/fs*n).*am;zeros(1,1)];%补一个零为了用于匹配fft变换%注意数组用点成

;%repmat( A , m , n )：将向量／矩阵在垂直方向复制m次，在水平方向复制n次。
%补一个零点为了画图匹配

t= [0:1:N1*T3*fs]'/fs;

f=[-fs/2:1/T3/N1:fs/2]';
s1noise=awgn(s1,SNR);

% 
% for lie=1:8
% 
% figure(lie)
% subplot(2,1,1)
% plot(t,s1noise(:,lie));
% % plot(t,s1(:,1));
% xlabel({'时间 t/s';'so1输出波形'})
% ylabel('幅度')
% subplot(2,1,2)
% plot(f,fftshift(abs(fft(s1noise(:,lie)))))
%  xlabel({'频率 f/Hz';'b) 常规信号加噪声频域波形'})
%  ylabel('FFT变换幅度')
% end
[Hd,order] = equiripplefilter;%order是返回的阶数，由于滤波器有延迟，因此在order/2之前的时域信号值接近0，等波纹FIR滤波器
% [Hd,order] =butterworthfilter;%巴特沃斯IIR滤波器，巴特沃斯滤波器之前为零的值不是阶数的一半
finaldata0=filter(Hd,s1noise);
finaldata=finaldata0(round(order/2):(3*N1*num),:);%去除之前的零值
%num*3为一组数

figure(1)
subplot(2,1,1)
plot(t,s1noise);
xlabel({'时间 t/s';'so1输出波形'})
ylabel('幅度')
subplot(2,1,2)
plot(t,finaldata0);
% plot(t,s1(:,1));
xlabel({'时间 t/s';'so1输出波形'})
ylabel('幅度')

figure(2)%通过观察波形验证程序正确性
subplot(2,1,1)
plot(f,fftshift(abs(fft(s1noise))))
 xlabel({'频率 f/Hz';'b) 常规信号加噪声频域波形'})
 ylabel('FFT变换幅度') 
subplot(2,1,2)
plot(f,fftshift(abs(fft(finaldata0))))
 xlabel({'频率 f/Hz';'b) 常规信号加噪声频域波形'})
 ylabel('FFT变换幅度') 

 %数据太慢，直接存
% filename = 'testdata.xlsx';
% xlswrite(filename,finaldata)%保存数据
