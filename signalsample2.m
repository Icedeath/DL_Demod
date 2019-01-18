%%������ 10-6.���������������о���

%�������
clear all
fcarrier=10e3;%10k
Tp=1/fcarrier;%�������ز����ڣ����bpsk��Ϊһ�����ص����ڣ���λs,������Ϊ�ز�Ƶ��0.05
num = 16 ;%һ�������ڵĲ������������������������о����ע��������3
fs=num/Tp;%��λHz������ǡ�������ڵ�����������
nbei=1;%s������ʵ�nbei��֮һ����һ����ռ�ü������ڣ�
bps=1/Tp/nbei;%������bpsk�������ֻ��һ�����ڴ�1��bit�����1/Tp������Ϊ
T3=3*nbei*Tp;%��������,��Ϊ��Ҫ�����ز����ڲŴ�һ������
SNR=10;

N1=1e5;%������ΪN1��
% N1=2;
n= [0:1:N1*T3*fs-1]';

% m=ceil(N/Ncomp); %ȡ��ȡ��
jumps=2.0*round(rand(3*N1,1))-1.0;
am=kron(jumps,ones(nbei*num,1));
% am=kron(jumps,ones(2,1));%�˴�������repmat�������ڲ���
s1=[sin(2*pi/Tp/fs*n).*am;zeros(1,1)];%��һ����Ϊ������ƥ��fft�任
% xiangwei1=[0*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num)]';%Ϊ��ƥ��֮ǰʱ�����ж�һ����
% xiangwei2=[0*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num)]';
% xiangwei3=[0*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num)]';
% xiangwei4=[0*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num)]';
% xiangwei5=[1*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num)]';
% xiangwei6=[1*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num)]';
% xiangwei7=[0*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num),0*pi*ones(1,nbei*num)]';
% xiangwei8=[1*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num),1*pi*ones(1,nbei*num)]';
% 
% 
% xiangwei0=[xiangwei1,xiangwei2,xiangwei3,xiangwei4,xiangwei5,xiangwei6,xiangwei7,xiangwei8];
% xiangwei=[repmat(xiangwei0,N1,1)];

% s1=[sin(2*pi/Tp/fs*repmat(n,1,8)+xiangwei);zeros(1,8)];%ע�������õ��
% s1=sin(2*pi/Tp/fs*repmat(n,1,8)+xiangwei);%����ͼ�Ͳ��ٲ�0��
%repmat( A , m , n )���������������ڴ�ֱ������m�Σ���ˮƽ������n�Ρ�
%��һ�����Ϊ�˻�ͼƥ��

t= [0:1:N1*T3*fs]'/fs;

f=[-fs/2:1/T3/N1:fs/2]';
s1noise=awgn(s1,SNR);

% 
% for lie=1:8
% 
% figure(lie)
% subplot(2,2,1)
% plot(t,s1noise(:,lie));
% % plot(t,s1(:,1));
% xlabel({'ʱ�� t/s';'so1�������'})
% ylabel('����')
% subplot(2,1,2)
% plot(f,fftshift(abs(fft(s1noise(:,lie)))))
%  xlabel({'Ƶ�� f/Hz';'b) �����źż�����Ƶ����'})
%  ylabel('FFT�任����')
% end
[Hd,order] = equiripplefilter;%order�Ƿ��صĽ����������˲������ӳ٣������order/2֮ǰ��ʱ���ź�ֵ�ӽ�0���Ȳ���FIR�˲���
% [Hd,order] =butterworthfilter;%������˹IIR�˲�����������˹�˲���֮ǰΪ���ֵ���ǽ�����һ��
finaldata0=filter(Hd,s1noise);
finaldata2=finaldata0((round(order/2/num/3)+1)*num*3+1:(3*N1*num));%ȥ��֮ǰ����Ϊ��ӵ�һ����ֵ
finalimput2=jumps((round(order/2/num/3)+1)*3+1:3*N1);
%num*3Ϊһ��������48��
% for lie=1:8
% figure(2*lie-1)
% subplot(2,2,1)
% plot(t,s1noise(:,lie));
% xlabel({'ʱ�� t/s';'so1�������'})
% ylabel('����')
% subplot(2,2,2)
% plot(t,finaldata0(:,lie));
% % plot(t,s1(:,1));
% xlabel({'ʱ�� t/s';'so1�������'})
% ylabel('����')
% 
% figure(2*lie)%ͨ���۲첨����֤������ȷ��
% subplot(2,2,1)
% plot(f,fftshift(abs(fft(s1noise(:,lie)))))
%  xlabel({'Ƶ�� f/Hz';'b) �����źż�����Ƶ����'})
%  ylabel('FFT�任����') 
% subplot(2,2,2)
% plot(f,fftshift(abs(fft(finaldata0(:,lie)))))
%  xlabel({'Ƶ�� f/Hz';'b) �����źż�����Ƶ����'})
%  ylabel('FFT�任����') 
% end
 %����̫����ֱ�Ӵ�
% filename = 'testdata.xlsx';
% xlswrite(filename,finaldata)%��������
