%% GFTT detector
clear all;
close all;
clc;

width = 4;     % Width in inches
height = 4.5;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 15;      % Fontsize
lw = 1.5;      % LineWidth
msz = 6;       % MarkerSize

figure(1);
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); %<- Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw); %<- Set properties

hold on;
grid off;

% GFTT
x=csvread('GFTT-ORB-r.csv');
x = x*100;x = [100 x];
p1=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','h', 'LineStyle','-.');

x=csvread('GFTT-BRIEF-r.csv');
x = x*100;x = [100 x];
p2=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','+', 'LineStyle','-.');

% FAST
x=csvread('FAST-ORB-r.csv');
x = x*100;x = [100 x];
p3=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','o', 'LineStyle','-.');

x=csvread('FAST-BRISK-r.csv');
x = x*100;x = [100 x];
p4=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','s', 'LineStyle','-.');

x=csvread('FAST-BRIEF-r.csv');
x = x*100;x = [100 x];
p5=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','^', 'LineStyle','-.');

x=csvread('FAST-FREAK-r.csv');
x = x*100;x = [100 x];
p6=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','>', 'LineStyle','-.');

% ORB
x=csvread('ORB-ORB-r.csv');
x = x*100;x = [100 x];
p7=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','p', 'LineStyle','-.');

x=csvread('ORB-BRIEF-r.csv');
x = x*100;x = [100 x];
p8=plot(x,'LineWidth',lw, 'MarkerSize',msz, 'Marker','x', 'LineStyle','-.');

title('Rotacion + escala');
xlabel('Imagen');
ylabel('Repetibilidad (%)');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6 p7 p8],'GFTT-ORB','GFTT-BRIEF','FAST-ORB','FAST-BRISK','FAST-BRIEF','FAST-FREAK', 
'ORB-ORB','ORB-BRIEF','FontSize', fsz, 'Location','Default');
hold off;

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);
 
% Save the file as PNG
 print('Rotation','-dpng','-r300');
