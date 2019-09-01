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

file=csvread('GFTT-ORB.csv');
r=file(:,1);
p=file(:,2);
p1=plot(p,r,'LineWidth',lw);

file=csvread('GFTT-BRIEF.csv');
r=file(:,1);
p=file(:,2);
p2=plot(p,r,'LineWidth',lw);

file=csvread('FAST-ORB.csv');
r=file(:,1);
p=file(:,2);
p3=plot(p,r,'LineWidth',lw);

file=csvread('FAST-BRIEF.csv');
r=file(:,1);
p=file(:,2);
p4=plot(p,r,'LineWidth',lw);

file=csvread('ORB-ORB.csv');
r=file(:,1);
p=file(:,2);
p5=plot(p,r,'LineWidth',lw);

file=csvread('ORB-BRIEF.csv');
r=file(:,1);
p=file(:,2);
p6=plot(p,r,'LineWidth',lw);

title('Punto de vista');
xlabel('1-Precision');
ylabel('Recall');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6],'GFTT-ORB','GFTT-BRIEF','FAST-ORB','FAST-BRIEF',
'ORB-ORB','ORB-BRIEF','FontSize', fsz, 'Location','SouthEast');
hold off

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);
 
% Save the file as PNG
 print('ViewpointRP','-dpng','-r300');