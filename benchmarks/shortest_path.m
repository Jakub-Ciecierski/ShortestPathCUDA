x_device = [16,100,10000,40000,250000,422500]
x_host = [16,100,10000,40000,250000,422500]

% msecs
%y_device = [2.4531,2.7624,21.5908,51.0944,478.9624, 841.6524]
%y_host = [0.0028,0.0317,165.0970,2337.1484,86299.0390, 246215.7968]

% secs
y_device = [0.0004, 0.0027, 0.02159, 0.05109, 0.4789, 0.8417]
y_host = [0.0024,0.0027,0.1650,2.3371484,86.299039, 246.2157968]

figure
plot(x_device,y_device,'--go',x_host,y_host,':r*');

axis([0 inf 0 max(y_host)])


title('Graph of Shortest Path. CUDA GPU, includes all memory copies (green) and Sequential CPU (red) ')
xlabel('Number of vertices') % x-axis label
ylabel('Time in seconds') % y-axis label



%%%%%%%%%%%%%%%%%%%%% DEVICE ONLY %%%%%%%%%%%%%%%%%%%%%

x_device = [16,100,10000,40000,250000,422500, 1000000, 2550000, 4000000]
y_device = [2.4531,2.7624,21.5908,51.0944,478.9624, 841.6524, 2873.8695, 9435.4267, 21847.2539]

figure
plot(x_device,y_device,'--go');

%set(gca,'XTick',[10000])
%set(gca,'YTick',[1000])
%set(gca,'YTick',0:1000:21847.2539)
%set(gca,'yTickLabel',{'0', ' ', 'pi', ' ', '2pi', ' ', '3pi', ' ', '4pi'})

%axis([0 inf -10000 246215.7968])

title('Graph of Shortest Path. CUDA GPU (includes all memory copies)')
xlabel('Number of vertices') % x-axis label
ylabel('Time in milliseconds') % y-axis label
