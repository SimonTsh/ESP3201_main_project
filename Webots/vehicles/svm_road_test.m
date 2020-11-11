%% generate data ('scanline' type)
% N = 50;
% 
% x = sort(rand(N,1) * (2 + 1) - 1);
% y = sort(rand(N,1) * (6));
% 
% [xx, yy] = meshgrid(x,y);
% t = (xx - 2).^2 + (yy - 3).^2;
% 
% valid = ((xx < 1) & (xx > -1) & (yy > 0) & (yy < 3)) | ...  % straight
% ((t < 9) & (t > 1) & (xx < 2) & (yy > 3));                  % bend
% 
% surf(xx, yy, valid * 1.0)
% axis equal
% 
% road_points = [xx(valid) yy(valid)];     % scatter(points(:,1),points(:,2))
% num_road_points = sum(sum(valid));
% ext_points = [xx(~valid) yy(~valid)];
% 
% data = [road_points;ext_points];
% class = [ones(num_road_points,1); zeros(N*N - num_road_points,1)];

%% generate data (scatter type)
% disp('generating points ... ')
% 
% N = 1000;
% x = (rand(N,1) * (2 + 1) - 1);
% y = (rand(N,1) * (6));
% t = (x - 2).^2 + (y - 3).^2;
% 
% data = [x,y];
% class = ((x < 1) & (x > -1) & (y > 0) & (y < 3)) | ...  % straight
%         ((t < 9) & (t > 1) & (x < 2) & (y > 3));        % bend
% 
% disp('points generated')

%% import data from python
T = readtable('castTo3D_pts.csv');

stride = 1;
data = table2array([T(1:stride:end,'x'), T(1:stride:end,'z')]);
class = table2array(T(1:stride:end,'IsRoad'));

rng(0)
p_drop = rand(length(data), 1) ./ data(:,2) * 2000;
% figure(10); scatter(data(:,2),p_drop)
data = data(p_drop < 1,:); class = class(p_drop < 1);
disp(size(data))

figure(11); gscatter(data(:,1),data(:,2),class,'rb','.')
% figure(12); histogram(data(:,2))

%% SVM

disp('starting svm...')
tic
svm = fitcsvm(data,class,'KernelFunction','rbf', ...
    'BoxConstraint',Inf,'ClassNames',[0,1]);

% svm = fitcsvm(data,class,'KernelFunction','mysigmoid','Standardize',true,'ClassNames',[0,1]);
toc
disp('svm done')

%% plot

d = 0.1;
[x1Grid,x2Grid] = meshgrid(min(data(:,1)):d:max(data(:,1)),...
    min(data(:,2)):d:max(data(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(svm,xGrid);

% Plot the data and the decision boundary
% h(1) = plot(road_points(:,1),road_points(:,2),'b.','MarkerSize',15);
% hold on
% h(2) = plot(ext_points(:,1),ext_points(:,2),'r.','MarkerSize',15);
figure(2);
h(1:2) = gscatter(data(:,1),data(:,2),class,'rb','.');
axis equal
hold on
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'Road','Ground','Support Vectors'});
axis equal
hold off

% figure;
% plot(road_points(:,1),road_points(:,2),'r.','MarkerSize',15)
% hold on
% plot(num_road_points(:,1),num_road_points(:,2),'b.','MarkerSize',15)
% axis equal
% hold off