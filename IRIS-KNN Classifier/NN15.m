pkg load statistics

% setup data
D = csvread('iris.csv');
X_train = D(:, 1:2);
y_train = D(:, end); 

% setup meshgrid
[x1, x2] = meshgrid(2:0.01:5, 0:0.01:3);
grid_size = size(x1);
X12 = [x1(:) x2(:)]; % 301*301

% compute 2NN decision 
n_X12 = size(X12, 1); % 90601
decision = zeros(n_X12, 1);
for i=1:n_X12    
    point = X12(i, :);
    
    % compute euclidan distance from the point to all training data
    dist = pdist2(X_train, point);
    
    % sort the distance, get the index
    [~, idx_sorted] = sort(dist);
    
    % find the class of the nearest neighbour
    
    pred1 = y_train(idx_sorted(1));
    pred2 = y_train(idx_sorted(2));
    pred3 = y_train(idx_sorted(3));
    pred4 = y_train(idx_sorted(4));
    pred5 = y_train(idx_sorted(5));
    pred6 = y_train(idx_sorted(6));
    pred7 = y_train(idx_sorted(7));
    pred8 = y_train(idx_sorted(8));
    pred9 = y_train(idx_sorted(9));
    pred10 = y_train(idx_sorted(10));
    pred11 = y_train(idx_sorted(11));
    pred12 = y_train(idx_sorted(12));
    pred13 = y_train(idx_sorted(13));
    pred14 = y_train(idx_sorted(14));
    pred15 = y_train(idx_sorted(15));
    
    A=[pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15];
    decision(i) = mode(A); %frequency
end

% plot decisions in the grid
figure;
decisionmap = reshape(decision, grid_size);
imagesc(2:0.01:5, 0:0.01:3, decisionmap);
set(gca,'ydir','normal');

% colormap for the classes
% class 1 = light red, 2 = light green, 3 = light blue
cmap = [1 0.8 0.8; 0.8 1 0.8; 0.8 0.8 1];
colormap(cmap);

% satter plot data
hold on;
scatter(X_train(y_train == 1, 1), X_train(y_train == 1, 2), 10, 'r');
scatter(X_train(y_train == 2, 1), X_train(y_train == 2, 2), 10, 'g');
scatter(X_train(y_train == 3, 1), X_train(y_train == 3, 2), 10, 'b');
hold off;