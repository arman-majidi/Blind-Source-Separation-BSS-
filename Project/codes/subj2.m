clc; clear; close all;

%% Initializtion
load subj_2.mat
runNum = 100;
filtersNum = 4;
filter1 = [20 45];
filter2 = [3 15];
filter3 = [0.1 2];
trainConfusionMatrix = zeros(4);
testConfusionMatrix = zeros(4);

%% Labeling
testLabel = [1, 2, 3, 4];
trainLabel = [1 * ones(1, size(data{1}, 3) - 1), ...
              2 * ones(1, size(data{2}, 3) - 1), ...
              3 * ones(1, size(data{3}, 3) - 1), ...
              4 * ones(1, size(data{4}, 3) - 1)];

%% Run
for runtimeNum = 1:runNum
    %% Initial Assignment
    ind1 = randi(size(data{1}, 3));
    ind2 = randi(size(data{2}, 3));
    ind3 = randi(size(data{3}, 3));
    ind4 = randi(size(data{4}, 3));
    train = {data{1}(1:end-1, :, [1:ind1-1 ind1+1:end]), data{2}(1:end-1, :, [1:ind2-1 ind2+1:end]), ...
             data{3}(1:end-1, :, [1:ind3-1 ind3+1:end]), data{4}(1:end-1, :, [1:ind4-1 ind4+1:end])};
    test = {data{1}(1:end-1, :, ind1), data{2}(1:end-1, :, ind2), ...
            data{3}(1:end-1, :, ind3), data{4}(1:end-1, :, ind4)};
    
    %% First Classification
    [one, all] = deal(train{4}, cat(3, train{1:3}));
    one = Filter(one, filter1);
    all = Filter(all, filter1);
    [W_csp1, W_lda1, thr1, rule1] = Train(one, all, filtersNum);

    %% Second Classification
    [one, all] = deal(train{3}, cat(3, train{1:2}));
    one = Filter(one, filter2);
    all = Filter(all, filter2);
    [W_csp2, W_lda2, thr2, rule2] = Train(one, all, filtersNum);

    %% Third Classification
    [one, all] = deal(train{2}, train{1});
    one = Filter(one, filter3);
    all = Filter(all, filter3);
    [W_csp3, W_lda3, thr3, rule3] = Train(one, all, filtersNum);

    %% Checking trian
    trainConfusionMatrix = trainConfusionMatrix + Test(cat(3, train{1:4}), ...
         W_csp1, W_lda1, thr1, rule1, filter1, ...
         W_csp2, W_lda2, thr2, rule2, filter2, ...
         W_csp3, W_lda3, thr3, rule3, filter3, trainLabel);

    %% Checking test
    testConfusionMatrix = testConfusionMatrix + Test(cat(3, test{1:4}), ...
         W_csp1, W_lda1, thr1, rule1, filter1, ...
         W_csp2, W_lda2, thr2, rule2, filter2, ...
         W_csp3, W_lda3, thr3, rule3, filter3, testLabel);

end
%% Final
trainConfusionMatrix = trainConfusionMatrix ./ sum(trainConfusionMatrix, 2);
testConfusionMatrix = testConfusionMatrix ./ sum(testConfusionMatrix, 2);
disp("Training Confusion Matrix:")
disp(trainConfusionMatrix)
disp("Test Confusion Matrix:")
disp(testConfusionMatrix)