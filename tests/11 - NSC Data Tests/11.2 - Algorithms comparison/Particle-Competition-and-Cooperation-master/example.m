% This is the implementation of Particle Competition and Cooperation (PCC) method 
% applied to several Data Sets from the UCI Machine Learning Repository
% This implementation is to obtain the score and kappa values for the 
% comparinson between semi-supervised learning methods
%
% by Guilherme Toso - 30/01/2020. Based on Fabricio Breve - 12/03/2019
%
% Loading the Data Set

load wine.csv; 

% Getting the dataset attributes (all colums, except the last one).
X = wine(:,1:end-1);
disp(X)
% Getting dataset labels (last column). Labels should be >0 and in
% sequence. Ex.: 1, 2, 3.
label = int32(wine(:,end));

% num of iterations
iters = 10;

% Accuracy and Kappa's vectores
acc_vec = NaN*ones(iters,1);
k_vec = NaN*ones(iters,1);

for i=1:iters
    % Randomly selecting 10% of the labels to be presented to the algorithm.
    % An unlabeled item is represented by 0.
    slabel = slabelgen(label,0.1);
    % Setting the k parameter (k-nearest neighbors)
    k = 10;
    disp('Running the algorithm in its pure Matlab implementation...')
    disp('Parameters k: 10, distance: Normalized Euclidean, others: Default.');
    tStart = tic;
    owner = pcc(X, slabel, k, 'seuclidean');
    tElapsed = toc(tStart);
    % Evaluating the classification accuracy.
    [acc,k,c] = stmwevalk(label,slabel,owner);
    fprintf('Classification accuracy: %0.4f - Execution Time: %0.4fs\n\n',acc,tElapsed);
    acc_vec(i) = acc;
    k_vec(i) = k;
    
end
% Calculating and displaying the accuracy and kappa's mean and standard 
% deviations
acc_mean = mean(acc_vec);
k_mean = mean(k_vec);
acc_std = std(acc_vec);
k_std = std(k_vec);
fprintf('\nAccuracy Mean in 10 trials: %0.4f', acc_mean);
fprintf('\nAccuracy Std in 10 trials: %0.4f', acc_std);
fprintf('\nKappa Mean in 10 trials: %0.4f', k_mean);
fprintf('\nKappa Std in 10 trials: %0.4f', k_std);

% Notice that classification accuracy may vary between the two different 
% implementations and among successive executions.
% This is expected behavior due to the algorithm''s stochastic nature.