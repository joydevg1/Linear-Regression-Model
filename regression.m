clear;
data=csvread('data_1.csv');
X=data(:,[1:4]);
y=data(:,5);
theta=zeros(1,4)'; 
#plotmatrix(features,labels)
alpha=0.0001;
num_iters =10;
hypo=zeros(size(X,1),1);
m = size(X,1); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    old_theta = theta;
    theta_0 = 0;
    theta_1 = 0;
    theta_2 = 0;
    theta_3 = 0;
    for i = 1:m
         theta_0 += ((X(i, :) * old_theta) - y(i)) * X(i, 1); 
         theta_1 += ((X(i, :) * old_theta) - y(i)) * X(i, 2);
         theta_2 += ((X(i, :) * old_theta) - y(i)) * X(i, 3);
         theta_3 += ((X(i, :) * old_theta) - y(i)) * X(i, 4);
    endfor
    theta(1) = theta(1) - (alpha * (1 / m) * theta_0); 
    theta(2) = theta(2) - (alpha * (1 / m) * theta_1);
    theta(3) = theta(3) - (alpha * (1 / m) * theta_2); 
    theta(4) = theta(4) - (alpha * (1 / m) * theta_3);

    % ============================================================

    % Save the cost J in every iteration    
    h = X * theta;
    J=0;
    for j = 1:m
      J += (h(j) - y(j))**2; 
    endfor
      
    J = J / (2 * m);

    #disp(theta);
    disp(J);
end
