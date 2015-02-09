function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

%m = length(X);
%one = zeros(m,1);
%zero = zeros(m,1);
%
%onecount = 0;
%zerocount = 0;
%
%for iter = 1:m
%	if(y(iter) == 1)
%		onecount = onecount + 1;
%		one(onecount) = iter;
%	else
%		zerocount = zerocount + 1;
%		zero(zerocount) = iter;
%	endif
%end
%one = one(1:onecount);
%zero = zero(1:zerocount);
%
%plot(X(one, 1), X(one, 2), 'k+','LineWidth', 2,'MarkerSize', 7);
%plot(X(zero, 1), X(zero, 2), 'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% =========================================================================



pos = find(y==1); neg = find(y == 0);
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
hold off;

end
