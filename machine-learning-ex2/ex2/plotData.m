function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


data_0 = [];
data_1 = [];

for i=1:length(y)
    if y(i)==0,
        data_0 = [data_0;X(i,:)];
    elseif y(i)==1,
        data_1 = [data_1;X(i,:)];
    else
        
    end
end

plot(data_0(:,1),data_0(:,2),'o');
legend('Not admitted');
plot(data_1(:,1),data_1(:,2),'+');
legend('Admitted');
xlabel('Student Exam 1 Results');
ylabel('Student Exam 2 Results');
title('Student Exam 1 and 2 Results Admitted vs. Not Admitted');

% =========================================================================



hold off;

end
