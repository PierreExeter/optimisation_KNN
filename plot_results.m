% save ibea_results.mat
% load ibea_results.mat

figure;
hold on;

plot(Archive_objectives(:, 1), Archive_objectives(:, 2), 'k+')
plot(samples_objectives(:, 1), samples_objectives(:, 2), 'ro')

xlabel('(-1) x cross validation');
ylabel('(-1) x accuracy');

legend('Archive', 'Samples');