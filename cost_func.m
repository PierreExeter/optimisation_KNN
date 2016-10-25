function y = cost_func(x, m)

if m~=2
    error('Function only works with two objectives');
end

y = zeros(1, 2);

% write knn parameters to file
fid = fopen('knn_params.txt','w');
fprintf(fid,'%d\n', x);
fclose(fid);

% run model
system('python knn_model.py');

% read objectives
fid = fopen('obj1.txt','r');
y(1) = -fscanf(fid,'%f');
fclose(fid);

fid = fopen('obj2.txt','r');
y(2) = -fscanf(fid, '%f');
fclose(fid);
