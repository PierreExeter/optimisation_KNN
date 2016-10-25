function write_best(index_best)
% write best decision vector to file

load ibea_results.mat

x = Archive(index_best, :);

fid = fopen('best_params.txt','w');
fprintf(fid,'%d\n', x);
fclose(fid);