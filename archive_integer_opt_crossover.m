function [A,Ao,samples,samples_o] = archive_integer_opt_crossover(evaluations,cost_function,l,num_obj,max_bit_flip,old_A,old_Ao,old_samples,old_samples_o)

%old_A = [];
%old_Ao = [];
%[A,Ao,grid_positions,samples,samples_o] = archive_integer_opt_crossover(10,'cost_func',30,2,0.1, old_A, old_Ao)

% ARGUMENTS
% evaluations = number of total function evaluations to run optimiser for
% cost_function = objective function, must accept as arguments a real
% vector of length l an a integer denoting the number of objectives, and a 
% structure of additional function arguments, must return a vector of 
% 'number of objective' elements. All parameters are on the range [0,1]  
% l = number of parameters
% num_obj = number of objectives 
% max_bit_flip = maximum proportion of elements that may be flipped (0,1]
% old_A = archive output of pervious run -- set as empty set [] if you do
%         not wish to restart from previous run
% old_Ao = archive objectives output of pervious run -- set as empty set [] 
%          if you do not wish to restart from previous run
%
% RETURNS
% A = Archive solutions
% Ao = Corresponding archive objective evaluations
% grid_positions = location of archive members in grid
% samples = matrix of all locations visited (evaluations by l elements)
% samples_o = objective evaluations of all locations visited (evalulations
% by num_obj elements)
%
% Jonathan Fieldsend, University of Exeter, 12/3/09

% INITIALISATION

% holding matrices for all locations visited
samples = zeros(evaluations, l);
samples_o = zeros(evaluations, num_obj);

if isempty(old_A)
    % Create random indiviual (Uniform) and evaluate
    % c = rand(1, l);
    indices = randperm(l);
    c = zeros(1, l);
    c(indices(1:ceil(max_bit_flip*l))) = 1; % random max_nbit_flip proportion set at 1.
    c_objectives = feval(cost_function, c, num_obj);
    
    % In the case of the first iteration, solutions with domination count 0
    % are the estimated Pareto front
    A = c;
    Ao = c_objectives;
      
    samples = c;
    samples_o(1, :) = c_objectives;
    c_index = 1;
    %initalise evaluation counter and print counter
    num_evaluations = 1;
else
    A = old_A; 
    Ao = old_Ao;
    samples = old_samples;
    samples_o = old_samples_o;
    indices = randperm(size(old_A, 1));
    c_index = indices(1);
    c = old_A(c_index, :);
    c_objectives = old_Ao(c_index, :);
    num_evaluations = 0;
end
  
print_counter = num_evaluations;

% Iterate until number of evalutions reaches maximium
kk = 0;

while (num_evaluations < evaluations)
    % iterate, generating lamdba solutions each evalution from the mu
    % parents, and then replacing the parents with the mu best children
    % (mu,lambda) or replacing the parents with the mu best of the
    % children and the parents (mu+lambda)
    kk = kk+1;
    if rand() < 0.5 || size(A, 1)==1
        m = perturb(c, l, max_bit_flip);
    else
        m = crossover(A);
    end
    
    m_objectives = feval(cost_function, m, num_obj);
    
    if (dominates(c_objectives, m_objectives, num_obj)==0) %if c dominates m then discard
        % otherwise
        if (dominates(m_objectives, c_objectives, num_obj)==1) %if m dominates c, replace c with m
            c = m;
            c_objectives = m_objectives;
            [A, Ao, c_index] = add_to_archive(m, m_objectives, A, Ao);
        elseif (archive_dominates(Ao, m_objectives)==0) %if Archive dominates m then discard otherwise apply test
            %to see if added to archive and if c updated
            [c, c_objectives, A, Ao, c_index] = apply_test(c, c_objectives, m, m_objectives, A, Ao, c_index);
        end
    end
    num_evaluations = num_evaluations+1;
    samples(num_evaluations, :) = m;
    samples_o(num_evaluations, :) = m_objectives;
    print_counter = print_counter + 1;
    
    fprintf('Evaluations %d, Archive size %d, min obj1: %f, min obj 2:%f\n', num_evaluations, size(A, 1), min(Ao(:, 1)), min(Ao(:, 2)));
    
    if (print_counter >= 100)
        save temp_moes_res.mat
        %print_counter = print_counter - 100;
    end
    % random sample of parent from archive
    I = randperm(size(A, 1));
    c = A(I(1), :);
    c_objectives = Ao(I(1), :);
end

if isempty(old_A) ~= 1
   samples = [old_A; samples];
   samples_o = [old_Ao; samples_o];
end

%--------------------------------------------------------------------------
function c = perturb(c, l, mbf)

n = ceil(mbf*l);
r = randperm(n);
r = r(1); % will flip 'r' elements, where 1<=r<=n
index = randperm(l);
% for a random 'r' elements
c(index(1:r)) = abs(c(index(1:r))-1);

%--------------------------------------------------------------------------
function c = crossover(A)

I = randperm(size(A, 1));
p1 = A(I(1), :);
p2 = A(I(2), :);

mask = (p1 ~= p2);

I = find(mask==1);
index = randperm(length(I));
v = randperm(length(I));
c = p1;
c(I(index(1:v(1)))) = p2(I(index(1:v(1))));

%--------------------------------------------------------------------------
function [A, Ao, c_index] = add_to_archive(m, m_objectives, A, Ao)
% Inserts x into archive and removes
% any dominated members from archive -- this is only called when m dominates c,
% as by definition 'c' is currently in the archive, at least one archive
% member will be removed when the new entrant is added, so will not grow
% beond size limit and we don't therefore need to check for this

[num, l] = size(A);
[num, d] = size(Ao);

dominated = zeros(num,1);   % count dominated archive members
for i = 1:d
    dominated = dominated + (Ao(:,i) >= m_objectives(i));
end
I = find(dominated==d); % find and remove dominated archive members
A(I, :) = [];
Ao(I, :) = [];
A = [A; m];                   % insert new archive member
Ao = [Ao; m_objectives];
c_index = size(A, 1);          % as c is replaced by m when this is called, record index into archive too

%--------------------------------------------------------------------------
function x = dominates(u, v, num_obj)
% Returns 1 if u dominates v, 0 otherwise
x = 0;
wd = sum(u <= v);
d = sum(u < v);
if ((wd == num_obj) && (d > 0))
    x = 1;
end

%--------------------------------------------------------------------------
function x = archive_dominates(A, v, num_obj)
% Returns 1 if any members of A dominate v, 0 otherwise
x = 0;
[num, num_obj] = size(A);
dominating = zeros(num, 1); % count dominating archive members
w_dominating = zeros(num, 1); % count dominating archive members

for i=1:num_obj
    w_dominating = w_dominating + (A(:,i) <= v(i));
    dominating = dominating + (A(:,i) < v(i));
end
I = find(w_dominating == num_obj);
I2 = find(dominating(I) > 0);
if (length(I2) > 0)
    x = 1;
end

%--------------------------------------------------------------------------
function [x, index] = dominates_archive(A, v, num_obj)
% Returns 1 if v dominates any members of A, 0 otherwise
% index contains the indices of members of A dominated by v
x = 0;
index = [];
[num, num_obj] = size(A);
dominating = zeros(num, 1); % count dominating archive members
w_dominating = zeros(num, 1); % count dominating archive members

for i = 1:num_obj
    w_dominating = w_dominating + (A(:,i) >= v(i));
    dominating = dominating + (A(:,i)>v(i));
end

I = find(w_dominating == num_obj);
I2 = find(dominating(I) > 0);

if (length(I2) > 0)
    x = 1;
    index = I(I2);
end

%-------------------------------------------------------------------------
function [c, c_objectives, A, Ao, c_index] = apply_test(c, c_objectives, m, m_objectives, A, Ao, c_index)

d = length(m_objectives);
%this part not in pre-print -- but NDS properties of archive not
%guaranteed without it, so assuming it was simple omission
[x, I] = dominates_archive(Ao, m_objectives, d);
Ao(I, :) = [];
A(I,:) = [];
l = sum(I < c_index);
c_index = c_index-l; %change to take into account removed elements, note that
                   %due to the design of PAES, c will not be amongst those
                   %reomved at this point so don't need to worry about that
                   %special case

n = size(Ao, 1);

A(end+1, :) = m;
Ao(end+1, :) = m_objectives;
c = m;
c_objectives = m_objectives;
c_index = size(A,  1);      %record index of new c in archive