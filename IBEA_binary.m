function [Archive,Archive_objectives, X, Xo, samples, samples_objectives] = IBEA_binary(pop_size, generations, cost_function, l, num_obj, x_over_type, key, p_mut, kappa, old_X, old_Xo, old_samples, old_samples_o)

% Implements the adaptive IBEA_epsilon+ algorithm described in 2004 PPSN paper by
% Zitzler and Kunzli, modified to manipulate binary string representation
%
% [Archive,Archive_objectives, X, Xo, samples, samples_objectives] = IBEA_binary(10, 5, 'cost_func', 30, 2, 0.1, 1, 0.01, 0.05)
% l (number of parameters) = nb of digit needed to encode an int up to 50 * number of parameters
% l = 6*5 = 30
%
% inputs:
% pop_size = number of members in search population
% generations = number of iterations of algorithm
% cost_function = string containing the name of the objective
%   function to optimise, must take as arguments the decision vector
%   followed by the number of objectives, and return an array (1 by
%   D) of the D objectives evaluated
% l = number of decision parameters
% num_obj = number of objectives
% x_over_type = crossover type, 1 indicates single point, uniform otherwise
% key: if key = 1, no initial population, if key = 2, initial population is
% generated from the parametric optimisation, else, the initial
% population is generated from previous run with a 'samples' history
% p_mut (optional) = probability of bit flip mutation, will default to 1/l
% kappa (optinal) = discount factor for indicator, will default to 0.05
% old_X = archive output of pervious run -- set as empty set [] if you do
%         not wish to restart from previous run
% old_Xo = archive objectives output of pervious run -- set as empty set [] 
%          if you do not wish to restart from previous run
% old_samples = samples output of pervious run -- set as empty set [] if you do
%         not wish to restart from previous run
% old_samples_o = samples objectives output of pervious run -- set as empty set [] 
%          if you do not wish to restart from previous run
%
% returns:
%
% Archive = matrix of archive decision vectors
% Archive_objectives = matrix of archive member evaluations
% samples = history of algorithm state in terms of its locations evaluated
% samples_objectives = corresponding objectives
%
% (c) Jonathan Fieldsend, University of Exeter, 2014

if ~exist('p_mut','var')
    p_mut = 1;
else
    p_mut = ceil(p_mut*l);
end
% p_mut now holds the number of elements of a vector to flip each time
if ~exist('kappa','var')
    kappa = 0.05;
end

% INITIALISATION
% generates an initial population of size pop_size

if ~exist('old_X','var')
    old_X = [];
end
if ~exist('old_Xo','var')
    old_Xo = [];
end
if ~exist('old_samples','var')
    old_samples = [];
end
if ~exist('old_samples_o','var')
    old_samples_o = [];
end

[samples, samples_objectives, sample_index, X, Xo] = initialise(pop_size, key, generations, l, num_obj, cost_function, old_X, old_Xo, old_samples, old_samples_o);
mating_pool = rand(pop_size,l);
offspring = rand(pop_size,l);
off_o = rand(pop_size,num_obj); 

for kk=1:generations % loop for generations

    % FITNESS ASSIGNMENT: scale objective and indicator values and use them
    % to assign fitness value to the individual in the initial population
    Xo_scaled = rescale_objectives(Xo);
    [fitness,c] = fitness_assignment(Xo_scaled,kappa);
    
    % ENVIRONMENTAL SELECTION
    while size(X,1)>pop_size
        % select the individual with the smallest fitness value = best
        % individual !
        [~,j] = min(fitness);
        % remove it from the population and store its objectives in ty
        X(j,:) =[];
        ty = Xo_scaled(j,:);
        Xo(j,:) = [];
        Xo_scaled(j,:)=[];
        fitness(j) = [];
        % update fitness value of the remaining individuals
        fitness = update_fitness(fitness,Xo_scaled,kappa,ty,c);
    end
    
    % MATING SELECTION
    for j=1:pop_size;
        I=randperm(pop_size);
        % binary tournament selection on fitness value with replacement
        % in order to fill the temporary mating pool
        if fitness(I(1))<fitness(I(2))
            mating_pool(j,:)=X(I(1),:);
        else
            mating_pool(j,:)=X(I(2),:);
        end            
    end
    
    % VARIATION: apply crossover and mutation to the 
    % mating pool and add the resulting offspring to the main population
    offspring = mating_pool;
    % CROSSOVER
    for j=1:2:pop_size-1;
        c1 = mating_pool(j,:);
        c2 = mating_pool(j+1,:);
        if rand()<0.9 % crossover with 90% probability
            if x_over_type==1 % single point
                k = randperm(l-1);
                ks = k(1);
                c1 (ks+1:end) = mating_pool(j+1,ks+1:end);
                c2 (ks+1:end) = mating_pool(j,ks+1:end);
            else % uniform
                k = randperm(l);
                % uniformly random selected elements, crossover 50%
                uni_I = k(1:ceil(length(k)/2));
                c1 (uni_I) = mating_pool(j+1,uni_I);
                c2 (uni_I) = mating_pool(j,uni_I);
            end    
        end % otherwise children are direct copies of parents
        offspring(j,:) = c1;
        offspring(j+1,:) = c2;
    end
    
    % BIT FLIP MUTATION
    for j=1:pop_size;
        k=randperm(l);
        % randomly bitflip p_mut elements
        offspring(j,k(1:p_mut))=abs(offspring(j,k(1:p_mut))-1); 
    end
    
    % add offspring to population 
    for i=1:pop_size
        % evaluate offpsring objectives
        off_o(i,:) = feval(cost_function,offspring(i,:),num_obj);
        % add offspring to population
        samples(sample_index,:) = offspring(i,:);
        samples_objectives(sample_index,:) = off_o(i,:);
        sample_index = sample_index+1;
    end
    if rem(kk,10)==0
        % print every 10 iterations
        fprintf('Iteration %d, Evaluation %d\n',kk,kk*pop_size+pop_size);
        save temp_ibea.mat
    end
    Xo = [Xo; off_o];
    X = [X; offspring];
    
end

% TERMINATION
I = pareto_front_with_duplicates(Xo);
Archive = X(I,:);
Archive_objectives = Xo(I,:);

%----------------------------------------------
function [samples, samples_objectives, sample_index, X, Xo] = initialise(pop_size, key, generations, l, num_obj, cost_function, old_X, old_Xo, old_samples, old_samples_o)

% if no initial population
if key == 1
    
    samples = zeros((generations+1)*pop_size,l);
    samples_objectives = zeros((generations+1)*pop_size,num_obj);
    % declare archive and associated objective evaluations as empty
    % Create random indiviual (Uniform) bits and evaluate
    X = floor(rand(pop_size,l)*2);
    Xo = zeros(pop_size,num_obj);
    
    for i=1:pop_size
        Xo(i,:) = feval(cost_function,X(i,:),num_obj);
    end
    
    samples(1:pop_size,:) = X;
    samples_objectives(1:pop_size,:) = Xo;
    sample_index = pop_size+1;
    
    % if the initial population is created from the parametric optimisation
elseif key == 2
    X = old_X;
    Xo = old_Xo;
    if size(X,1) < pop_size
        temp_length = size(X,1);
        
        %%% if fewer than pop_size from parametric optimisation, then fill
        % out rest of the serach population with random solutions
        X =  [X; floor(rand(pop_size-temp_length,l)*2)];
        Xo = [Xo; zeros(pop_size-temp_length,num_obj)];
        
        for i=temp_length:pop_size
            Xo(i,:) = feval(cost_function,X(i,:),num_obj);
        end
        %%%
    end
    
    samples = zeros(generations*pop_size+size(X,1),l);
    samples_objectives = zeros(generations*pop_size+size(Xo,1),num_obj);
    
    samples(1:size(X,1),:) = X;
    samples_objectives(1:size(Xo,1),:) = Xo;
    sample_index = size(X,1)+1;
    
    % if initial population is created from previous run with a 'sample' history
else
    %%% to check with Jonathan
    if size(old_X,1) ~= 2*pop_size
        error('old population size does not match the population size now being used -- the old population should be twice the pop_size argument');
    end
    %%%
    
    X = old_X;
    Xo = old_Xo;
    samples = [old_samples; zeros(generations*pop_size+size(X,1),l)];
    samples_objectives = [old_samples_o; zeros(generations*pop_size+size(Xo,1),num_obj)];
    sample_index = size(old_samples,1);
    
    samples(sample_index+1:sample_index+size(X,1),:) = X;
    samples_objectives(sample_index+1:sample_index+size(Xo,1),:) = Xo;
    sample_index = sample_index+size(Xo,1)+1;
    
end

%----------------------------------------------
function Xo_scaled = rescale_objectives(Xo)
% rescale each objectives to the interval [0,1]

n = size(Xo,1);
upb = max(Xo);
lwb = min(Xo);

Xo_scaled = (Xo-repmat(lwb,n,1))./repmat(upb-lwb,n,1);


%----------------------------------------------
function [fitness,c] = fitness_assignment(Xo,kappa)

[n,m] = size(Xo);
fitness = zeros(n,1);
indicator = zeros(n,n);

for i=1:n
    for j=1:n
        if i~=j
            indicator(i,j) = max(Xo(i,:)-Xo(j,:)); % get shift value
            %fitness(i) = fitness(i) -exp(-indicator/kappa); 
        end
    end
end

c =max(max(indicator));

for j=1:n
    fitness(j) = sum(-exp(-indicator(:,j)/(c*kappa)));
end


%----------------------------------------------
function fitness = update_fitness(fitness,Xo,kappa,old_val,c)

[n,m] = size(Xo);
for i=1:n
    indicator = max(old_val-Xo(i,:)); % get shift value
    fitness(i) = fitness(i) + exp(-indicator/(c*kappa)); 
end


%----------------------------------------------
function [indices] = pareto_front_with_duplicates(Y)
% Y = A n by m matrix of objectives, where m is the number of objectives
% and n is the number of points
%
% copes with duplicates
% assumes minimisation

[n,m] = size(Y);
S = zeros(n,1);


for i=1:n
    % get number of points that dominate Y
    S(i) = sum((sum(Y<=repmat(Y(i,:),n,1),2) == m) & (sum(Y<repmat(Y(i,:),n,1),2) > 0));
end
indices = find(S==0);
