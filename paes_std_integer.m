function [A,Ao,grid_positions,samples,samples_o] = paes_std_integer(evaluations,cost_function,l,num_obj,max_bit_flip,old_A,old_Ao,max_archive_size,grid_bisections,range_buffer)

%old_A = [];
%old_Ao = [];
% [A,Ao,grid_positions,samples,samples_o] = paes_std_integer(2, 'cost_func', 30,2,0.1,old_A,old_Ao)

% Implementation of the Pareto Archived Evolution Strategy ((1+1) varient), 
% based on the description within:
%
% Knowles, J.D., Corne, D.W. (2000) Approximating the nondominated front
% using the Pareto Archived Evolution Strategy. Evolutionary Computation, 
% 8(2), pp. 149-172
%
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
% max_archive_size = Maximum size Archive maintained before truncation 
% applied (optional argument, default 100) 
% grid_bisections  = Number of grid bisections in each dimension (optiomal
% argument, default 5)
% range_buffer = Proportion of objective range used as a buffer (at each
% end) before resizing grid and reallocating (optional argument, default
% 0.1)
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

if nargin ==9
    range_buffer=0.1;
    fprintf('Range buffer not specified, so set to 10 percent\n');
elseif nargin ==8
    grid_bisections=5;
    range_buffer=0.1;
    fprintf('Grid bisection not specified, so set to 5\n');
    fprintf('Range buffer not specified, so set to 10 percent\n');
elseif nargin ==7
    max_archive_size=100;
    grid_bisections=5;
    range_buffer=0.1;
    fprintf('Maximum archive size not specified, so set to 100\n');
    fprintf('Grid bisection not specified, so set to 5\n');
    fprintf('Range buffer not specified, so set to 10 percent\n');
elseif nargin ==5
    max_archive_size=100;
    grid_bisections=5;
    range_buffer=0.1;
    fprintf('Maximum archive size not specified, so set to 100\n');
    fprintf('Grid bisection not specified, so set to 5\n');
    fprintf('Range buffer not specified, so set to 10 percent\n');
    old_A=[];
    old_Ao=[];
end

% holding matrices for all locations visited
samples = zeros(evaluations,l);
samples_o = zeros(evaluations,num_obj);

if isempty(old_A)
    % Create random indiviual (Uniform) and evaluate
    %c = rand(1, l);
    indices = randperm(l);
    c =zeros(1,l);
    c(indices(1:ceil(max_bit_flip*l))) = 1; % random max_nbit_flip proportion set at 1.
    c_objectives = feval(cost_function,c,num_obj);
    
    % In the case of the first iteration, solutions with domination count 0
    % are the estimated Pareto front
    A = c;
    Ao = c_objectives;
      
    samples(1,:) = c;
    samples_o(1,:) = c_objectives;
    c_index=1;
    %initalise evaluation counter and print counter
    num_evaluations = 1;
else
    A = old_A; 
    Ao = old_Ao;
    indices = randperm(size(old_A,1));
    c_index = indices(1);
    c = old_A(c_index,:);
    c_objectives = old_Ao(c_index,:);
    num_evaluations =0;
end
  

print_counter = num_evaluations;

%initialise the ranges
range_max=c_objectives;
range_min=c_objectives;
grid_positions=zeros(1);

%Iterate until number of evalutions reaches maximium
kk=0;
while (num_evaluations<evaluations)
    % iterate, generating lamdba solutions each evalution from the mu
    % parents, and then replacing the parents with the mu best children
    % (mu,lambda) or replacing the parents with the mu best of the
    % children and the parents (mu+lambda)
    kk=kk+1;
    m = perturb(c,l,max_bit_flip);
    m_objectives = feval(cost_function,m,num_obj);
    if (dominates(c_objectives,m_objectives,num_obj)==0) %if c dominates m then discard
        % otherwise
        if (dominates(m_objectives,c_objectives,num_obj)==1) %if m dominates c, replace c with m
            c=m;
            c_objectives=m_objectives;
            [A,Ao, grid_positions, c_index]=add_to_archive(m,m_objectives,A,Ao,range_max,range_min,grid_positions,grid_bisections,range_buffer);
        elseif (archive_dominates(Ao,m_objectives)==0) %if Archive dominates m then discard otherwise apply test
            %to see if added to archive and if c updated
            [c,c_objectives,A,Ao,grid_positions,range_max,range_min,c_index]=apply_test(c,c_objectives,m,m_objectives,A,Ao,grid_positions,max_archive_size,grid_bisections,range_max,range_min,range_buffer,c_index);
        end
    end
    num_evaluations = num_evaluations+1;
    samples(num_evaluations,:) = m;
    samples_o(num_evaluations,:) = m_objectives;
    print_counter = print_counter + 1;
    
        fprintf('Evaluations %d, Archive size %d, min obj1: %f, min obj 2:%f\n', num_evaluations, size(A,1),min(Ao(:,1)), min(Ao(:,2)));
    if (print_counter >= 100)
        save temp_moes_res.mat
        %print_counter = print_counter - 100;
    end
    % random sample of parent from archive
    I = randperm(size(A,1));
    c = A(I(1),:);
    c_objectives = Ao(I(1),:);
end

if isempty(old_A)~=1
   samples = [old_A; samples];
   samples_o = [old_Ao; samples_o];
end

%--------------------------------------------------------------------------
function c = perturb(c,l,mbf)

n = ceil(mbf*l);
r = randperm(n);
r = r(1); % will flip 'r' elements, where 1<=r<=n
index = randperm(l);
% for a random 'r' elements
c(index(1:r)) = abs(c(index(1:r))-1);


% %Perturbs a single parameter of a solution vector
% I=randperm(l);
% i=I(1); %select a decision variable at random
% r=-1;
% while (r<0) || (r>1)    %ensure in valid range
%     r=randn()*std_mut+c(i); %mutate with additive Gaussian noise
% end
% c(i)=r;
%--------------------------------------------------------------------------
function [A,Ao, grid_positions,c_index]=add_to_archive(m,m_objectives,A,Ao,range_max,range_min,grid_positions,grid_bisections,range_buffer)
% Inserts x into archive and removes
% any dominated members from archive -- this is only called when m dominates c,
% as by definition 'c' is currently in the archive, at least one archive
% member will be removed when the new entrant is added, so will not grow
% beond size limit and we don't therefore need to check for this

[num,l] = size(A);
[num,d] = size(Ao);

dominated = zeros(num,1);   % count dominated archive members
for i=1:d
    dominated = dominated + (Ao(:,i)>=m_objectives(i));
end
I=find(dominated==d); % find and remove dominated archive members
A(I,:)=[];
Ao(I,:)=[];
grid_positions(I)=[];
A=[A; m];                   % insert new archive member
Ao=[Ao; m_objectives];
c_index=size(A,1);          % as c is replaced by m when this is called, record index into archive too
[grid_positions] = range_change(range_max,range_min,grid_positions,Ao,grid_bisections,m_objectives,d,range_buffer);
[grid_positions] = insert_into_grid(grid_positions,range_max,range_min,m_objectives,grid_bisections,d,size(A,1));
%--------------------------------------------------------------------------
function x=dominates(u,v,num_obj)
% Returns 1 if u dominates v, 0 otherwise
x = 0;
wd = sum(u<=v);
d = sum(u<v);
if ((wd==num_obj) && (d>0))
    x = 1;
end
%--------------------------------------------------------------------------
function x=archive_dominates(A,v,num_obj)
% Returns 1 if any members of A dominate v, 0 otherwise
x = 0;
[num,num_obj]=size(A);
dominating = zeros(num,1); % count dominating archive members
w_dominating = zeros(num,1); % count dominating archive members
for i=1:num_obj
    w_dominating = w_dominating + (A(:,i)<=v(i));
    dominating = dominating + (A(:,i)<v(i));
end
I=find(w_dominating==num_obj);
I2=find(dominating(I)>0);
if (length(I2)>0)
    x=1;
end
%--------------------------------------------------------------------------
function [x,index]=dominates_archive(A,v,num_obj)
% Returns 1 if v dominates any members of A, 0 otherwise
% index contains the indices of members of A dominated by v
x = 0;
index=[];
[num,num_obj]=size(A);
dominating = zeros(num,1); % count dominating archive members
w_dominating = zeros(num,1); % count dominating archive members
for i=1:num_obj
    w_dominating = w_dominating + (A(:,i)>=v(i));
    dominating = dominating + (A(:,i)>v(i));
end
I=find(w_dominating==num_obj);
I2=find(dominating(I)>0);
if (length(I2)>0)
    x=1;
    index=I(I2);
end
%-------------------------------------------------------------------------
function [c,c_objectives,A,Ao,grid_positions,range_max,range_min,c_index]=apply_test(c, ...
    c_objectives,m,m_objectives,A,Ao,grid_positions,max_arcive_size, ...
    grid_bisections,range_max,range_min,range_buffer,c_index)

d=length(m_objectives);
%this part not in pre-print -- but NDS properties of archive not
%guaranteed without it, so assuming it was simple omission
[x,I]=dominates_archive(Ao,m_objectives,d);
Ao(I,:)=[];
A(I,:)=[];
grid_positions(I)=[];
l=sum(I<c_index);
c_index=c_index-l; %change to take into account removed elements, note that
                   %due to the design of PAES, c will not be amongst those
                   %reomved at this point so don't need to worry about that
                   %special case

%update the ranges of the gridding if beyond limits, or shrinking
%include 'buffer_range' extra amount
[grid_positions] = range_change(range_max,range_min,grid_positions,Ao,grid_bisections,m_objectives,d,range_buffer);

n=size(Ao,1);
if (n<max_arcive_size);
    A(end+1,:)=m;
    Ao(end+1,:)=m_objectives;
    [grid_positions]=insert_into_grid(grid_positions,range_max,range_min,m_objectives,grid_bisections,d,size(A,1));
    m_crowd=length(find(grid_positions==grid_positions(end)));
    c_crowd=length(find(grid_positions==grid_positions(c_index)));
    %if m is in a less crowded reagion than c, make as new c
    if (m_crowd<c_crowd)
        c=m;
        c_objectives=m_objectives;
        c_index=size(A,1);      %record index of new c in archive
    end
else %max archive size already reached, so will have to decide whether to accept or not
    u=unique(grid_positions); %vector containing indices of all current used grids
    m_crowd=length(find(grid_positions==grid_positions(end)));
    crowding=zeros(size(u));
    for i=1:length(u)
        crowding(i)=length(find(grid_positions==u(i)));
    end
    %if m is in less crowded area than existing populated grid
    if m_crowd<max(crowding)
        [x,i]=max(crowding);
        I=find(grid_positions==u(i)); %indices of all members of most crowded grid
        P=randperm(length(I));
        Ao(I(P(1)),:)=m_objectives;
        A(I(P(1)),:)=m;
        [grid_positions]=insert_into_grid(grid_positions,range_max,range_min,m_objectives,grid_bisections,d,I(P(1)));
        %if m is in a less crowded reagion than c, make as new c
        c_crowd=length(find(grid_positions==grid_positions(c_index)));
        if m_crowd<c_crowd
            c=m;
            c_objectives=m_objectives;
            c_index=I(P(1));  %record index of new c in archive
        end
    end
    % lines 13-15 of the Psudocode for test(c,m,archive) in the preprint
    % of Knowles and Corne's 2000 paper will never be evaluated as it is
    % an 'else' statement on the crowding -- but if m is not less crowded
    % than the most crowded region in the archive it will definitely not
    % be less crowded than c (as in the worst case, this will be in the
    % most crowded region, which it is already evalauted as being no
    % less crowded than. As such I have not put the code in here.
end

%-----------------------
function [grid_positions]=insert_into_grid(grid_positions,range_max,range_min,m_objectives,grid_bisections,d,index)

prop_space=(m_objectives-range_min)./(range_max-range_min);
coordinate_multiplier=ones(1,d);
for i=2:d;
    coordinate_multiplier(i)=2^(grid_bisections*(i-1));
end
prop_space=prop_space.*coordinate_multiplier;
grid_positions(index)=sum(prop_space);
%-----------------------
function [grid_positions,range_max,range_min] = range_change(range_max,range_min, grid_positions,Ao,grid_bisections,m_objectives,d,range_buffer)



max_v=max([Ao;m_objectives]);
min_v=min([Ao;m_objectives]);
range=max_v-min_v;
rc=0;
if (sum(range_min>(min_v-range_buffer*range)>0))
    rc=1; %if current range min stored too big
elseif (sum(range_min<(min_v+range_buffer*range)>0))
    rc=1; %if current range min stored to small
elseif (sum(range_max<(max_v+range_buffer*range)>0))
    rc=1;  %if current range max stored too small
elseif (sum(range_max>(max_v-range_buffer*range)>0))
    rc=1; %if current range max stored too big
end

if (rc==1) % if grid ranges changed, update archive
    range_min=min_v-range_buffer*range;
    range_max=max_v+range_buffer*range;

    grid_positions=grid_positions*0; %reset positions
    min_m=repmat(range_min,size(Ao,1),1);
    max_m=repmat(range_max,size(Ao,1),1);
    prop_space=(Ao-min_m)./(max_m-min_m);
    %prop_space now contains obj-dimensioal grid coordinates for archive
    %members
    prop_space=floor(prop_space*2^grid_bisections);
    %take care of infentessimally small chance (when buffer is
    %non-zero) that top edge individual will be placed in grid 2^grid_bisections not
    %grid 2^grid_bisections-1 on that coordinate (out of 'd' coordinates)
    I=find(prop_space==2^grid_bisections);
    prop_space(I)=2^grid_bisections-1;
    coordinate_multiplier=ones(1,d);
    for i=2:d;
        coordinate_multiplier(i)=2^(grid_bisections*(i-1));
    end
    prop_space=prop_space.*repmat(coordinate_multiplier,size(Ao,1),1);
    grid_positions=sum(prop_space,2);
end