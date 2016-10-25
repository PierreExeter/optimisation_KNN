def bin_to_params(filename):
    """ convert binary decision vector into params for sklearn """

    # read knn params
    f = open(filename, "r")
    params_list = f.read().split('\n')
    f.close()    
    
    params_list.remove('')
    
    for i in range(len(params_list)):
        params_list[i] = int(params_list[i])
    
    chunks = [params_list[x:x+6] for x in xrange(0, len(params_list), 6)]
    
    for item in chunks:
        for i in range(len(item)):
            item[i] = str(item[i])
    
    chunks_str = []
    for item in chunks:
        chunks_str.append(''.join(item))
    
    params_list = []
    for item in chunks_str:
        params_list.append(int(item, 2))
    
    
    # convert decision vector from integer [0, 63] to the values for knn function
    
    if params_list[0] >= 32:
        algorithm = 'ball_tree'
    else:
        algorithm = 'kd_tree'
        
    leaf_size = params_list[1]
    n_neighbors = params_list[2]
    
    if params_list[3] >= 32:
        p = 1
    else:
        p = 2
        
    if params_list[4] >= 32:
        weights = 'uniform'
    else:
        weights = 'distance'
            
    return algorithm, leaf_size, n_neighbors, p, weights
