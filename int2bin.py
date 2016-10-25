def int2bin(l):
    """ convert list of int l into list of binary ll """

    ll = []    
    
    for item in l:
        ll.append('{0:06b}'.format(item))

    return ll