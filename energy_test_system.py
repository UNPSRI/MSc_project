def EnergyBrick( position):
    #hull = ConvexHull(initial_position)
    #vertice = hull.vertices
    #x = np.array([initial_position[i, 0] for i in vertice])
    #y = np.array([initial_position[i, 1] for i in vertice])
    #z = np.array([initial_position[i, 2] for i in vertice])
    k = 10**30
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    a = 1
    b = 1
    c = 1
    m = 1
    n = 1
    p = 1
    
    loss1 = 0.0
    for i in range( len( position ) ):
        x = position[i][0]
        y = position[i][1]
        z = position[i][2]
        
        #This part penalises particles that are outside the shape, with a spring that brings them back
        rel_pos = ((x/a)**(2*m) + (y/b)**(2*n) + (z/c)**(2*p))
               
        if rel_pos <= 1:
            loss1 += 0.0
        else:
            loss1 += k * rel_pos**2
    
    ##Calculate repulsive contribution, this should distribute particles homogeneously within the volume
    n2 = -10
    r_ij = pdist( position )
    #loss2 = ( r_ij**n2 ).sum() 
    loss2 = 500*np.exp( -r_ij/max(a,b,c) ).sum()   
    
    loss3 = 0
    f = lambda xn: a*np.cos(xn) #*np.sin(yn) #+ np.cos(xn)
    g = lambda xn: b*np.sin(xn) #*np.sin(yn) #+ np.sin(xn)
    h = lambda xn: c*np.cos(xn*0.5)
    ## Define minimum distance function
    d_min2 = lambda xn,x,y,z:  ( (x - f(xn) )**2 + (y - g(xn))**2 + (z - h(xn))**2 ).min()
    for i in range (len(position)-490):
        phi = np.linspace(0,2*np.pi,100)
        #theta = np.linspace(-np.pi,np.pi,20)
        #theta, phi = np.meshgrid(theta, phi)
        loss3 += d_min2(phi,position[i][0],position[i][1],position[i][2])**40
        
    
    loss4 = 0
    for i in range (len(position)-490):
        loss4 += 0.5 * (500) * ((position[i]-position[i+1])**2).sum()
   
    #loss3 = 0
    #h = 0.01
    #for i in range (len(position)):
        # Initial guess
        #x0 = -1
        # Solve using Newton-Raphson method
        #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
        #if root == None:
            #x0 = 1
            #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
            #if root == None:
                #print (root)
                #x0 = 0
                #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
        #loss3 += f_0(root, position[i][0], position[i][1], position[i][2])**(6)
    
    return loss1 + loss2 

def EnergyLine( position):
    #hull = ConvexHull(initial_position)
    #vertice = hull.vertices
    #x = np.array([initial_position[i, 0] for i in vertice])
    #y = np.array([initial_position[i, 1] for i in vertice])
    #z = np.array([initial_position[i, 2] for i in vertice])
    k = 10**30
    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    a = 1
    b = 1
    c = 1
    m = 2
    n = 2
    p = 2
    
    loss1 = 0.0
    for i in range( len( position ) ):
        x = position[i][0]
        y = position[i][1]
        z = position[i][2]
        
        #This part penalises particles that are outside the shape, with a spring that brings them back
        rel_pos = ((x/a)**(2*m) + (y/b)**(2*n) + (z/c)**(2*p))
               
        if rel_pos <= 1:
            loss1 += 0.0
        else:
            loss1 += k * rel_pos**2
    
    ##Calculate repulsive contribution, this should distribute particles homogeneously within the volume
    n2 = -6
    r_ij = pdist( position )
    #loss2 = ( r_ij**n2 ).sum() 
    loss2 = (500*np.exp( -r_ij/2 )).sum()   
    
    loss3 = 0
    ## Define minimum distance function
    #d_min2 = lambda xn,x,y,z:  ( (x - f(xn) )**2 + (y - g(xn))**2 + (z - h(xn))**2 ).min()
    d_min2 = lambda x,x0,y0,z0: min((np.cos(x) - x0)**2 + (np.sin(x) - y0)**2 + ( 20*x/(8*np.pi) - z0)**2)
    for i in range (len(position)):
        phi = np.linspace(0,10*np.pi,40)
        #theta = np.linspace(-np.pi,np.pi,20)
        #theta, phi = np.meshgrid(theta, phi)
        loss3 += d_min2(phi,position[i][0],position[i][1],position[i][2])**50
        
    
    loss4 = 0
    for i in range (len(position)-1):
        loss4 += 0.5 * (150) * ((position[i]-position[i+1])**2).sum()
   
    #loss3 = 0
    #h = 0.01
    #for i in range (len(position)):
        # Initial guess
        #x0 = -1
        # Solve using Newton-Raphson method
        #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
        #if root == None:
            #x0 = 1
            #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
            #if root == None:
                #print (root)
                #x0 = 0
                #root = newton_raphson(f_0, x0, h, position[i][0], position[i][1], position[i][2])
        #loss3 += f_0(root, position[i][0], position[i][1], position[i][2])**(6)
    
    return loss3 + loss4 + loss2 
