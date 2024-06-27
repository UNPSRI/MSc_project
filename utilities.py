import numpy as np

def marsaglia_polar(dim):
    while True:
        u1 = random.uniform(-1, 1)
        u2 = random.uniform(-1, 1)
        u3 = random.uniform(-1, 1)
        s = u1**2 + u2**2 
        if s >= 1 or s == 0:
            continue
        #print (s)
        factor = np.sqrt(-2.0 * np.log(s) / s)
        z0 = u1 * factor
        z1 = u2 * factor
        #print (z0,z1)
        if dim == 3:
            while True:
                u3 = random.uniform(0, 1)
                s = u1**2 + u2**2 + u3**2
                if s >= 1 or s == 0:
                    continue
                factor3 = np.sqrt(-2.0 * np.log(s) / s)
                z0 = u1 * factor3
                z1 = u2 * factor3
                z2 = u3 * factor3
                return z0, z1, z2
        return z0, z1
