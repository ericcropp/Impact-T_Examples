from impact import Impact
from distgen import Generator

def update_distgen(G,settings=None,verbose=False):
    G.verbose=verbose
    if settings:
        for key in settings:
            val = settings[key]
            if key.startswith('distgen:'):
                key = key[len('distgen:'):]
                if verbose:
                    print(f'Setting distgen {key} = {val}')
                G[key] = val
            
    
    # Get particles
    
    return G

def update_impact(I,settings=None,
               impact_config=None,
               verbose=False):
    
    I.verbose=verbose
    if settings:
        for key in settings:
            val = settings[key]
            if not key.startswith('distgen:'):
               # Assume impact
                if verbose:
                    print(f'Setting impact {key} = {val}')          
                I[key] = val                
   
    return I