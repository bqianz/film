import numpy as np
from matplotlib.colors import to_hex
import gmplot as gp

def plot_component_strength_map(strengths, channel, name, latitudes, longitudes, size = 50):
    # strenghts: 1d array
    # channel: int e.g 0, 1, or 2
    # name: str e.g "components_analysis/components_strenghts_n{}.html".format(n_components)

    length = len(strengths)

    colour = np.zeros([length, 3])

    for i, c in enumerate(strengths):
        if c > 0:
            colour[i, channel] = c
        else:
            for k in range(3):
                if k!=channel:
                    colour[i, k] = -c
    
    colour = colour / np.max(colour.flatten()) # all values magnitude between 0 and 1

    colour_hex = [None] * colour.shape[0]

    for i, color in enumerate(colour):
        colour_hex[i] = to_hex(color)

    with open('components_analysis/apikey.txt') as f:
        apikey = f.readlines()[0]
    
    gmap = gp.GoogleMapPlotter(np.mean(latitudes), np.mean(longitudes), 8, apikey = apikey) #not sure what 8 does


    gmap.scatter(latitudes, longitudes, size = size, color=colour_hex, marker=False) 


    gmap.draw(name)