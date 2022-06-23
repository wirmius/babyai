import pandas as pd
import numpy as np
from optparse import OptionParser
from babyai.levels import level_dict
from tqdm import tqdm
from functools import partial

parser = OptionParser()

parser.add_option(
    "--df_path",
    default=None,
    type="str",
    help="The path to the pickled dataframe produced by the sampler."
)

parser.add_option(
    "--level",
    default=None
)

parser.add_option(
    "--batching",
    default=-1,
    type=int,
    help="Batching together the episodes into files."
)

parser.add_option(
    "--tile_size",
    default=16,
    type=int,
    help="The tile size of the rendering."
)

(options, args) = parser.parse_args()

# load the dataframe
main_df = pd.read_pickle(options.df_path)

# load the required environment, the seed doesnt matter, its only about the rendering
mission = level_dict[options.level]()

# convert all the observations images into actual renderings
rend = partial(mission.get_obs_render, tile_size=options.tile_size)

# apply the function to the observation images
rendered_obs = main_df['obs_env'].apply(rend)#.to_list()
#rendered_obs = np.stack(rendered_obs, axis=0)

main_df['obs_env_img'] = rendered_obs

pd.to_pickle(main_df, options.df_path)
# save as an npz file
# np.savez(options.df_path,x=rendered_obs)