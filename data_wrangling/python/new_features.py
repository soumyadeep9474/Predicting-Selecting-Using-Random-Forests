def add_neighboring_wells(df):
  pad_child_count = dict()
  pad_ids = df["pad_id"].values
  for id in pad_ids:
      if id in pad_child_count:
        pad_child_count[id] += 1
      else:
        pad_child_count[id] = 1

  df["num_neighboring_wells"] = df["pad_id"].map(pad_child_count)
  return df

def euclid_surface_bh_dist(df):
    df['surface_bottom_dist'] = ((df['surface_x'] - df['bh_x'])**2 + (df['surface_y'] - df['bh_y'])**2)**0.5
    return df

def euclid_toe_dist(df):
    df['toe_dist'] = ((df['horizontal_midpoint_x'] - df['horizontal_toe_x'])**2 + (df['horizontal_midpoint_y'] - df['horizontal_toe_y'])**2)**0.5
    return df

def surface_bottom_angle(df):
   df['surface_bottom_angle'] = np.arctan2(df['surface_y'] - df['bh_y'], df['surface_x'] - df['bh_x'])
   return df

def toe_angle(df):
   df['toe_angle'] = np.arctan2(df['horizontal_midpoint_y'] - df['horizontal_toe_y'], df['horizontal_midpoint_x'] - df['horizontal_toe_x'])
   return df
