import math
import numpy as np
import torch
import wandb

class StructuredDatasetBase():
    is_onehot = None  # must be set by subclass

    def __getitem__(self, index):
        raise NotImplementedError

    def get_lat_long_points(self, data):
        """
        Extracts and returns the latitude and longitude points from the dataset.
        """
        start_lat = data[:, 6]  #7th column is StartLatitude
        start_long = data[:, 7]  #8th column is StartLongitude
        end_lat = data[:, 8]  #9th column is EndLatitude
        end_long = data[:, 9]  #10th column is EndLongitude
        return start_lat, start_long, end_lat, end_long

    def log_batch(self, data, return_dict=False):
        d = {}
        start_lat, start_long, end_lat, end_long = self.get_lat_long_points(data)

        if torch.isnan(start_lat).any() or torch.isnan(start_long).any() or \
           torch.isnan(end_lat).any() or torch.isnan(end_long).any():
            print('Not logging NaN tensor')
            return d

        d["Samples/StartLatitude"] = wandb.Histogram(start_lat.cpu().numpy())
        d["Samples/StartLongitude"] = wandb.Histogram(start_long.cpu().numpy())
        d["Samples/EndLatitude"] = wandb.Histogram(end_lat.cpu().numpy())
        d["Samples/EndLongitude"] = wandb.Histogram(end_long.cpu().numpy())

        if return_dict:
            return d
        wandb.log(d)

    def load_network_state_dict(*args, **kwargs):
        # Assume no state
        pass

class GraphicalStructureBase():

    def adjust_st_batch(self, st_batch):
        # for things like adjusting coordinates if needed
        pass

    def get_auto_target(self, st_batch, adjust_val):
        return st_batch.get_flat_lats()

    def get_auto_target_IS(self, xt_dp1_st_batch, adjust_val_dp1, adjust_val_d):
        return xt_dp1_st_batch.get_flat_lats()

def gridify_lat_long(start_lat, start_long, end_lat, end_long):
    B_ = start_lat.shape[0]
    rows = math.ceil(math.sqrt(B_))
    cols = math.ceil(B_ / rows)

    lat_long_points = np.stack([start_lat, start_long, end_lat, end_long], axis=1)
    lat_long_points = np.concatenate([lat_long_points, np.zeros([rows*cols - B_, *lat_long_points.shape[1:]])], axis=0)
    B, L = lat_long_points.shape
    lat_long_points = lat_long_points.reshape(rows, cols, L)  # rows cols L
    return lat_long_points
