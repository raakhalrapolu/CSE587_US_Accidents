import joblib
import numpy as np
import tensorflow as tf
import pickle
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def severity_pred(input_array, scaler, scaler_std, saved_model):
    # print(input_array, scaler_path, std_path, model_path, "----jjjjjjjjjjj")
    minmax_features = np.array([input_array[0:5]])
    bool_features = np.array([input_array[5:]])
    # print("---", minmax_features.dtype, "---", bool_features, "kshdkahdksadhkakdh")
    # scaler = joblib.load(scaler_path)
    # print(scaler)
    minmax_features = scaler.transform(minmax_features)
    # print(minmax_features, "-------------------jjjj", bool_features.shape)
    final_arr = np.array([np.concatenate((minmax_features[0], bool_features[0]))])
    # scaler_std = joblib.load(std_path)
    final_arr = scaler_std.transform(final_arr)
    out = saved_model.predict(final_arr)
    ans = np.where(out[0] == np.amax(out[0]))[0][0]
    return ans + 1

