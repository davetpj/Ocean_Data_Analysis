import numpy as np


def cos_agerage(data, lat, lon):

    cosarray = np.zeros((len(lat), len(lon)))
    for x in range(len(lat)):
        cosarray[x, :] = np.cos(lat[x]*np.pi/180)

    idxNaN2d = (np.isnan(data[0, :, :]) == True)
    cosarray[idxNaN2d] = np.NaN

    wgt_data0 = np.zeros(data.shape)
    for t in range(0, len(data)):
        wgt_data0[t, :, :] = data[t, :, :]*cosarray

    wgt_data = np.nansum(wgt_data0, 2)
    avg_data = np.nansum(wgt_data, 1)/np.nansum(cosarray)

    return avg_data



###
def weight_average(data, weight):

    data = np.array(data)
    weight = np.array(weight)
    idx = np.where(np.isnan(data) == True)
    weight = weight.astype('float')
    weight[idx] = np.NaN
    wgt_data_avg = np.nansum(data*weight) / np.nansum(weight)

    return wgt_data_avg
