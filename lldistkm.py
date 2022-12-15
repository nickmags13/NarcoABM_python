import numpy as np


def lldistkm(latlon1, latlon2):

    radius = 6371
    lat1 = latlon1[:, 1] * np.pi / 180
    lat2 = latlon2[:, 1] * np.pi / 180
    lon1 = latlon1[:, 2] * np.pi / 180
    lon2 = latlon2[:, 2] * np.pi / 180
    deltaLat = lat2 - lat1
    deltaLon = lon2 - lon1
    a = np.sin((deltaLat) / 2) ** 2 + np.multiply(np.multiply(np.cos(lat1), np.cos(lat2)), np.sin(deltaLon / 2) ** 2)
    c = 2.0 * atan2(np.sqrt(a), np.sqrt(1 - a))
    d1km = np.multiply(radius, c)

    x = np.multiply(deltaLon, np.cos((lat1 + lat2) / 2))
    y = deltaLat
    d2km = np.multiply(radius, np.sqrt(np.multiply(x, x) + np.multiply(y, y)))

    return d1km, d2km