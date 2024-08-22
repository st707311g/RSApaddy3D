import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit
from skimage.morphology import ball


def covert_distance_transformation_value_to_root_diameter(v: float):
    a = 1.98316325
    b = 0.67376092
    c = -1.99578897
    return v * 2 + (a * (v * 2 + c)) / (b + (v * 2 + c)) - 1


# // Estimation of formulas for converting from distance transformed values to diameter
if __name__ == "__main__":

    def MichaelisMenten_func(X, a, b, c):
        X = np.asarray(X)
        Y = (a * (X + c)) / (b + (X + c)) - 1
        return Y

    x = []
    y = []
    diff = []
    for radius in range(1, 50):
        b = ball(radius - 1)
        dm = distance_transform_edt(b)
        real_diameter = radius * 2 - 1
        calculated_diameter = dm[radius - 1, radius - 1, radius - 1] * 2

        y.append(real_diameter)
        x.append(calculated_diameter)
        diff.append(real_diameter - calculated_diameter)

    popt, pcov = curve_fit(MichaelisMenten_func, x, diff)

    fig = plt.figure(figsize=(4, 3), tight_layout=True)
    plt.scatter(x, diff, label="simulated")
    plt.plot(x, [MichaelisMenten_func(it, *popt) for it in x], label="fitted")
    plt.legend()
    plt.ylabel("actual diameter - distance\u00d72")
    plt.xlabel("distance\u00d72")
    # plt.savefig("diameter.svg")
    plt.show()
