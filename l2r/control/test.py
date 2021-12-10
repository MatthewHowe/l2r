from NMPC import BicycleModel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def test_tyre_model():
    vehicle_model = BicycleModel()

    F_z = 1000
    F_nomz = 4000

    points = 20

    data = pd.DataFrame()
    data = {"s": [], "alpha": [], "F_x": [], "F_y": []}
    s_vals = []
    F_x_vals = []

    alpha_vals = []
    F_y_vals = []

    for i in range(points):
        s = -1 + i / (points / 2)
        alpha = -20 + i * 20 / (points / 2)
        data["s"].append(s)
        data["alpha"].append(alpha)

        F_x = vehicle_model.longitudinal_tyre_force(s, F_z, F_nomz)
        F_y = vehicle_model.lateral_tyre_force(alpha, F_z, F_nomz)

        data["F_x"].append(F_x)
        data["F_y"].append(F_y)

    data = pd.DataFrame(data)

    sns.lineplot(x=data["s"], y=data["F_x"])
    plt.show()

    sns.lineplot(x=data["alpha"], y=data["F_y"])
    plt.show()


if __name__ == "__main__":
    test_tyre_model()
