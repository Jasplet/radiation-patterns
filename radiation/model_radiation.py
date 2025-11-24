# Author: J Asplet

# Functions to plot radiation patterns

# To-writes:
# 1. Rad pattern P (1-D)
# 2. Rad pattern SV/SH (1-D)
# 3. 2-D radiation patterns (for P/SV/SH)
# 4. Beach ball
# 5. 4-panel combination

import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.gridspec import GridSpec
# import obspy
from numpy import deg2rad
from radiation_patterns.radiation import patterns


def plot_2d_radiation_patterns(fault, stations, nazis=361, ntakeoffs=91, out=None):
    """
    Calculate 2-D (azimuth/takeoff) radiation patterns and return a
    4-panel plot for P, SH, SH, and a "pseudo" beachball
    """

    # Calculate 2-D radiation patterns
    rad_P, rad_SV, rad_SH = patterns.radiation_patterns_2d(fault, nazis, ntakeoffs)

    # Mean radiation pattern over whole focal sphere per Boore and Boatwright (1984).
    mean_P = 0.44
    mean_S = 0.6
    # normalise radiation patterns by mean amplitude
    rad_P = rad_P / mean_P
    rad_SV = rad_SV / mean_S
    rad_SH = rad_SH / mean_S

    azis = np.linspace(0, 2 * np.pi, nazis)
    takeoffs_plot = np.sqrt(2) * np.sin(np.linspace(0, np.pi / 4, ntakeoffs))

    figure, axs = plt.subplots(
        2, 2, layout="constrained", subplot_kw={"projection": "polar"}
    )
    ax1 = axs[0, 0]
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction("clockwise")
    # ax1.grid(False)
    C1 = ax1.pcolormesh(
        azis,
        takeoffs_plot,
        rad_P,
        shading="nearest",
        cmap="PuOr",
        vmin=-1 / mean_P,
        vmax=1 / mean_P,
    )
    ax1.contour(azis, takeoffs_plot, np.abs(rad_P), levels=[1], colors="black")

    ax1.set_rticks([])
    ax1.set_title("a) P Radiation Pattern", fontsize=12)
    plt.colorbar(C1, ax=ax1)
    # Second Panel
    ax2 = axs[0, 1]
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction("clockwise")
    # ax1.grid(False)
    # r, theta = lambert_equal_area_conversion(takeoffs, azis)
    C2 = ax2.pcolormesh(
        azis, takeoffs_plot, np.sign(rad_P), shading="nearest", cmap="binary"
    )
    plt.colorbar(C2, ax=ax2)
    # ax1.plot(azis, np.ones((361,))*deg2rad(45))
    ax2.grid(False)
    ax2.set_rticks([])
    # ax2.set_xticks([])
    ax2.set_title("b) Moment Tensor", fontsize=12)
    # Third Panel SV radiation pattern
    ax3 = axs[1, 0]
    ax3.set_theta_zero_location("N")
    ax3.set_theta_direction("clockwise")

    C3 = ax3.pcolormesh(
        azis,
        takeoffs_plot,
        rad_SV,
        shading="nearest",
        cmap="PuOr",
        vmin=-1 / mean_S,
        vmax=1 / mean_S,
    )
    ax3.contour(azis, takeoffs_plot, np.abs(rad_SV), levels=[1], colors="black")

    ax3.set_rticks([])
    ax3.set_title("c) SV Radiation Pattern", fontsize=12)
    plt.colorbar(C3, ax=ax3)

    #   Fourth Panel SH radiation pattern
    ax4 = axs[1, 1]
    ax4.set_theta_zero_location("N")
    ax4.set_theta_direction("clockwise")
    # ax1.grid(False)
    C4 = ax4.pcolormesh(
        azis,
        takeoffs_plot,
        rad_SH,
        shading="nearest",
        cmap="PuOr",
        vmin=-1 / mean_S,
        vmax=1 / mean_S,
    )
    ax4.contour(azis, takeoffs_plot, np.abs(rad_SH), levels=[1], colors="black")
    ax4.set_rticks([])
    ax4.set_title("d) SH Radiation Pattern", fontsize=12)
    plt.colorbar(C4, ax=ax4)

    if stations is not None:
        n = len(stations["name"])
        #  copy to preserve in/output station dict
        stations_out = stations.copy()
        # Add Stations
        stations_out["azis"] = np.zeros((n,))
        stations_out["takeoff_p"] = np.zeros((n,))
        stations_out["takeoff_s"] = np.zeros((n,))
        stations_out["P_radiation"] = np.zeros((n,))
        stations_out["S_radiation"] = np.zeros((n,))
        stations_out["SV_radiation"] = np.zeros((n,))
        stations_out["SH_radiation"] = np.zeros((n,))
        stations_out["Mw_diff_P"] = np.zeros((n,))
        stations_out["Mw_diff_S"] = np.zeros((n,))
        markers = ["o", "P", "X", "D", "s" "8", "h", "v"]

        for i in range(len(stations["name"])):
            print(i, stations["name"][i])
            _, stat_az = patterns.calc_station_dist_az_rad(
                stations["lat"][i], stations["lon"][i], fault
            )

            takeoff_p, takeoff_s = patterns.model_ray_param_ak135(
                stations["lat"][i],
                stations["lon"][i],
                fault["latitude"],
                fault["longitude"],
            )
            rp_stat, _, _ = patterns.calc_rad_patterns(
                rake=deg2rad(fault["rake"]),
                dip=deg2rad(fault["dip"]),
                strike=deg2rad(fault["strike"]),
                reciever_azi=stat_az,
                takeoff_angle=deg2rad(takeoff_p),
            )

            _, rsv_stat, rsh_stat = patterns.calc_rad_patterns(
                rake=deg2rad(fault["rake"]),
                dip=deg2rad(fault["dip"]),
                strike=deg2rad(fault["strike"]),
                reciever_azi=stat_az,
                takeoff_angle=deg2rad(takeoff_s),
            )
            stations_out["azis"][i] = np.round(stat_az, decimals=1)
            stations_out["takeoff_p"][i] = np.round(takeoff_p, decimals=1)
            stations_out["takeoff_s"][i] = np.round(takeoff_s, decimals=1)
            stations_out["P_radiation"][i] = np.round(
                np.abs(rp_stat) / mean_P, decimals=3
            )

            # To make a better comparison to the mean S radation pattern we need to combine SH and SV in the same way
            # sqrt(SV**2 + SH**2)

            stations_out["S_radiation"][i] = np.round(
                np.sqrt(rsv_stat**2 + rsh_stat**2) / mean_S, decimals=3
            )
            stations_out["SV_radiation"][i] = rsv_stat
            stations_out["SH_radiation"][i] = rsh_stat
            stations_out["Mw_diff_P"][i] = np.round(
                (2 / 3) * np.log10(stations_out["P_radiation"][i]), decimals=2
            )
            stations_out["Mw_diff_S"][i] = np.round(
                (2 / 3) * np.log10(stations_out["S_radiation"][i]), decimals=2
            )
            ax1.plot(
                stations_out["azis"][i],
                np.sqrt(2) * np.sin(deg2rad(stations_out["takeoff_p"][i] / 2)),
                markers[i],
                mfc="white",
                mec="black",
                markersize=8,
                label=stations["name"][i],
            )
            ax3.plot(
                stations_out["azis"][i],
                np.sqrt(2) * np.sin(deg2rad(stations_out["takeoff_s"][i] / 2)),
                markers[i],
                mfc="white",
                mec="black",
                markersize=8,
                label=stations_out["name"][i],
            )
            ax4.plot(
                stations_out["azis"][i],
                np.sqrt(2) * np.sin(deg2rad(stations_out["takeoff_s"][i] / 2)),
                markers[i],
                mfc="white",
                mec="black",
                markersize=8,
                label=stations_out["name"][i],
            )

    ax1.grid(False)
    ax3.grid(False)
    ax4.grid(False)
    # add figlegend
    handles, labels = ax1.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    title1 = f'Modelled event: strike {fault["strike"]:3.0f}°, dip {fault["dip"]:3.0f}°, rake {fault["rake"]:3.0f}°.'
    plt.suptitle(f"{title1}")
    if out:
        figure.savefig(out, dpi=400)

    plt.show()

    return stations_out


def model_p_radiation(fault, stations, takeoff=np.pi / 2, out="None"):
    """
    Calculates geometric radiation pattern for a given fault.
    Radation patterns are normalised by the mean

    If station_azi provided then will add a line for that station
    """
    azimuths = np.linspace(0, 2 * np.pi, 361)
    # i.e., 1 point per degree
    p_rad, sv_rad, sh_rad = patterns.calc_rad_patterns(
        rake=np.deg2rad(fault["rake"]),
        dip=np.deg2rad(fault["dip"]),
        strike=np.deg2rad(fault["strike"]),
        reciever_azi=azimuths,
        takeoff_angle=takeoff,
    )
    fig = plt.figure(figsize=[6, 6])

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ml_assume = np.mean(np.abs(p_rad))
    p_rad = p_rad / ml_assume
    ax.plot(azimuths[p_rad > 0], p_rad[p_rad > 0], color="tab:blue")
    ax.plot(azimuths[p_rad < 0], np.abs(p_rad[p_rad < 0]), color="tab:red")
    ax.plot(azimuths, np.ones(azimuths.shape), "k--")
    for i, azi in enumerate(stations["azis"]):
        print(np.rad2deg(azi))
        ax.plot([azi, azi], [0, np.max(np.abs(p_rad))], "k-", label=stations["name"][i])

        p_rad_arr, _, _ = patterns.calc_rad_patterns(
            rake=np.deg2rad(fault["rake"]),
            dip=np.deg2rad(fault["dip"]),
            strike=np.deg2rad(fault["strike"]),
            reciever_azi=azi,
            takeoff_angle=takeoff,
        )
        p_rad_arr = np.abs(p_rad_arr / ml_assume)
        bbox = dict(boxstyle="round", fc="blanchedalmond", ec="orange", alpha=0.5)
        text = f'{stations["name"][i]} \n' f"$R_P = $ {p_rad_arr:4.2f}"
        if i == 0:
            ax.text(
                azi + np.deg2rad(5),
                0.9 * np.max(np.abs(p_rad)),
                text,
                fontsize=10,
                bbox=bbox,
            )
        elif i == 1:
            ax.text(
                azi - np.deg2rad(5),
                0.9 * np.max(np.abs(p_rad)),
                text,
                fontsize=10,
                bbox=bbox,
            )
        elif i == 2:
            ax.text(
                azi + np.deg2rad(15),
                0.9 * np.max(np.abs(p_rad)),
                text,
                fontsize=10,
                bbox=bbox,
            )
    title1 = f'Strike {fault["strike"]:3.0f}°, dip {fault["dip"]:3.0f}°, rake {fault["rake"]:3.0f}°.'
    ax.set_title(f"{title1}")
    if out:
        fig.savefig(f"figs/{out}", format="png", dpi=500)

    plt.show()


def model_s_radiation(fault, stations, takeoff=np.pi / 2, out="None"):
    """
    Calculates geometric radiation pattern for a given fault.
    Radation patterns are normalised by the mean

    If station_azi provided then will add a line for that station
    """
    azimuths = np.linspace(0, 2 * np.pi, 361)  # i.e., 1 point per degree
    _, sv_rad, sh_rad = patterns.calc_rad_patterns(
        rake=np.deg2rad(fault["rake"]),
        dip=np.deg2rad(fault["dip"]),
        strike=np.deg2rad(fault["strike"]),
        reciever_azi=azimuths,
        takeoff_angle=takeoff,
    )
    fig = plt.figure(figsize=[6, 6])

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")

    ax.plot(azimuths[sv_rad > 0], sv_rad[sv_rad > 0], color="tab:blue")
    ax.plot(azimuths[sv_rad < 0], np.abs(sv_rad[sv_rad < 0]), color="tab:red")

    ax.plot(
        azimuths[sh_rad > 0], sh_rad[sh_rad > 0], color="tab:blue", linestyle="dashed"
    )
    ax.plot(
        azimuths[sh_rad < 0],
        np.abs(sh_rad[sh_rad < 0]),
        color="tab:red",
        linestyle="dashed",
    )

    for i, azi in enumerate(stations["azis"]):
        print(f'Array: {stations["name"][i]}.' + f"Azi : {np.rad2deg(azi):4.0f}")
        ax.plot(
            [azi, azi],
            [0, np.max([np.abs(sv_rad), np.abs(sh_rad)])],
            "k-",
            label=stations["name"][i],
        )

    title1 = f'Strike {fault["strike"]:3.0f}°, dip {fault["dip"]:3.0f}°, rake {fault["rake"]:3.0f}°.'
    ax.set_title(f"{title1}")
    if out:
        fig.savefig(f"figs/{out}", format="png", dpi=500)

    plt.show()

    return fig
