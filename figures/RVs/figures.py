from __future__ import print_function
import astropy
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.time import Time
import pandas as pd
import numpy as np
import glob
from datetime import datetime, timedelta
from brokenaxes import brokenaxes
import h5py
from astropy.io import fits
import matplotlib.dates as mdates

def compute_rms_errors(bjd, rv, rv_err, threshold=0.4):
    """
    For each nightly visit, fit a line (mean + slope), compute RMS of residuals,
    and return new per-observation errors = sqrt(RMS^2 + formal_err^2).
    threshold=0.4 days separates nights (~9.6 hours gap).
    """
    groups = group_tracks(bjd, threshold=threshold)
    new_err = np.zeros(len(rv))
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        bjd_g = bjd[idx]
        rv_g  = rv[idx]
        err_g = rv_err[idx]
        if len(idx) < 3:
            # not enough points to fit a line, just use formal errors
            new_err[idx] = err_g
            continue
        # weighted line fit
        w = 1.0 / err_g**2
        coeffs = np.polyfit(bjd_g - np.mean(bjd_g), rv_g, deg=1, w=w)
        residuals = rv_g - np.polyval(coeffs, bjd_g - np.mean(bjd_g))
        rms = np.sqrt(np.mean(residuals**2))
        new_err[idx] = np.sqrt(rms**2 + err_g**2)
    return new_err

def jld2_read(jld2_file, variable, index):
    array = jld2_file[variable[index]][()]
    array = np.array(array)
    return array

def jd2datetime(times):
    return np.array([Time(time,format="jd",scale="utc").datetime for time in times])

def group_tracks(bjds,threshold=0.05,plot=False):
    """
    
    INPUT:
        bjds - bjd times
        threshold - the threshold to define a new track
                    assumes a start of a new track if it is more than threshold apart
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(bjds))
    diff = np.diff(bjds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(bjds))
        for i in x:
            plt.plot(i,bjds[i],marker='o',color=cc[groups[i]])
    return groups

def weighted_average(x,e):
    """
    Calculate weigted average

    INPUT:
        x
        e

    OUTPUT:
        xx: weighted average of x
        ee: associated error
    """
    xx, ww = np.average(x,weights=e**(-2.),returned=True)
    return xx, np.sqrt(1./ww)

def bin_rvs_by_track(bjd,RV,RV_err,threshold):
    """

    INPUT:
        bjd
        RV
        RV_err

    OUTPUT:

    """
    track_groups = group_tracks(bjd,threshold=threshold,plot=False)
    date = [str(i)[0:10] for i in jd2datetime(bjd)]
    df = pd.DataFrame(zip(date,track_groups,bjd,RV,RV_err),
                      columns=['date','track_groups','bjd','RV','RV_err'])
    g = df.groupby(['track_groups'])
    ngroups = len(g.groups)

    # track_bins
    nbjd, nRV, nRV_err = np.zeros(ngroups),np.zeros(ngroups),np.zeros(ngroups)
    for i, (source, idx) in enumerate(g.groups.items()):
        cut = df.loc[idx]
        #nRV[i], nRV_err[i] = wsem(cut.RV.values,cut.RV_err)
        nRV[i], nRV_err[i] = weighted_average(cut.RV.values,cut.RV_err.values)
        nbjd[i] = np.mean(cut.bjd.values)
    return nbjd, nRV, nRV_err

import numpy as np

def bin_rvs_by_day(bjd, rv, rv_err):
    """
    Bin radial velocities by calendar day.

    For each unique day in the BJD array, computes the weighted mean RV
    and propagated uncertainty across all observations that night.

    Parameters
    ----------
    bjd    : array-like, BJD timestamps (days)
    rv     : array-like, radial velocities (m/s or km/s)
    rv_err : array-like, RV uncertainties (same units as rv)

    Returns
    -------
    bin_bjd : np.ndarray, weighted mean BJD per bin
    bin_rv  : np.ndarray, weighted mean RV per bin
    bin_err : np.ndarray, propagated uncertainty per bin
    """
    bjd    = np.asarray(bjd,    dtype=float)
    rv     = np.asarray(rv,     dtype=float)
    rv_err = np.asarray(rv_err, dtype=float)

    # integer day index for each observation (floor of BJD)
    day_idx = np.floor(bjd).astype(int)
    unique_days = np.unique(day_idx)

    bin_bjd = np.zeros(len(unique_days))
    bin_rv  = np.zeros(len(unique_days))
    bin_err = np.zeros(len(unique_days))

    for k, day in enumerate(unique_days):
        mask = day_idx == day
        w    = 1.0 / rv_err[mask]**2          # weights = 1/σ²

        bin_bjd[k] = np.average(bjd[mask], weights=w)
        bin_rv[k]  = np.average(rv[mask],  weights=w)
        bin_err[k] = 1.0 / np.sqrt(np.sum(w))  # propagated uncertainty

    return bin_bjd, bin_rv, bin_err


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,6), sharey=True, dpi=600, gridspec_kw={'wspace': 0.02})

df_hpf = pd.read_csv('../../data/HPF_rv_unbin.csv')
rvs = df_hpf["rv"]
rv_err = df_hpf["e_rv"]
bjd = df_hpf["bjd"]
mask = rv_err <= 10
rvs = rvs[mask]
rv_err = rv_err[mask]
bjd = bjd[mask]

# Update HPF errors to reflect nightly scatter
rv_err = compute_rms_errors(
    bjd.values,
    rvs.values,
    rv_err.values
)

mask_dates = ~(((Time(bjd, format='jd')).to_datetime() >= datetime(2026, 1, 1)) & ((Time(bjd, format='jd')).to_datetime() <= datetime(2026, 1, 13)))
nbjd, nRV, nRV_err = bin_rvs_by_track(bjd[mask_dates],rvs[mask_dates],rv_err[mask_dates],0.5) 
label2="HPF Unbinned: RMS={:0.2f}m/s, Median(errorbar)={:0.2f}m/s".format(np.sqrt(np.nanmean((rvs[mask_dates] - np.nanmean(rvs[mask_dates]))**2)), np.nanmedian(rv_err[mask_dates]))
label3="HPF Binned: RMS={:0.2f}m/s, Median(errorbar)={:0.2f}m/s".format(np.sqrt(np.nanmean((nRV - np.nanmean(nRV))**2)), np.nanmedian(nRV_err))

nbjd, nRV, nRV_err = bin_rvs_by_track(bjd,rvs,rv_err,0.5) 
ax1.errorbar((Time(bjd, format='jd')).to_datetime(),rvs,rv_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "pink", label = label2, zorder=1)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='red', label = label3, zorder=3)
ax1.set_xlim(datetime(2025, 10, 1), datetime(2025, 11, 1))
ax2.errorbar((Time(bjd, format='jd')).to_datetime(),rvs,rv_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "pink", zorder=1)
ax2.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='red', label = label3, zorder=3)                
ax2.set_xlim(datetime(2025, 12, 1), datetime(2025, 12, 15))
ax3.errorbar((Time(bjd, format='jd')).to_datetime(),rvs,rv_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "pink", zorder=1)
ax3.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='red', label = label3, zorder=3)
ax3.set_xlim(datetime(2026, 1, 1), datetime(2026, 2, 1))

df_neid = pd.read_csv('../../data/NEID_rv_unbin.csv')
rvs_neid = df_neid["rv"][12:-1]
rv_err_neid = df_neid["e_rv"][12:-1]
bjd_neid = df_neid["bjd"][12:-1]

# Update NEID errors to reflect nightly scatter
rv_err_neid = compute_rms_errors(
    bjd_neid.values,
    rvs_neid.values,
    rv_err_neid.values
)

mask_dates = ~(((Time(bjd_neid, format='jd')).to_datetime() >= datetime(2026, 1, 9)) & ((Time(bjd_neid, format='jd')).to_datetime() <= datetime(2026, 1, 13)))
nbjd, nRV, nRV_err = bin_rvs_by_day(bjd_neid[mask_dates],rvs_neid[mask_dates],rv_err_neid[mask_dates]) 
label4="NEID Unbinned: RMS={:0.2f}m/s, Median(errorbar)={:0.2f}m/s".format(np.sqrt(np.nanmean((rvs_neid[mask_dates] - np.nanmean(rvs_neid[mask_dates]))**2)), np.nanmedian(rv_err_neid[mask_dates]))
label5="NEID Binned: RMS={:0.2f}m/s, Median(errorbar)={:0.2f}m/s".format(np.sqrt(np.nanmean((nRV - np.nanmean(nRV))**2)), np.nanmedian(nRV_err))

nbjd, nRV, nRV_err = bin_rvs_by_day(bjd_neid,rvs_neid,rv_err_neid) 
ax1.errorbar((Time(bjd_neid, format='jd')).to_datetime(),rvs_neid,rv_err_neid,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "blue", label = label4, zorder=2)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='green', label = label5, zorder=4)
ax1.set_xlim(datetime(2025, 10, 1), datetime(2025, 11, 1))
ax2.errorbar((Time(bjd_neid, format='jd')).to_datetime(),rvs_neid,rv_err_neid,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "blue", zorder=2)
ax2.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='green', zorder=5)                
ax2.set_xlim(datetime(2025, 12, 1), datetime(2025, 12, 15))
ax3.errorbar((Time(bjd_neid, format='jd')).to_datetime(),rvs_neid,rv_err_neid,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "blue", zorder=2)
ax3.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='green', zorder=5)
ax3.set_xlim(datetime(2026, 1, 1), datetime(2026, 2, 1))

ax1.tick_params(axis="y",pad=3, labelsize=12)
ax1.tick_params(axis="x",pad=3, labelsize=12)
ax2.tick_params(axis="x",pad=3, labelsize=12)
ax3.tick_params(axis="x",pad=3, labelsize=12)
ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=2))
ax1.set_ylabel("RV [m/s]",fontsize=16)
ax2.set_xlabel("Time [UT]",fontsize=16)
handles, labels = ax1.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='lower left', fontsize=12,
                 bbox_to_anchor=(0.12, 0.11))  # fine-tune position as needed
leg.set_zorder(10)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax2.tick_params(left=False, which='both')
ax3.tick_params(left=False, which='both')
d = .85
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=15,
              linestyle='none', color='k', mec='k', mew=1, clip_on=False)

# Between ax1 and ax2
ax1.plot([1, 1], [1, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)

# Between ax2 and ax3
ax2.plot([1, 1], [1, 0], transform=ax2.transAxes, **kwargs)  # was ax1.transAxes
ax3.plot([0, 0], [0, 1], transform=ax3.transAxes, **kwargs)  # was ax2.transAxes

start_date = pd.Timestamp('2026-01-10 07:28')
end_date = pd.Timestamp('2026-01-10 11:13')
ax3.axvspan(start_date, end_date, color='gray', alpha=0.4)
plt.tight_layout()
plt.savefig("figure2.pdf")

fig, ax1 = plt.subplots(figsize=(10,6), dpi=600)

DV_iso_2P_file = h5py.File("data/projected_gpu_SH_DV.jld2", "r")
time_extended  = DV_iso_2P_file["time"][()]
DV_iso_2P = DV_iso_2P_file["RV_list_no_cb"][()]

SH_SB_file = h5py.File("data/projected_gpu_SH_SB.jld2", "r")
SH_SB = SH_SB_file["RV_list_no_cb"][()]

CB_SB_file = h5py.File("data/projected_gpu_CB_SB.jld2", "r")
CB_SB = CB_SB_file["RV_list_no_cb"][()]

model_time = []
for i in time_extended:
    dt = datetime.strptime(i.decode("utf-8"), "%Y-%m-%dT%H:%M:%S.%f")
    model_time.append((Time(dt)).jd)

DV_iso_2P_arr = jld2_read(DV_iso_2P_file, DV_iso_2P, 0)
SH_SB_arr = jld2_read(SH_SB_file, SH_SB, 0)
CB_SB_arr = jld2_read(CB_SB_file, CB_SB, 0)

plt.plot((Time(model_time, format='jd')).to_datetime(), DV_iso_2P_arr, label = "DV97 SH", color = "k", zorder=5) 
plt.plot((Time(model_time, format='jd')).to_datetime(), SH_SB_arr, label = "SB03 SH", color = 'purple', zorder=6) 
plt.plot((Time(model_time, format='jd')).to_datetime(), CB_SB_arr, label = "SB03 CB", color = 'orange', zorder=7) 

nbjd, nRV_hpf, nRV_err_hpf = bin_rvs_by_track(bjd,rvs,rv_err,0.5) 
data_dt_binned  = Time(nbjd, format='jd').to_datetime()
ax1.errorbar((Time(bjd, format='jd')).to_datetime(),rvs,rv_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "pink", label = "HPF Unbinned", zorder=1)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV_hpf,nRV_err_hpf,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='red', label = "HPF Binned", zorder=3)

nbjd, nRV, nRV_err = bin_rvs_by_day(bjd_neid,rvs_neid,rv_err_neid) 
data_dt_neid_binned  = Time(nbjd, format='jd').to_datetime()
ax1.errorbar((Time(bjd_neid, format='jd')).to_datetime(),rvs_neid,rv_err_neid,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "blue", label = "NEID Unbinned", zorder=2)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='green', label = "NEID Binned", zorder=4)
ax1.set_xlim(datetime(2026, 1, 5), datetime(2026, 1, 27))

zoom_regions = [
        # (x_start, x_end, y_start, y_end, inset_position)
        (datetime(2026,1,10,5), datetime(2026,1,10,8), -40, -8,  [0.35, 0.8, 0.18, 0.18]),
        (datetime(2026,1,11,8), datetime(2026,1,11,13), -30, 30, [0.58, 0.8, 0.18, 0.18]),
        (datetime(2026,1,12,4,30), datetime(2026,1,12,9,30), -12, 22,[0.8, 0.8, 0.18, 0.18]),
]
model_dt = Time(model_time, format='jd').to_datetime()
data_dt_neid  = Time(bjd_neid, format='jd').to_datetime()
data_dt  = Time(bjd, format='jd').to_datetime()

for j, (x1, x2, y1, y2, pos) in enumerate(zoom_regions):

    axins = ax1.inset_axes(pos)

    # --- plot SAME data as main plot ---
    axins.plot(model_dt, DV_iso_2P_arr, color = 'k', zorder=5)
    axins.plot(model_dt, SH_SB_arr, color = 'purple', zorder=6)
    axins.plot(model_dt, CB_SB_arr, color = 'orange', zorder=7)

    axins.errorbar(data_dt_neid, rvs_neid, rv_err_neid, fmt='h', markersize=4, color = "blue", zorder=1)
    axins.errorbar(data_dt, rvs, rv_err, fmt='h', markersize=4, color = "pink", zorder=2)
    axins.errorbar(data_dt_binned, nRV_hpf, nRV_err_hpf, fmt='h', marker="s", markersize=4, color = "red", zorder=3)
    axins.errorbar(data_dt_neid_binned, nRV, nRV_err, fmt='h', marker="s", markersize=4, color = "green", zorder=4)

    # --- apply zoom ---
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # axins.set_xticklabels([])
    axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%d %H"))
    # axins.set_yticklabels([])

ax1.tick_params(axis="y",pad=3, labelsize=12)
ax1.tick_params(axis="x",pad=3, labelsize=12)
ax1.set_ylabel("RV [m/s]",fontsize=16)
ax1.set_xlabel("Time [UT]",fontsize=16)
plt.legend(loc='lower right', fontsize=12)

start_date = pd.Timestamp('2026-01-10 07:28')
end_date = pd.Timestamp('2026-01-10 11:13')
ax1.axvspan(start_date, end_date, color='gray', alpha=0.4)
plt.tight_layout()
plt.ylim(-40,35)
plt.savefig("figure3.pdf")

fig, ax1 = plt.subplots(figsize=(10,6), dpi=600)

SH_SB_file = h5py.File("data/projected_gpu_SH_fit.jld2", "r")
time_extended  = DV_iso_2P_file["time"][()]
SH_SB = SH_SB_file["RV_list_no_cb"][()]

CB_SB_file = h5py.File("data/projected_gpu_CB_fit.jld2", "r")
CB_SB = CB_SB_file["RV_list_no_cb"][()]

model_time = []
for i in time_extended:
    dt = datetime.strptime(i.decode("utf-8"), "%Y-%m-%dT%H:%M:%S.%f")
    model_time.append((Time(dt)).jd)

SH_SB_arr = jld2_read(SH_SB_file, SH_SB, 30)
CB_SB_arr = jld2_read(CB_SB_file, CB_SB, 30)

plt.plot((Time(model_time, format='jd')).to_datetime(), SH_SB_arr, label = "SH", color = 'k', zorder=5) 
plt.plot((Time(model_time, format='jd')).to_datetime(), CB_SB_arr, label = "CB", color = 'orange', zorder=6) 

nbjd, nRV, nRV_err = bin_rvs_by_track(bjd,rvs,rv_err,0.5) 
ax1.errorbar((Time(bjd, format='jd')).to_datetime(),rvs,rv_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "pink", label = "HPF Unbinned", zorder=1)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='red', label = "HPF Binned", zorder=3)

nbjd, nRV, nRV_err = bin_rvs_by_day(bjd_neid,rvs_neid,rv_err_neid) 
ax1.errorbar((Time(bjd_neid, format='jd')).to_datetime(),rvs_neid,rv_err_neid,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color = "blue", label = "NEID Unbinned", zorder=2)
ax1.errorbar((Time(nbjd, format='jd')).to_datetime(),nRV,nRV_err,marker="s",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,color='green', label = "NEID Binned", zorder=4)
ax1.set_xlim(datetime(2026, 1, 5), datetime(2026, 1, 27))

zoom_regions = [
        # (x_start, x_end, y_start, y_end, inset_position)
        (datetime(2026,1,10,5), datetime(2026,1,10,8), -40, -8,  [0.35, 0.8, 0.18, 0.18]),
        (datetime(2026,1,11,8), datetime(2026,1,11,13), -30, 30, [0.58, 0.8, 0.18, 0.18]),
        (datetime(2026,1,12,4,30), datetime(2026,1,12,9,30), -12, 22,[0.8, 0.8, 0.18, 0.18]),
]
model_dt = Time(model_time, format='jd').to_datetime()
data_dt_neid  = Time(bjd_neid, format='jd').to_datetime()
data_dt  = Time(bjd, format='jd').to_datetime()

for j, (x1, x2, y1, y2, pos) in enumerate(zoom_regions):

    axins = ax1.inset_axes(pos)

    # --- plot SAME data as main plot ---
    axins.plot(model_dt, SH_SB_arr, color = 'k', zorder=5)
    axins.plot(model_dt, CB_SB_arr, color = 'orange', zorder=6)

    axins.errorbar(data_dt_neid, rvs_neid, rv_err_neid, fmt='h', markersize=4, color = "blue", zorder=1)
    axins.errorbar(data_dt, rvs, rv_err, fmt='h', markersize=4, color = "pink", zorder=2)
    axins.errorbar(data_dt_binned, nRV_hpf, nRV_err_hpf, fmt='h', marker="s", markersize=4, color = "red", zorder=3)
    axins.errorbar(data_dt_neid_binned, nRV, nRV_err, fmt='h', marker="s", markersize=4, color = "green", zorder=4)

    # --- apply zoom ---
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    # axins.set_xticklabels([])
    axins.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axins.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
    axins.xaxis.set_major_formatter(mdates.DateFormatter("%d %H"))
    # axins.set_yticklabels([])

ax1.tick_params(axis="y",pad=3, labelsize=12)
ax1.tick_params(axis="x",pad=3, labelsize=12)
ax1.set_ylabel("RV [m/s]",fontsize=16)
ax1.set_xlabel("Time [UT]",fontsize=16)
plt.legend(loc='lower right', fontsize=12)

start_date = pd.Timestamp('2026-01-10 07:28')
end_date = pd.Timestamp('2026-01-10 11:13')
ax1.axvspan(start_date, end_date, color='gray', alpha=0.4)
plt.tight_layout()
plt.ylim(-40,35)
plt.savefig("figure4.pdf")