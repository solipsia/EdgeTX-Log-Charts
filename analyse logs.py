import matplotlib.pyplot as plt  # pip install matplotlib
import pandas as pd  # pip install pandas
import numpy as np
from matplotlib.collections import LineCollection

# Enable interactive mode
plt.ion()

# Read the CSV file (update the path if needed)
df = pd.read_csv(r"X:\Drone NAS\14 June 2025\7 gem-2025-06-14-112046.csv")

# Optionally, if "Time" is not numeric, convert it to datetime:
# df["Time"] = pd.to_datetime(df["Time"])

# Define tick style settings for consistency
tkw = dict(size=4, width=1.5)

def make_patch_spines_invisible(ax):
    """Helper function to hide extra frame patches for additional y-axes."""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# ----------------------------
# Chart 1: ELRS Link Quality Chart
fig1, host = plt.subplots(figsize=(10, 6))
par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()

par2.spines["right"].set_position(("axes", 1.1))
par3.spines["right"].set_position(("axes", 1.2))
make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)
make_patch_spines_invisible(par3)
par3.spines["right"].set_visible(True)

host.plot(df["Time"], df["RQly(%)"], "b-", label="RQly(%)")
par1.plot(df["Time"], df["1RSS(dB)"], "r-", label="1RSS(dB)")
par2.plot(df["Time"], df["2RSS(dB)"], "g-", label="2RSS(dB)")
par3.plot(df["Time"], df["RSNR(dB)"], "m-", label="RSNR(dB)")

host.set_xlabel("Time")
host.set_ylabel("RQly(%)")
par1.set_ylabel("1RSS(dB)")
par2.set_ylabel("2RSS(dB)")
par3.set_ylabel("RSNR(dB)")

host.yaxis.label.set_color("b")
par1.yaxis.label.set_color("r")
par2.yaxis.label.set_color("g")
par3.yaxis.label.set_color("m")

host.tick_params(axis='y', colors="b", **tkw)
par1.tick_params(axis='y', colors="r", **tkw)
par2.tick_params(axis='y', colors="g", **tkw)
par3.tick_params(axis='y', colors="m", **tkw)
host.tick_params(axis='x', **tkw)

host.grid(False)
par1.grid(False)
par2.grid(False)
par3.grid(False)

lines = host.get_lines()  # get all line objects
host.legend(lines, [l.get_label() for l in lines])
plt.title("ELRS Link Quality Chart")
plt.tight_layout()
plt.show(block=False)

# ----------------------------
# Chart 2: Battery Chart
fig2, host = plt.subplots(figsize=(10, 6))
par1 = host.twinx()
par2 = host.twinx()

par2.spines["right"].set_position(("axes", 1.1))
make_patch_spines_invisible(par2)
par2.spines["right"].set_visible(True)

host.plot(df["Time"], df["Curr(A)"], "b-", label="Curr(A)")
par1.plot(df["Time"], df["RxBt(V)"], "r-", label="RxBt(V)")
par2.plot(df["Time"], df["Capa(mAh)"], "g-", label="Capa(mAh)")

host.set_xlabel("Time")
host.set_ylabel("Curr(A)")
par1.set_ylabel("RxBt(V)")
par2.set_ylabel("Capa(mAh)")

host.yaxis.label.set_color("b")
par1.yaxis.label.set_color("r")
par2.yaxis.label.set_color("g")

host.tick_params(axis='y', colors="b", **tkw)
par1.tick_params(axis='y', colors="r", **tkw)
par2.tick_params(axis='y', colors="g", **tkw)
host.tick_params(axis='x', **tkw)

host.grid(False)
par1.grid(False)
par2.grid(False)

lines = host.get_lines()
host.legend(lines, [l.get_label() for l in lines])
plt.title("Battery Chart")
plt.tight_layout()
plt.show(block=False)

# ----------------------------
# Chart 3: GPS Chart
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(df["Time"], df["Sats"], "b-")
ax.set_xlabel("Time")
ax.set_ylabel("Sats")
ax.grid(False)
plt.title("Sattelite Chart")
plt.tight_layout()
plt.show(block=False)

# ----------------------------
# Chart 4: Motion Chart
fig4, host = plt.subplots(figsize=(10, 6))
par1 = host.twinx()

host.plot(df["Time"], df["Alt(m)"], "b-", label="Alt(m)")
par1.plot(df["Time"], df["GSpd(kmh)"], "r-", label="GSpd(kmh)")

host.set_xlabel("Time")
host.set_ylabel("Alt(m)")
par1.set_ylabel("GSpd(kmh)")

host.yaxis.label.set_color("b")
par1.yaxis.label.set_color("r")

host.tick_params(axis='y', colors="b", **tkw)
par1.tick_params(axis='y', colors="r", **tkw)
host.tick_params(axis='x', **tkw)

host.grid(False)
par1.grid(False)

lines = host.get_lines()
host.legend(lines, [l.get_label() for l in lines])
plt.title("Motion Chart")
plt.tight_layout()
plt.show(block=False)

# ----------------------------
# Chart 5: Map (Drone Motion) with RSNR Color Gradient
# Extract GPS coordinates. If "GPS" is available, split into Latitude and Longitude.
if "GPS" in df.columns:
    df["Latitude"], df["Longitude"] = zip(*df["GPS"].apply(lambda x: (float(x.split()[0]), float(x.split()[1]))))
elif "Lat" in df.columns and "Lon" in df.columns:
    df["Latitude"] = df["Lat"]
    df["Longitude"] = df["Lon"]
else:
    print("No GPS coordinate data found in the CSV file.")
    df["Latitude"] = None
    df["Longitude"] = None

fig5, ax = plt.subplots(figsize=(10, 6))

# Ensure there is valid GPS data
if df["Latitude"].notnull().all() and df["Longitude"].notnull().all():
    # Create an array of (longitude, latitude) points.
    x = df["Longitude"].values
    y = df["Latitude"].values
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Match RSNR data using time (assuming each row's RSNR corresponds to the GPS row).
    if "RSNR(dB)" in df.columns:
        # Compute average RSNR for each segment (using adjacent points).
        rsnr = df["RSNR(dB)"].values
        noise_seg = (rsnr[:-1] + rsnr[1:]) / 2.0
    else:
        print("RSNR(dB) data not found. Defaulting to zero values.")
        noise_seg = np.zeros(len(segments))
    
    # Create a LineCollection with the desired colormap.
    # The "RdYlGn" colormap maps low values to red and high values to green.
    norm = plt.Normalize(rsnr.min(), rsnr.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
    lc.set_array(noise_seg)
    lc.set_linewidth(2)
    line = ax.add_collection(lc)

    # Optionally, add a colorbar to show RSNR scale.
    plt.colorbar(lc, ax=ax, label='RSNR(dB)')
else:
    # Fallback to a simple plot if GPS data is missing.
    ax.plot(df["Longitude"], df["Latitude"], "bo-", markersize=3, linewidth=1)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Drone Motion Map")
ax.grid(False)
ax.autoscale()  # Adjust the view limits to the data
plt.tight_layout()
plt.show(block=False)

# Prevent script from closing immediately so you can view all windows
input("Press Enter to exit and close all charts...")
