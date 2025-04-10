import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

gdf_file = "file.gdf"
raw = mne.io.read_raw_gdf(gdf_file, preload=True)
raw.filter(1., 40., fir_design="firwin")

montage = mne.channels.make_standard_montage("standard_1020")
positions = montage.get_positions()
new_montage = mne.channels.make_dig_montage(ch_pos=positions["ch_pos"], coord_frame="head")
raw.set_montage(new_montage)

raw = mne.preprocessing.compute_current_source_density(raw)

fig = raw.plot_sensors(kind="topomap", show_names=True, sphere=(0, 0, 0, 0.14))
plt.title("Check This Layout")
plt.show()

events, _ = mne.events_from_annotations(raw)

unique_event_ids = np.unique(events[:, 2])
print("Available Event IDs in the GDF file:", unique_event_ids)

event1 = int(input("Enter the first event ID to analyze: "))
event2 = int(input("Enter the second event ID to analyze: "))

event_id = {
    "event1": event1,
    "event2": event2
}

epochs = mne.Epochs(raw, events, event_id=event_id,
                     tmin=0.0, tmax=2.0, baseline=None, preload=True, event_repeated='drop')

def bandpower(epochs, fmin, fmax):
    spectrum = epochs.compute_psd(method="welch")
    psds, freqs = spectrum.get_data(return_freqs=True)  
    idx = np.logical_and(freqs >= fmin, freqs <= fmax) 
    return psds[:, :, idx].mean(axis=2) 

bp_alpha_event1 = bandpower(epochs["event1"], 8, 12)
bp_alpha_event2 = bandpower(epochs["event2"], 8, 12)
bp_beta_event1 = bandpower(epochs["event1"], 13, 30)
bp_beta_event2 = bandpower(epochs["event2"], 13, 30)

def compute_r_squared(class1, class2):
    n_channels = class1.shape[1]
    r2 = np.zeros(n_channels)
    for ch in range(n_channels):
        t, _ = ttest_ind(class1[:, ch], class2[:, ch])
        n1, n2 = len(class1), len(class2)
        r2[ch] = (t**2) / (t**2 + (n1 + n2 - 2))
    return r2

r2_alpha = compute_r_squared(bp_alpha_event1, bp_alpha_event2)
r2_beta = compute_r_squared(bp_beta_event1, bp_beta_event2)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
im_alpha, _ = mne.viz.plot_topomap(r2_alpha, epochs.info, axes=axes[0], show=False, sphere=(0, 0, 0, 0.14),
                                   cmap="RdBu_r")
im_beta, _ = mne.viz.plot_topomap(r2_beta, epochs.info, axes=axes[1], show=False, sphere=(0, 0, 0, 0.14),
                                  cmap="RdBu_r")
axes[0].set_title(f"R² - Alpha (8–12 Hz) [{event1} vs {event2}]")
axes[1].set_title(f"R² - Beta (13–30 Hz) [{event1} vs {event2}]")

# Add color bars to the plots with color scale adjustment
cbar_alpha = fig.colorbar(im_alpha, ax=axes[0], orientation="vertical")
cbar_alpha.set_label("R² Value")
cbar_beta = fig.colorbar(im_beta, ax=axes[1], orientation="vertical")
cbar_beta.set_label("R² Value")

plt.tight_layout()
plt.show()


channel_names = epochs.ch_names
x = np.arange(len(channel_names))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, r2_alpha, width, label="Alpha", color="blue")
ax.bar(x + width/2, r2_beta, width, label="Beta", color="red")
ax.set_ylabel("R²")
ax.set_title(f"R² per Channel ({event1} vs {event2})")
ax.set_xticks(x)
ax.set_xticklabels(channel_names, rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
