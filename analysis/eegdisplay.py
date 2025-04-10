import mne

gdf_file_path = "file.gdf"
raw = mne.io.read_raw_gdf(gdf_file_path, preload=True)
raw.plot(title="Raw GDF Data")

events, event_id = mne.events_from_annotations(raw)
print("Events:\n", events)
print("Event IDs:\n", event_id)


sfreq = raw.info['sfreq']
event_times = events[:, 0] / sfreq
for i, (event, time) in enumerate(zip(events, event_times)):
    print(f"Event {i + 1}: ID={event[2]}, Time={time:.2f} seconds")

mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_id)
