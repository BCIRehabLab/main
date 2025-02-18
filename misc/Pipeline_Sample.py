#Sample MNE EEG Processing Pipeline
#Artur Agaronyan
#Catholic University of America
#June 5, 2024

import mne #load the mne module
import os #load file IO processing module
import matplotlib.pyplot as plt #load plotting module. Label it as “plt” to simplify our code-writing
import autoreject #load automated epoch rejection library

filename = '../data/data.set'
raw = mne.io.read_raw_eeglab(filename)
print(raw) #display the header information
print(raw.info) #display info about the EEG data, such as sampling frequency
raw.plot(duration=5, n_channels=30) #plot the raw dataset in an interactive viewer

rmChans  = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']
raw.drop_channels(rmChans)

raw.filter(l_freq=0.5, h_freq=None)

epochLowLim = -0.3;
epochHiLim  = 0.7;
cond1    = 'oddball_with_reponse'
cond2    = 'standard'
events_from_annot, event_dict = mne.events_from_annotations(raw)
epochs_all = mne.Epochs(raw, events_from_annot, tmin=epochLowLim, tmax=epochHiLim, event_id=event_dict, preload=True, event_repeated='drop')
epochs = epochs_all[cond1, cond2]


ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,n_jobs=1, verbose=True)
ar.fit(epochs[:20])  # fit on a few epochs to save time
epochs_ar, reject_log = ar.transform(epochs, return_log=True)


plt.plot(epochs_ar[0].times, epochs_ar[0].get_data()[0,1,:].transpose())
plt.plot(epochs_ar[1].times, epochs_ar[1].get_data()[0,1,:].transpose())
plt.legend([cond1,cond2])
plt.show()

fileout = os.path.splitext(filename)[0];
fileout_cond1 = fileout + '_cond1_mne.set'
fileout_cond2 = fileout + '_cond2_mne.set'
epochs_ar[cond1].export(fileout_cond1, overwrite=True)
epochs_ar[cond2].export(fileout_cond2, overwrite=True)
