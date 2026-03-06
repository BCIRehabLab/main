#Traditional Analysis Package v0.1
#Artur Aharonyan
#The Catholic University of America

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
from scipy import signal
import os
import warnings
warnings.filterwarnings('ignore')

class EEGAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EEG Analysis Tool - BrainProducts 128ch")
        self.root.geometry("1400x900")
        
        # Variables
        self.raw = None
        self.events = None
        self.event_id = None
        self.epochs = None
        self.current_file = None
        self.current_evoked = None
        self.is_resting_state = False
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # File selection
        ttk.Button(control_frame, text="Load VHDR File", 
                  command=self.load_vhdr).grid(row=0, column=0, padx=5)
        self.file_label = ttk.Label(control_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=5)
        
        # Analysis buttons frame - Row 1
        analysis_frame1 = ttk.Frame(control_frame)
        analysis_frame1.grid(row=1, column=0, columnspan=4, pady=5)
        
        ttk.Button(analysis_frame1, text="View Events", 
                  command=self.view_events).grid(row=0, column=0, padx=5)
        ttk.Button(analysis_frame1, text="Preprocess Data", 
                  command=self.preprocess_data).grid(row=0, column=1, padx=5)
        ttk.Button(analysis_frame1, text="Segment Data", 
                  command=self.segment_data).grid(row=0, column=2, padx=5)
        ttk.Button(analysis_frame1, text="Plot ERP/PSD", 
                  command=self.plot_analysis).grid(row=0, column=3, padx=5)
        
        # Analysis buttons frame - Row 2
        analysis_frame2 = ttk.Frame(control_frame)
        analysis_frame2.grid(row=2, column=0, columnspan=4, pady=5)
        
        ttk.Button(analysis_frame2, text="Show Montage", 
                  command=self.show_montage).grid(row=0, column=0, padx=5)
        ttk.Button(analysis_frame2, text="Topographic Maps", 
                  command=self.show_topomaps).grid(row=0, column=1, padx=5)
        ttk.Button(analysis_frame2, text="Channel Spectra", 
                  command=self.plot_channel_spectra).grid(row=0, column=2, padx=5)
        ttk.Button(analysis_frame2, text="Export Results", 
                  command=self.export_results).grid(row=0, column=3, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_text = ScrolledText(status_frame, height=8, width=100)
        self.status_text.grid(row=0, column=0)
        
        # Create a notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Tab 1: Main plots
        self.main_plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_plot_frame, text='Main Plots')
        
        # Tab 2: Montage visualization
        self.montage_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.montage_frame, text='Montage & Topography')
        
        # Create matplotlib figures for both tabs
        self.fig_main, self.ax_main = plt.subplots(figsize=(10, 6))
        self.canvas_main = FigureCanvasTkAgg(self.fig_main, master=self.main_plot_frame)
        self.canvas_main.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.fig_montage, self.ax_montage = plt.subplots(figsize=(10, 6))
        self.canvas_montage = FigureCanvasTkAgg(self.fig_montage, master=self.montage_frame)
        self.canvas_montage.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Toolbars for both tabs
        toolbar_main = NavigationToolbar2Tk(self.canvas_main, self.main_plot_frame)
        toolbar_main.update()
        
        toolbar_montage = NavigationToolbar2Tk(self.canvas_montage, self.montage_frame)
        toolbar_montage.update()
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
    def log_message(self, message):
        """Add message to status text"""
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.root.update()
        
    def clear_main_plot(self):
        """Clear the main plot"""
        self.ax_main.clear()
        
    def clear_montage_plot(self):
        """Clear the montage plot"""
        self.ax_montage.clear()
        
    def load_vhdr(self):
        """Load BrainProducts VHDR file"""
        filename = filedialog.askopenfilename(
            title="Select VHDR file",
            filetypes=[("BrainVision files", "*.vhdr"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                self.log_message(f"Loading file: {filename}")
                self.raw = mne.io.read_raw_brainvision(filename, preload=True)
                self.current_file = filename
                self.file_label.config(text=os.path.basename(filename))
                
                # Get the standard montage
                self.log_message("Getting montage: brainproducts-RNP-BA-128")
                montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')
                
                # Find channels that are in the montage
                montage_channels = set(montage.ch_names)
                raw_channels = set(self.raw.ch_names)
                channels_to_keep = [ch for ch in self.raw.ch_names if ch in montage_channels]
                channels_to_drop = [ch for ch in self.raw.ch_names if ch not in montage_channels]
                
                # Drop channels not in montage
                if channels_to_drop:
                    self.log_message(f"\nDropping {len(channels_to_drop)} channels not in montage:")
                    for ch in sorted(channels_to_drop):
                        self.log_message(f"  - {ch}")
                    self.raw.drop_channels(channels_to_drop)
                
                # Set the montage
                self.log_message(f"Setting montage for {len(self.raw.ch_names)} channels")
                self.raw.set_montage(montage)
                
                # Extract events
                events, event_dict = mne.events_from_annotations(self.raw)
                self.events = events
                self.event_id = event_dict
                
                # Check if this is resting state (only one event type)
                if len(event_dict) == 1:
                    self.is_resting_state = True
                    self.log_message("\n✓ Detected resting state data (single event type)")
                else:
                    self.is_resting_state = False
                    self.log_message(f"\n✓ Detected task data ({len(event_dict)} event types)")
                
                self.log_message(f"\nFile loaded successfully")
                self.log_message(f"Channels: {len(self.raw.ch_names)}")
                self.log_message(f"EEG channels: {len(mne.pick_types(self.raw.info, eeg=True))}")
                self.log_message(f"Time duration: {self.raw.times[-1]:.2f} seconds")
                self.log_message(f"Sampling frequency: {self.raw.info['sfreq']} Hz")
                
                # Display basic info in main plot
                self.plot_channel_info()
                
                # Show montage in montage tab
                self.show_montage()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
                self.log_message(f"Error: {str(e)}")
    
    def plot_channel_info(self):
        """Plot channel information"""
        self.clear_main_plot()
        
        try:
            # Get all channel names and types
            ch_names = self.raw.ch_names
            ch_types = self.raw.get_channel_types()
            
            # Create a summary plot
            self.ax_main.bar(['EEG', 'EOG', 'STIM', 'MISC'], 
                            [ch_types.count('eeg'), ch_types.count('eog'), 
                             ch_types.count('stim'), ch_types.count('misc')],
                            color=['blue', 'green', 'red', 'gray'], alpha=0.6)
            self.ax_main.set_ylabel('Number of Channels')
            self.ax_main.set_title('Channel Type Distribution')
            
            # Add text with file info
            info_text = f"Total channels: {len(self.raw.ch_names)}\n"
            info_text += f"EEG channels: {ch_types.count('eeg')}\n"
            info_text += f"Duration: {self.raw.times[-1]:.1f}s\n"
            info_text += f"Sampling rate: {self.raw.info['sfreq']}Hz\n"
            info_text += f"Mode: {'Resting State' if self.is_resting_state else 'Task ERP'}"
            
            self.ax_main.text(0.5, -0.2, info_text, transform=self.ax_main.transAxes,
                        ha='center', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            self.ax_main.text(0.5, 0.5, f"Channel info plot error:\n{str(e)}", 
                        transform=self.ax_main.transAxes, ha='center', va='center')
        
        self.canvas_main.draw()
    
    def show_montage(self):
        """Display the electrode montage"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.clear_montage_plot()
        
        try:
            # Switch to montage tab
            self.notebook.select(self.montage_frame)
            
            # Get the montage
            montage = self.raw.get_montage()
            
            if montage is None:
                self.ax_montage.text(0.5, 0.5, "No montage information available",
                                    transform=self.ax_montage.transAxes, ha='center', va='center')
                self.canvas_montage.draw()
                return
            
            # Plot the montage with all channels
            montage.plot(show_names=True, show=False, axes=self.ax_montage)
            self.ax_montage.set_title(f"Complete Electrode Montage - {len(montage.ch_names)} channels")
            
            self.canvas_montage.draw()
            self.log_message(f"\nMontage displayed with all {len(montage.ch_names)} channels")
            
        except Exception as e:
            self.ax_montage.text(0.5, 0.5, f"Montage plot error:\n{str(e)}",
                                transform=self.ax_montage.transAxes, ha='center', va='center')
            self.canvas_montage.draw()
            self.log_message(f"Error showing montage: {str(e)}")
    
    def segment_data(self):
        """Segment data based on event type (resting or task)"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        # Switch to main plot tab
        self.notebook.select(self.main_plot_frame)
        self.clear_main_plot()
        
        try:
            if self.is_resting_state:
                # For resting state: create continuous segments
                self.segment_resting_state()
            else:
                # For task data: create epochs around events
                self.create_epochs()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to segment data: {str(e)}")
            self.log_message(f"Error: {str(e)}")
    
    def segment_resting_state(self):
        """Create segments for resting state data"""
        self.log_message("\n=== SEGMENTING RESTING STATE DATA ===")
        
        # Get the single event type
        event_name = list(self.event_id.keys())[0]
        event_id = self.event_id[event_name]
        
        # Create 2-second segments (adjustable based on your needs)
        segment_duration = 2.0  # seconds
        n_samples_per_segment = int(segment_duration * self.raw.info['sfreq'])
        n_segments = len(self.raw.times) // n_samples_per_segment
        
        self.log_message(f"Creating {n_segments} segments of {segment_duration}s each")
        self.log_message(f"Event type: {event_name}")
        
        # Create artificial events for segmentation
        segment_starts = np.arange(0, n_segments * n_samples_per_segment, n_samples_per_segment)
        segment_events = np.array([[start, 0, event_id] for start in segment_starts])
        
        # Create epochs from segments
        tmin, tmax = 0, segment_duration
        self.epochs = mne.Epochs(self.raw, segment_events, event_id=event_id,
                                tmin=tmin, tmax=tmax, baseline=None,
                                preload=True, reject_by_annotation=False)
        
        self.log_message(f"Created {len(self.epochs)} resting state segments")
        
        # Show segment summary in plot
        self.ax_main.plot(self.raw.times[:10*n_samples_per_segment], 
                         self.raw.get_data(picks='eeg')[0, :10*n_samples_per_segment])
        
        # Mark segment boundaries
        for i in range(10):
            self.ax_main.axvline(x=i*segment_duration, color='r', linestyle='--', alpha=0.5)
        
        self.ax_main.set_xlabel('Time (s)')
        self.ax_main.set_ylabel('Amplitude (µV)')
        self.ax_main.set_title(f'Resting State - First 10 Segments (red lines = segment boundaries)')
        self.canvas_main.draw()
    
    def create_epochs(self):
        """Create epochs for task data"""
        # Get event dictionary
        events, event_dict = mne.events_from_annotations(self.raw)
        
        # Use events after preprocessing (if available)
        if hasattr(self, 'events') and self.events is not None:
            events = self.events
        
        # Check for duplicate event times
        event_times = events[:, 0]
        unique_times, counts = np.unique(event_times, return_counts=True)
        duplicates = unique_times[counts > 1]
        
        if len(duplicates) > 0:
            self.log_message(f"\nFound {len(duplicates)} time points with multiple events")
            self.log_message("Using event_repeated='merge' to handle duplicates")
        
        # Create epochs window
        tmin, tmax = -0.2, 0.8  # 200ms before to 800ms after event
        
        self.log_message(f"\n=== CREATING EPOCHS FOR ALL EVENTS ===")
        self.log_message(f"Epoch window: {tmin} to {tmax} seconds")
        self.log_message(f"Number of event types: {len(event_dict)}")
        
        # Create epochs for all events
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict,
                                tmin=tmin, tmax=tmax, baseline=(tmin, 0),
                                preload=True, reject_by_annotation=True,
                                event_repeated='merge')
        
        self.log_message(f"Created {len(self.epochs)} total epochs")
        
        # Show detailed counts by event type
        self.log_message("\nEpochs by event type:")
        for event_name, event_id in event_dict.items():
            if event_name in self.epochs.event_id:
                n_epochs = len(self.epochs[event_name])
                self.log_message(f"  {event_name}: {n_epochs}")
        
        # Plot epoch summary
        self.clear_main_plot()
        event_names = list(self.epochs.event_id.keys())
        event_counts = [len(self.epochs[event_name]) for event_name in event_names]
        
        bars = self.ax_main.bar(range(len(event_names)), event_counts, alpha=0.6)
        self.ax_main.set_xticks(range(len(event_names)))
        self.ax_main.set_xticklabels(event_names, rotation=45, ha='right')
        self.ax_main.set_ylabel('Number of Epochs')
        self.ax_main.set_title(f'All Events - Epoch Distribution (Total: {len(self.epochs)})')
        
        # Add count labels
        for bar, count in zip(bars, event_counts):
            height = bar.get_height()
            self.ax_main.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
        
        self.canvas_main.draw()
    
    def plot_analysis(self):
        """Plot appropriate analysis based on data type"""
        if self.epochs is None:
            messagebox.showwarning("Warning", "Please segment data first")
            return
        
        if self.is_resting_state:
            self.plot_resting_analysis()
        else:
            self.plot_erp_all_events()
    
    def plot_resting_analysis(self):
        """Plot resting state analysis (PSD, topomaps)"""
        # Switch to main plot tab
        self.notebook.select(self.main_plot_frame)
        self.clear_main_plot()
        
        try:
            # Create a figure with multiple subplots for resting analysis
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: PSD for all channels
            from scipy import signal as sig
            data, _ = self.raw[:, :]
            eeg_picks = mne.pick_types(self.raw.info, eeg=True)
            
            freqs, psd_all = sig.welch(data[eeg_picks], fs=self.raw.info['sfreq'], 
                                      nperseg=self.raw.info['sfreq']*2)
            
            # Plot all channels with transparency
            for psd in psd_all:
                axes[0, 0].semilogy(freqs[:100], psd[:100], 'b-', alpha=0.1, linewidth=0.5)
            
            # Plot average
            axes[0, 0].semilogy(freqs[:100], np.mean(psd_all, axis=0)[:100], 
                               'r-', linewidth=2, label='Average')
            axes[0, 0].set_xlabel('Frequency (Hz)')
            axes[0, 0].set_ylabel('Power (µV²/Hz)')
            axes[0, 0].set_title('Power Spectrum - All Channels')
            axes[0, 0].set_xlim(0, 50)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Alpha power topomap
            alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
            alpha_power = np.mean(psd_all[:, alpha_idx], axis=1)
            mne.viz.plot_topomap(alpha_power, self.raw.info, axes=axes[0, 1], 
                                show=False, contours=6)
            axes[0, 1].set_title('Alpha Power (8-13 Hz)')
            
            # Plot 3: Beta power topomap
            beta_idx = np.where((freqs >= 13) & (freqs <= 30))[0]
            beta_power = np.mean(psd_all[:, beta_idx], axis=1)
            mne.viz.plot_topomap(beta_power, self.raw.info, axes=axes[1, 0], 
                                show=False, contours=6)
            axes[1, 0].set_title('Beta Power (13-30 Hz)')
            
            # Plot 4: Theta power topomap
            theta_idx = np.where((freqs >= 4) & (freqs <= 8))[0]
            theta_power = np.mean(psd_all[:, theta_idx], axis=1)
            mne.viz.plot_topomap(theta_power, self.raw.info, axes=axes[1, 1], 
                                show=False, contours=6)
            axes[1, 1].set_title('Theta Power (4-8 Hz)')
            
            plt.suptitle('Resting State Analysis - All Channels', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Update canvas
            self.fig_main = fig
            self.canvas_main.figure = fig
            self.canvas_main.draw()
            
            self.log_message("\nResting state analysis complete - all channels displayed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot resting analysis: {str(e)}")
            self.log_message(f"Error: {str(e)}")
    
    def plot_erp_all_events(self):
        """Plot ERP for all event types using all channels"""
        # Switch to main plot tab
        self.notebook.select(self.main_plot_frame)
        self.clear_main_plot()
        
        try:
            # Get all event types
            event_types = list(self.epochs.event_id.keys())
            
            # Create a figure with subplots
            n_events = len(event_types)
            n_cols = min(3, n_events)
            n_rows = (n_events + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_events == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            # Plot ERP for each event type
            for idx, event_name in enumerate(event_types):
                if event_name in self.epochs.event_id:
                    evoked = self.epochs[event_name].average()
                    times = evoked.times * 1000  # Convert to ms
                    
                    # Plot all channels with low opacity
                    for ch_data in evoked.data:
                        axes[idx].plot(times, ch_data, 'b-', alpha=0.1, linewidth=0.5)
                    
                    # Plot global field power (GFP)
                    gfp = np.std(evoked.data, axis=0)
                    axes[idx].plot(times, gfp, 'r-', linewidth=2, label='GFP')
                    
                    axes[idx].set_xlabel('Time (ms)')
                    axes[idx].set_ylabel('Amplitude (µV)')
                    axes[idx].set_title(f'{event_name}\n(n={len(self.epochs[event_name])})')
                    axes[idx].axvline(x=0, color='k', linestyle='--', alpha=0.5)
                    axes[idx].axhline(y=0, color='k', linestyle='-', alpha=0.3)
                    axes[idx].legend()
                    axes[idx].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(event_types), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('ERP Analysis - All Events, All Channels', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Update canvas
            self.fig_main = fig
            self.canvas_main.figure = fig
            self.canvas_main.draw()
            
            self.log_message(f"\nERP plots created for all {len(event_types)} event types")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot ERP: {str(e)}")
            self.log_message(f"Error: {str(e)}")
    
    def show_topomaps(self):
        """Show comprehensive topographic maps"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.clear_montage_plot()
        
        try:
            # Switch to montage tab
            self.notebook.select(self.montage_frame)
            
            eeg_picks = mne.pick_types(self.raw.info, eeg=True)
            
            if len(eeg_picks) == 0:
                self.ax_montage.text(0.5, 0.5, "No EEG channels found",
                                    transform=self.ax_montage.transAxes, ha='center', va='center')
                self.canvas_montage.draw()
                return
            
            # Compute PSD for all channels
            from scipy import signal as sig
            data, _ = self.raw[:, :]
            freqs, psd = sig.welch(data[eeg_picks], fs=self.raw.info['sfreq'], 
                                  nperseg=self.raw.info['sfreq']*2)
            
            # Create comprehensive topomap figure
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            # Define frequency bands
            bands = [
                ('Delta (1-4 Hz)', (1, 4)),
                ('Theta (4-8 Hz)', (4, 8)),
                ('Alpha (8-13 Hz)', (8, 13)),
                ('Beta (13-30 Hz)', (13, 30)),
                ('Gamma (30-45 Hz)', (30, 45)),
                ('Broadband (1-45 Hz)', (1, 45))
            ]
            
            for idx, (band_name, (fmin, fmax)) in enumerate(bands):
                # Get frequency indices
                freq_idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                if len(freq_idx) > 0:
                    band_power = np.mean(psd[:, freq_idx], axis=1)
                    
                    # Normalize for better visualization
                    band_power = band_power / np.max(band_power)
                    
                    # Create topomap
                    im, _ = mne.viz.plot_topomap(band_power, self.raw.info, axes=axes[idx], 
                                                show=False, contours=6, extrapolate='local')
                    axes[idx].set_title(band_name, fontweight='bold')
                    plt.colorbar(im, ax=axes[idx], fraction=0.05, shrink=0.5)
            
            plt.suptitle('Complete Topographic Analysis - All Frequency Bands', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Update canvas
            self.fig_montage = fig
            self.canvas_montage.figure = fig
            self.canvas_montage.draw()
            
            self.log_message("\nComprehensive topographic maps displayed for all frequency bands")
            
        except Exception as e:
            self.ax_montage.clear()
            self.ax_montage.text(0.5, 0.5, f"Topomap error:\n{str(e)}",
                                transform=self.ax_montage.transAxes, ha='center', va='center')
            self.canvas_montage.draw()
            self.log_message(f"Error creating topomaps: {str(e)}")
    
    def plot_channel_spectra(self):
        """Plot spectra for all channels"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        self.clear_montage_plot()
        
        try:
            # Switch to montage tab
            self.notebook.select(self.montage_frame)
            
            eeg_picks = mne.pick_types(self.raw.info, eeg=True)
            
            if len(eeg_picks) == 0:
                self.ax_montage.text(0.5, 0.5, "No EEG channels found",
                                    transform=self.ax_montage.transAxes, ha='center', va='center')
                self.canvas_montage.draw()
                return
            
            # Compute PSD for all channels
            from scipy import signal as sig
            data, _ = self.raw[:, :]
            freqs, psd = sig.welch(data[eeg_picks], fs=self.raw.info['sfreq'], 
                                  nperseg=self.raw.info['sfreq']*2)
            
            # Create a figure with two subplots
            fig, (ax_topo, ax_spec) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Topographic map of alpha power
            alpha_idx = np.where((freqs >= 8) & (freqs <= 13))[0]
            alpha_power = np.mean(psd[:, alpha_idx], axis=1)
            
            mne.viz.plot_topomap(alpha_power, self.raw.info, axes=ax_topo, 
                                show=False, contours=6, extrapolate='local')
            ax_topo.set_title('Alpha Power Topography (8-13 Hz)', fontweight='bold')
            
            # Plot 2: All spectra
            for i in range(len(eeg_picks)):
                ax_spec.semilogy(freqs[:100], psd[i, :100], 
                               alpha=0.3, linewidth=0.5, color='blue')
            
            # Plot average with confidence interval
            mean_psd = np.mean(psd, axis=0)
            std_psd = np.std(psd, axis=0)
            ax_spec.semilogy(freqs[:100], mean_psd[:100], 'r-', linewidth=2, label='Mean')
            ax_spec.fill_between(freqs[:100], 
                                (mean_psd - std_psd)[:100], 
                                (mean_psd + std_psd)[:100], 
                                alpha=0.2, color='red')
            
            ax_spec.set_xlabel('Frequency (Hz)', fontweight='bold')
            ax_spec.set_ylabel('Power (µV²/Hz)', fontweight='bold')
            ax_spec.set_title(f'All Channel Spectra (n={len(eeg_picks)})', fontweight='bold')
            ax_spec.set_xlim(0, 50)
            ax_spec.grid(True, alpha=0.3)
            ax_spec.legend()
            
            plt.suptitle('Complete Spectral Analysis - All Channels', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Update canvas
            self.fig_montage = fig
            self.canvas_montage.figure = fig
            self.canvas_montage.draw()
            
            self.log_message(f"\nSpectra displayed for all {len(eeg_picks)} EEG channels")
            
        except Exception as e:
            self.ax_montage.clear()
            self.ax_montage.text(0.5, 0.5, f"Spectra plot error:\n{str(e)}",
                                transform=self.ax_montage.transAxes, ha='center', va='center')
            self.canvas_montage.draw()
            self.log_message(f"Error creating spectra plot: {str(e)}")
    
    def view_events(self):
        """Display all events"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        # Switch to main plot tab
        self.notebook.select(self.main_plot_frame)
        self.clear_main_plot()
        
        try:
            # Get events
            events, event_dict = mne.events_from_annotations(self.raw)
            
            self.log_message("\n=== ALL EVENTS ===")
            self.log_message(f"Total events: {len(events)}")
            self.log_message(f"Unique event types: {len(event_dict)}")
            
            # Display all events by type
            for event_name, event_id in event_dict.items():
                event_indices = np.where(events[:, 2] == event_id)[0]
                event_times = events[event_indices, 0] / self.raw.info['sfreq']
                self.log_message(f"\n{event_name} (ID: {event_id}): {len(event_indices)} events")
            
            # Plot all events
            self.plot_events_custom(events, event_dict)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to view events: {str(e)}")
            self.log_message(f"Error: {str(e)}")
    
    def plot_events_custom(self, events, event_dict):
        """Create a custom event plot showing all events"""
        self.clear_main_plot()
        
        try:
            unique_event_ids = np.unique(events[:, 2])
            event_names = {v: k for k, v in event_dict.items()}
            
            # Create color map for events
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_event_ids)))
            
            for i, event_id in enumerate(unique_event_ids):
                event_times = events[events[:, 2] == event_id, 0] / self.raw.info['sfreq']
                self.ax_main.scatter(event_times, [i] * len(event_times), 
                                    color=colors[i], alpha=0.6, s=20,
                                    label=f"{event_names.get(event_id, f'Event {event_id}')} (n={len(event_times)})")
            
            self.ax_main.set_xlabel('Time (s)')
            self.ax_main.set_ylabel('Event Type')
            self.ax_main.set_yticks(range(len(unique_event_ids)))
            self.ax_main.set_yticklabels([f"{event_names.get(eid, f'Event {eid}')}" 
                                        for eid in unique_event_ids])
            self.ax_main.set_title(f'All Events - Total: {len(events)}')
            self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            self.ax_main.grid(True, alpha=0.3)
            
        except Exception as e:
            self.ax_main.text(0.5, 0.5, f"Event plot error:\n{str(e)}", 
                        transform=self.ax_main.transAxes, ha='center', va='center')
        
        self.canvas_main.draw()
    
    def preprocess_data(self):
        """Basic preprocessing: filtering"""
        if self.raw is None:
            messagebox.showwarning("Warning", "Please load a file first")
            return
        
        # Switch to main plot tab
        self.notebook.select(self.main_plot_frame)
        self.clear_main_plot()
        
        try:
            self.log_message("\n=== PREPROCESSING ALL CHANNELS ===")
            
            # Apply bandpass filter
            self.log_message("Applying bandpass filter (1-40 Hz)...")
            self.raw.filter(1., 40., fir_design='firwin')
            
            # Apply notch filter
            self.log_message("Applying notch filter at 50 Hz...")
            self.raw.notch_filter(50.)
            
            # Handle duplicate events if they exist
            events, event_dict = mne.events_from_annotations(self.raw)
            
            # Remove exact duplicates
            unique_events = []
            seen = set()
            for event in events:
                event_key = tuple(event)
                if event_key not in seen:
                    unique_events.append(event)
                    seen.add(event_key)
            
            self.events = np.array(unique_events)
            
            self.log_message(f"Preprocessing complete")
            self.log_message(f"Events after deduplication: {len(self.events)}")
            
            # Show preprocessing summary
            self.ax_main.text(0.5, 0.5, '✓ Preprocessing Complete\nAll channels filtered', 
                            transform=self.ax_main.transAxes, ha='center', va='center', 
                            fontsize=14, color='green', 
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            self.ax_main.set_title("Preprocessing Status")
            self.ax_main.axis('off')
            self.canvas_main.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Preprocessing failed: {str(e)}")
            self.log_message(f"Error: {str(e)}")
    
    def export_results(self):
        """Export analysis results"""
        if self.epochs is None and self.raw is None:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        try:
            # Ask for export directory
            export_dir = filedialog.askdirectory(title="Select Export Directory")
            if not export_dir:
                return
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            # Export raw data info
            info_df = pd.DataFrame({
                'Parameter': ['Filename', 'Channels', 'EEG Channels', 'Duration (s)', 
                            'Sampling Rate (Hz)', 'Events', 'Data Type'],
                'Value': [os.path.basename(self.current_file), 
                         len(self.raw.ch_names),
                         len(mne.pick_types(self.raw.info, eeg=True)),
                         self.raw.times[-1],
                         self.raw.info['sfreq'],
                         len(self.events) if self.events is not None else 0,
                         'Resting State' if self.is_resting_state else 'Task ERP']
            })
            info_df.to_csv(os.path.join(export_dir, f'eeg_info_{timestamp}.csv'), index=False)
            
            # Export event info if available
            if self.events is not None and self.event_id is not None:
                event_df = pd.DataFrame(self.events, columns=['Sample', 'Duration', 'Event ID'])
                event_df['Time (s)'] = event_df['Sample'] / self.raw.info['sfreq']
                event_df['Event Type'] = event_df['Event ID'].map(
                    {v: k for k, v in self.event_id.items()})
                event_df.to_csv(os.path.join(export_dir, f'events_{timestamp}.csv'), index=False)
            
            # Export power data
            from scipy import signal as sig
            eeg_picks = mne.pick_types(self.raw.info, eeg=True)
            data, _ = self.raw[:, :]
            freqs, psd = sig.welch(data[eeg_picks], fs=self.raw.info['sfreq'], 
                                  nperseg=self.raw.info['sfreq']*2)
            
            psd_df = pd.DataFrame(psd.T, columns=[self.raw.ch_names[i] for i in eeg_picks])
            psd_df['Frequency (Hz)'] = freqs
            psd_df.to_csv(os.path.join(export_dir, f'psd_all_channels_{timestamp}.csv'), index=False)
            
            self.log_message(f"\nResults exported to: {export_dir}")
            messagebox.showinfo("Success", f"Results exported successfully to:\n{export_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
            self.log_message(f"Error exporting: {str(e)}")

def main():
    root = tk.Tk()
    app = EEGAnalyzerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()