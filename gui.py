"""
Road Load Simulator GUI
========================
Graphical user interface for the road load simulator.
Provides interactive parameter input and real-time visualization.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
from datetime import datetime
import numpy as np

# Import the simulator modules
from road_load_simulator import (
    VehicleParams,
    load_drive_cycle,
    calculate_road_load,
    estimate_range,
    estimate_range_multi_cycle,
    simulate_erev,
    simulate_gasoline,
    estimate_erev_range,
    EREVResults,
    create_output_directory,
    generate_summary_text,
    save_summary_text,
    save_timeseries_csv,
    save_erev_timeseries_csv,
    save_summary_html,
    generate_erev_plots,
    generate_erev_range_plots,
    generate_bev_range_plots,
    generate_erev_range_detailed_plots,
    generate_bev_range_detailed_plots,
    save_erev_range_html,
    save_bev_range_html,
    MPS_TO_MPH,
    W_TO_HP
)
import json


class RoadLoadSimulatorGUI:
    """Main GUI application for road load simulation."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Road Load Simulator")
        self.root.geometry("1400x800")
        self.root.minsize(1000, 600)  # Set minimum window size
        
        # Current results
        self.results = None
        self.vehicle = None
        self.cycle_name = None
        self.cycle_filepath = None
        
        # Setup GUI
        self.setup_ui()
        
    def setup_ui(self):
        """Create the main UI layout."""
        # Root expands
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Horizontal split so the sidebar can be resized
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left pane: scrollable parameters
        left_container = ttk.Frame(paned)
        paned.add(left_container, weight=0)
        try:
            paned.paneconfigure(left_container, minsize=440)
        except Exception:
            pass
        # Nudge initial sash so the sidebar starts wider
        try:
            paned.after(0, lambda: paned.sashpos(0, 460))
        except Exception:
            pass

        left_container.rowconfigure(0, weight=1)
        left_container.columnconfigure(0, weight=1)

        # Create canvas and scrollbar for left panel
        canvas = tk.Canvas(left_container, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas, padding=10)

        # Keep scrollregion and width in sync to avoid cut-off
        def _update_scrollregion(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", _update_scrollregion)

        self._left_canvas_window = canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw"
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_canvas_configure(event):
            # Make the inner frame match the canvas width
            canvas.itemconfigure(self._left_canvas_window, width=event.width)
            _update_scrollregion()

        canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel / trackpad scrolling (macOS uses small deltas)
        windowing_system = str(self.root.tk.call('tk', 'windowingsystem'))

        def _on_mousewheel(event):
            if windowing_system == 'aqua':
                delta = int(-event.delta)
                if delta != 0:
                    canvas.yview_scroll(delta, "units")
            else:
                canvas.yview_scroll(int(-event.delta / 120), "units")

        def _on_mousewheel_linux(event):
            if event.num == 4:
                canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                canvas.yview_scroll(1, "units")

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel_linux)
            canvas.bind_all("<Button-5>", _on_mousewheel_linux)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_mousewheel)
        canvas.bind("<Leave>", _unbind_mousewheel)

        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S), padx=(6, 0))

        # Right pane: plots
        right_frame = ttk.Frame(paned, padding=10)
        paned.add(right_frame, weight=1)
        
        # Setup left panel (inputs)
        self.setup_input_panel(self.scrollable_frame)
        
        # Setup right panel (plots)
        self.setup_plot_panel(right_frame)
    
    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            label = tk.Label(tooltip, text=text, 
                           background="#ffffcc",  # Light yellow background
                           foreground="#000000",  # Black text for readability
                           relief="solid", borderwidth=1, 
                           font=("Arial", 10),
                           wraplength=400, justify=tk.LEFT, 
                           padx=10, pady=8)
            label.pack()
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                delattr(widget, 'tooltip')
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
        
    def setup_input_panel(self, parent):
        """Create the input parameter panel."""
        # Allow the form to stretch to full sidebar width
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Title
        title = ttk.Label(parent, text="Simulation Parameters", 
                         font=("Arial", 14, "bold"))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Vehicle Parameters Section
        vehicle_frame = ttk.LabelFrame(parent, text="Vehicle Parameters", padding=10)
        vehicle_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configure column weights for proper alignment
        vehicle_frame.columnconfigure(0, weight=0, minsize=180)  # Label column fixed width
        vehicle_frame.columnconfigure(1, weight=1)  # Entry column expands
        
        row = 0
        # Vehicle Class Selector
        ttk.Label(vehicle_frame, text="Vehicle Class:").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        self.vehicle_class_var = tk.StringVar(value="bev")
        self.vehicle_class_dropdown = ttk.Combobox(
            vehicle_frame, 
            textvariable=self.vehicle_class_var,
            values=["Battery EV", "EREV", "Gasoline"],
            state="readonly"
        )
        self.vehicle_class_dropdown.current(0)
        self.vehicle_class_dropdown.grid(row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.vehicle_class_dropdown.bind('<<ComboboxSelected>>', self.on_vehicle_class_change)
        
        row += 1
        # Separator
        ttk.Separator(vehicle_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=2, sticky='ew', pady=10)
        
        row += 1
        # Common Parameters (all vehicle types)
        ttk.Label(vehicle_frame, text="Mass (kg):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.mass_var = tk.StringVar(value="1847")
        ttk.Entry(vehicle_frame, textvariable=self.mass_var).grid(
            row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        row += 1
        ttk.Label(vehicle_frame, text="Frontal Area (m²):").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        self.frontal_area_var = tk.StringVar(value="2.22")
        ttk.Entry(vehicle_frame, textvariable=self.frontal_area_var).grid(
            row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        row += 1
        ttk.Label(vehicle_frame, text="Drag Coefficient:").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        self.cd_var = tk.StringVar(value="0.23")
        ttk.Entry(vehicle_frame, textvariable=self.cd_var).grid(
            row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        row += 1
        ttk.Label(vehicle_frame, text="Rolling Resistance:").grid(
            row=row, column=0, sticky=tk.W, pady=5)
        self.crr_var = tk.StringVar(value="0.01")
        ttk.Entry(vehicle_frame, textvariable=self.crr_var).grid(
            row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        self.common_row_end = row
        
        # BEV-specific frame (shown for BEV and EREV)
        row += 1
        self.bev_frame = ttk.Frame(vehicle_frame)
        self.bev_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.bev_frame_row = row
        
        # Configure BEV frame columns
        self.bev_frame.columnconfigure(0, weight=0, minsize=180)
        self.bev_frame.columnconfigure(1, weight=1)
        
        bev_row = 0
        ttk.Label(self.bev_frame, text="Regen Efficiency:").grid(
            row=bev_row, column=0, sticky=tk.W, pady=5)
        self.regen_var = tk.StringVar(value="0.75")
        ttk.Entry(self.bev_frame, textvariable=self.regen_var).grid(
            row=bev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        bev_row += 1
        ttk.Label(self.bev_frame, text="Aux Power (W):").grid(
            row=bev_row, column=0, sticky=tk.W, pady=5)
        self.aux_power_var = tk.StringVar(value="0")
        ttk.Entry(self.bev_frame, textvariable=self.aux_power_var).grid(
            row=bev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        bev_row += 1
        ttk.Label(self.bev_frame, text="Battery (kWh):").grid(
            row=bev_row, column=0, sticky=tk.W, pady=5)
        self.battery_var = tk.StringVar(value="75")
        ttk.Entry(self.bev_frame, textvariable=self.battery_var).grid(
            row=bev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        bev_row += 1
        ttk.Label(self.bev_frame, text="Usable Battery %:").grid(
            row=bev_row, column=0, sticky=tk.W, pady=5)
        self.usable_battery_var = tk.StringVar(value="90")
        ttk.Entry(self.bev_frame, textvariable=self.usable_battery_var).grid(
            row=bev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # EREV-specific frame (only shown for EREV)
        row += 1
        self.erev_frame = ttk.Frame(vehicle_frame)
        self.erev_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.erev_frame_row = row
        
        # Configure EREV frame columns
        self.erev_frame.columnconfigure(0, weight=0, minsize=180)
        self.erev_frame.columnconfigure(1, weight=1)
        
        erev_row = 0
        ttk.Separator(self.erev_frame, orient='horizontal').grid(
            row=erev_row, column=0, columnspan=2, sticky='ew', pady=5)
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="Generator Power (kW):").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.generator_power_var = tk.StringVar(value="100")
        ttk.Entry(self.erev_frame, textvariable=self.generator_power_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="BSFC (g/kWh):").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.bsfc_var = tk.StringVar(value="250")
        ttk.Entry(self.erev_frame, textvariable=self.bsfc_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="Fuel Tank (gal):").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.fuel_tank_var = tk.StringVar(value="10")
        ttk.Entry(self.erev_frame, textvariable=self.fuel_tank_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        mode_label_frame = ttk.Frame(self.erev_frame)
        mode_label_frame.grid(row=erev_row, column=0, sticky=tk.W, pady=5)
        ttk.Label(mode_label_frame, text="EREV Mode:").pack(side=tk.LEFT)
        info_btn = ttk.Label(mode_label_frame, text=" ℹ", foreground="blue", cursor="hand2")
        info_btn.pack(side=tk.LEFT)
        
        # Add detailed tooltip for EREV modes
        mode_info = (
            "EREV Operating Modes:\n\n"
            "• charge_depleting: Pure EV mode until battery reaches\n"
            "  SOC Sustain threshold, then generator maintains that level.\n"
            "  Best for maximizing electric-only range.\n\n"
            "• blended: Generator assists when power demand is high\n"
            "  or SOC drops below Blended threshold. Optimizes power\n"
            "  distribution for demanding routes.\n\n"
            "• hold: Generator runs to maintain starting SOC level,\n"
            "  battery only covers power peaks. Use to preserve\n"
            "  battery for later (e.g., save EV for city after highway)."
        )
        self.create_tooltip(info_btn, mode_info)
        
        self.erev_mode_var = tk.StringVar(value="charge_depleting")
        self.erev_mode_dropdown = ttk.Combobox(
            self.erev_frame,
            textvariable=self.erev_mode_var,
            values=["charge_depleting", "blended", "hold"],
            width=20,
            state="readonly"
        )
        self.erev_mode_dropdown.current(0)
        self.erev_mode_dropdown.grid(row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="SOC Sustain %:").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.soc_sustain_var = tk.StringVar(value="20")
        ttk.Entry(self.erev_frame, textvariable=self.soc_sustain_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="SOC Blended %:").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.soc_blended_var = tk.StringVar(value="30")
        ttk.Entry(self.erev_frame, textvariable=self.soc_blended_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        erev_row += 1
        ttk.Label(self.erev_frame, text="Starting SOC %:").grid(
            row=erev_row, column=0, sticky=tk.W, pady=5)
        self.starting_soc_var = tk.StringVar(value="100")
        ttk.Entry(self.erev_frame, textvariable=self.starting_soc_var).grid(
            row=erev_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Gasoline-specific frame (only shown for Gasoline)
        row += 1
        self.gas_frame = ttk.Frame(vehicle_frame)
        self.gas_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E))
        self.gas_frame_row = row
        
        # Configure Gasoline frame columns
        self.gas_frame.columnconfigure(0, weight=0, minsize=180)
        self.gas_frame.columnconfigure(1, weight=1)
        
        gas_row = 0
        ttk.Label(self.gas_frame, text="Fuel Tank (gal):").grid(
            row=gas_row, column=0, sticky=tk.W, pady=5)
        self.gas_fuel_tank_var = tk.StringVar(value="15")
        ttk.Entry(self.gas_frame, textvariable=self.gas_fuel_tank_var).grid(
            row=gas_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        gas_row += 1
        ttk.Label(self.gas_frame, text="Fuel Economy (MPG):").grid(
            row=gas_row, column=0, sticky=tk.W, pady=5)
        self.fuel_economy_var = tk.StringVar(value="25")
        ttk.Entry(self.gas_frame, textvariable=self.fuel_economy_var).grid(
            row=gas_row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Initially hide EREV and Gasoline frames
        self.erev_frame.grid_remove()
        self.gas_frame.grid_remove()
        
        # Drive Cycle Section
        cycle_frame = ttk.LabelFrame(parent, text="Drive Cycle", padding="10")
        cycle_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10), padx=(0, 10))
        
        # Configure cycle frame columns
        cycle_frame.columnconfigure(0, weight=0, minsize=180)
        cycle_frame.columnconfigure(1, weight=1)
        
        # Available cycles
        ttk.Label(cycle_frame, text="Select Cycle:").grid(
            row=0, column=0, sticky=tk.W, pady=5)
        
        self.cycle_var = tk.StringVar()
        cycle_choices = self.get_available_cycles()
        self.cycle_dropdown = ttk.Combobox(cycle_frame, textvariable=self.cycle_var, 
                                          values=cycle_choices, state="readonly")
        if cycle_choices:
            self.cycle_dropdown.current(0)
        self.cycle_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Or browse for custom cycle
        ttk.Button(cycle_frame, text="Browse...", 
                  command=self.browse_cycle).grid(row=1, column=1, pady=5, sticky=tk.W)
        
        # Action Buttons
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Run Simulation", 
                  command=self.run_simulation, width=20).pack(pady=5)
        ttk.Button(button_frame, text="Estimate Range", 
                  command=self.estimate_range, width=20).pack(pady=5)
        ttk.Button(button_frame, text="Save Results", 
                  command=self.save_results, width=20).pack(pady=5)
        ttk.Button(button_frame, text="Load Preset", 
                  command=self.load_preset, width=20).pack(pady=5)
        ttk.Button(button_frame, text="Save Preset", 
                  command=self.save_preset, width=20).pack(pady=5)
    
    def on_vehicle_class_change(self, event=None):
        """Handle vehicle class dropdown change - show/hide appropriate parameter frames."""
        vehicle_class = self.vehicle_class_dropdown.get()
        
        if vehicle_class == "Battery EV":
            self.bev_frame.grid()
            self.erev_frame.grid_remove()
            self.gas_frame.grid_remove()
        elif vehicle_class == "EREV":
            self.bev_frame.grid()
            self.erev_frame.grid()
            self.gas_frame.grid_remove()
        elif vehicle_class == "Gasoline":
            self.bev_frame.grid_remove()
            self.erev_frame.grid_remove()
            self.gas_frame.grid()
        
    def setup_plot_panel(self, parent):
        """Create the plot display panel with resizable split."""
        # Create PanedWindow for resizable split between plots and results
        paned_window = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Top pane: plots
        plot_frame = ttk.Frame(paned_window)
        paned_window.add(plot_frame, weight=3)
        
        # Create matplotlib figure with responsive sizing
        self.fig = Figure(figsize=(9, 6), dpi=100)
        
        # Create 2x2 subplots
        self.ax1 = self.fig.add_subplot(2, 2, 1)  # Speed
        self.ax2 = self.fig.add_subplot(2, 2, 2)  # Power
        self.ax3 = self.fig.add_subplot(2, 2, 3)  # Energy
        self.ax4 = self.fig.add_subplot(2, 2, 4)  # Grade
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plots
        self.update_plots(None)
        
        # Bottom pane: Results Summary (resizable)
        results_frame = ttk.LabelFrame(paned_window, text="Results Summary", padding="5")
        paned_window.add(results_frame, weight=1)
        
        # Treeview for results table with style
        style = ttk.Style()
        style.configure("Compact.Treeview", background="white", foreground="black", 
                       fieldbackground="white", rowheight=20)
        style.configure("Compact.Treeview.Heading", background="white", foreground="white", 
                       borderwidth=0, relief="flat")
        style.layout("Compact.Treeview", [('Compact.Treeview.treearea', {'sticky': 'nswe'})])
        style.map("Compact.Treeview", background=[('selected', '#e6f2ff')])
        
        columns = ('value1', 'value2')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', 
                                        height=8, style="Compact.Treeview")
        
        self.results_tree.heading('value1', text='')
        self.results_tree.heading('value2', text='')
        self.results_tree.column('value1', width=350, minwidth=250, anchor=tk.W)
        self.results_tree.column('value2', width=350, minwidth=250, anchor=tk.W)
        
        # Add grid lines using tags
        self.results_tree.tag_configure('oddrow', background='white')
        self.results_tree.tag_configure('evenrow', background='#f5f5f5')
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def get_available_cycles(self):
        """Scan drive_cycles folder and return list of available cycles."""
        cycles_dir = "drive_cycles"
        if not os.path.exists(cycles_dir):
            return []
        
        cycle_files = [f for f in os.listdir(cycles_dir) if f.endswith('.csv')]
        return sorted([os.path.splitext(f)[0] for f in cycle_files])
    
    def browse_cycle(self):
        """Open file dialog to select a drive cycle CSV."""
        filename = filedialog.askopenfilename(
            title="Select Drive Cycle",
            initialdir="drive_cycles",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            # Add to dropdown and select it
            cycle_name = os.path.splitext(os.path.basename(filename))[0]
            self.cycle_var.set(cycle_name)
            # Update dropdown values if needed
            current_values = list(self.cycle_dropdown['values'])
            if cycle_name not in current_values:
                current_values.append(cycle_name)
                self.cycle_dropdown['values'] = current_values
    
    def get_vehicle_class_code(self):
        """Convert dropdown selection to vehicle class code."""
        class_map = {
            "Battery EV": "bev",
            "EREV": "erev",
            "Gasoline": "gasoline"
        }
        return class_map.get(self.vehicle_class_dropdown.get(), "bev")
    
    def get_vehicle_params(self):
        """Get vehicle parameters from input fields based on vehicle class."""
        try:
            vehicle_class = self.get_vehicle_class_code()
            
            # Common parameters
            params = {
                'mass': float(self.mass_var.get()),
                'frontal_area': float(self.frontal_area_var.get()),
                'drag_coefficient': float(self.cd_var.get()),
                'rolling_resistance': float(self.crr_var.get()),
                'vehicle_class': vehicle_class
            }
            
            if vehicle_class == 'bev':
                params.update({
                    'regen_efficiency': float(self.regen_var.get()),
                    'auxiliary_power': float(self.aux_power_var.get()),
                    'battery_capacity': float(self.battery_var.get()),
                    'usable_battery_pct': float(self.usable_battery_var.get())
                })
            elif vehicle_class == 'erev':
                params.update({
                    'regen_efficiency': float(self.regen_var.get()),
                    'auxiliary_power': float(self.aux_power_var.get()),
                    'battery_capacity': float(self.battery_var.get()),
                    'usable_battery_pct': float(self.usable_battery_var.get()),
                    'generator_power_kw': float(self.generator_power_var.get()),
                    'bsfc_g_kwh': float(self.bsfc_var.get()),
                    'fuel_tank_gallons': float(self.fuel_tank_var.get()),
                    'erev_mode': self.erev_mode_var.get(),
                    'soc_sustain_pct': float(self.soc_sustain_var.get()),
                    'soc_blended_threshold_pct': float(self.soc_blended_var.get())
                })
            elif vehicle_class == 'gasoline':
                params.update({
                    'fuel_tank_gallons': float(self.gas_fuel_tank_var.get()),
                    'fuel_economy_mpg': float(self.fuel_economy_var.get())
                })
            
            vehicle = VehicleParams(**params)
            return vehicle
        except ValueError as e:
            messagebox.showerror("Input Error", 
                               f"Invalid input values. Please check all fields.\n{str(e)}")
            return None
    
    def run_simulation(self):
        """Execute the road load simulation."""
        # Get parameters
        self.vehicle = self.get_vehicle_params()
        if self.vehicle is None:
            return
        
        # Get cycle
        self.cycle_name = self.cycle_var.get()
        if not self.cycle_name:
            messagebox.showerror("Error", "Please select a drive cycle.")
            return
        
        self.cycle_filepath = os.path.join("drive_cycles", f"{self.cycle_name}.csv")
        if not os.path.exists(self.cycle_filepath):
            messagebox.showerror("Error", f"Drive cycle file not found: {self.cycle_filepath}")
            return
        
        try:
            # Load cycle (in m/s format)
            cycle = load_drive_cycle(self.cycle_filepath)
            
            # Run base road load (wheel demand)
            self.results = calculate_road_load(self.vehicle, cycle)

            # If EREV, run generator/battery split simulation
            self.erev_results = None
            if self.vehicle.vehicle_class == 'erev':
                try:
                    starting_soc = float(self.starting_soc_var.get())
                except ValueError:
                    messagebox.showerror("Input Error", "Starting SOC must be a number.")
                    return

                starting_soc = max(min(starting_soc, 100.0), 0.0)

                self.erev_results = simulate_erev(
                    self.vehicle,
                    self.cycle_filepath,
                    starting_soc=starting_soc,
                    precomputed_results=self.results
                )
            
            # Update display
            self.update_plots(self.results)
            self.update_results_text()
            self.results_tree.update_idletasks()
            
        except Exception as e:
            messagebox.showerror("Simulation Error", f"Error running simulation:\n{str(e)}")
    
    def estimate_range(self):
        """Open range estimation window."""
        # Get vehicle parameters
        self.vehicle = self.get_vehicle_params()
        if not self.vehicle:
            return
        
        # Get starting SOC for EREV
        starting_soc = 100.0
        if self.vehicle.vehicle_class == 'erev':
            try:
                starting_soc = float(self.starting_soc_var.get())
            except:
                starting_soc = 100.0
        
        # Open range estimator window
        RangeEstimatorGUI(self.root, self.vehicle, starting_soc)
    
    def update_plots(self, results):
        """Update the plot displays."""
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        if results is None:
            # Show empty plots with labels
            self.ax1.set_title("Speed vs Distance")
            self.ax1.set_xlabel("Distance (miles)")
            self.ax1.set_ylabel("Speed (mph)")
            self.ax1.grid(True, alpha=0.3)
            
            self.ax2.set_title("Power vs Distance")
            self.ax2.set_xlabel("Distance (miles)")
            self.ax2.set_ylabel("Power (kW)")
            self.ax2.grid(True, alpha=0.3)
            
            self.ax3.set_title("Cumulative Energy")
            self.ax3.set_xlabel("Distance (miles)")
            self.ax3.set_ylabel("Energy (kWh)")
            self.ax3.grid(True, alpha=0.3)
            
            self.ax4.set_title("Road Grade")
            self.ax4.set_xlabel("Distance (miles)")
            self.ax4.set_ylabel("Grade (%)")
            self.ax4.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            return
        
        dt = np.diff(results.time, prepend=0)
        distance_miles = np.cumsum(results.speed * dt) / 1609.34

        # Plot 1: Speed
        self.ax1.plot(distance_miles, results.speed * MPS_TO_MPH, 'b-', linewidth=1.5)
        self.ax1.set_xlabel("Distance (miles)")
        self.ax1.set_ylabel("Speed (mph)")
        self.ax1.set_title("Vehicle Speed")
        self.ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power
        power_kw = results.power / 1000
        self.ax2.plot(distance_miles, power_kw, 'r-', linewidth=1.5)
        self.ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax2.fill_between(distance_miles, 0, power_kw, where=power_kw >= 0,
                             color='red', alpha=0.3, label='Traction')
        self.ax2.fill_between(distance_miles, 0, power_kw, where=power_kw < 0,
                             color='green', alpha=0.3, label='Regen')
        self.ax2.set_xlabel("Distance (miles)")
        self.ax2.set_ylabel("Power (kW)")
        self.ax2.set_title("Power Demand")
        self.ax2.legend(loc='upper right')
        self.ax2.grid(True, alpha=0.3)
        
        # Plot 3: Energy
        energy_kwh = results.energy_cumulative / 3.6e6
        self.ax3.plot(distance_miles, energy_kwh, 'g-', linewidth=1.5)
        self.ax3.set_xlabel("Distance (miles)")
        self.ax3.set_ylabel("Energy (kWh)")
        self.ax3.set_title("Cumulative Energy")
        self.ax3.grid(True, alpha=0.3)
        
        # Plot 4: Grade
        self.ax4.plot(distance_miles, results.grade, 'm-', linewidth=1.5)
        self.ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        self.ax4.fill_between(distance_miles, 0, results.grade, alpha=0.3, color='m')
        self.ax4.set_xlabel("Distance (miles)")
        self.ax4.set_ylabel("Grade (%)")
        self.ax4.set_title("Road Grade")
        self.ax4.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def update_results_text(self):
        """Update the results summary table."""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.results is None or self.vehicle is None:
            self.results_tree.insert('', 'end', values=('No Results - Run simulation', ''))
            return
        
        # Calculate key metrics
        peak_traction_kw = self.results.peak_power_traction / 1000
        peak_traction_hp = self.results.peak_power_traction * W_TO_HP
        peak_regen_kw = self.results.peak_power_regen / 1000
        peak_regen_hp = self.results.peak_power_regen * W_TO_HP
        avg_power_kw = self.results.average_power / 1000
        avg_power_hp = self.results.average_power * W_TO_HP
        total_energy_kwh = self.results.total_energy / 3.6e6
        traction_energy_kwh = self.results.traction_energy / 3.6e6
        regen_energy_kwh = self.results.regen_energy / 3.6e6
        
        duration = self.results.time[-1]
        max_speed_mps = np.max(self.results.speed)
        max_speed_mph = max_speed_mps * MPS_TO_MPH
        avg_speed_mps = np.mean(self.results.speed)
        avg_speed_mph = avg_speed_mps * MPS_TO_MPH
        
        # Calculate distance
        dt = np.diff(self.results.time, prepend=0)
        distance_m = np.sum(self.results.speed * dt)
        distance_km = distance_m / 1000
        distance_mi = distance_m / 1609.34
        
        energy_recovery_pct = (regen_energy_kwh/traction_energy_kwh*100) if traction_energy_kwh > 0 else 0
        
        # Create compact two-column data layout
        data = [
            ('Cycle', self.cycle_name),
            ('Duration', f'{duration:.1f} s ({duration/60:.1f} min)'),
            ('Max Speed', f'{max_speed_mps:.2f} m/s ({max_speed_mph:.1f} mph)'),
            ('Avg Speed', f'{avg_speed_mps:.2f} m/s ({avg_speed_mph:.1f} mph)'),
            ('Peak Traction Power', f'{peak_traction_kw:.2f} kW ({peak_traction_hp:.1f} hp)'),
            ('Peak Regen Power', f'{peak_regen_kw:.2f} kW ({peak_regen_hp:.1f} hp)'),
            ('Avg Power', f'{avg_power_kw:.2f} kW ({avg_power_hp:.1f} hp)'),
            ('Distance', f'{distance_km:.2f} km ({distance_mi:.1f} mi)'),
            ('Net Energy', f'{total_energy_kwh:.4f} kWh'),
            ('Traction Energy', f'{traction_energy_kwh:.4f} kWh'),
            ('Regen Energy', f'{regen_energy_kwh:.4f} kWh'),
            ('Regen Recovery', f'{energy_recovery_pct:.1f}%'),
            ('Vehicle Mass', f'{self.vehicle.mass} kg'),
            ('Frontal Area', f'{self.vehicle.frontal_area} m²'),
            ('Drag Coeff', f'{self.vehicle.drag_coefficient}'),
            ('Roll Resist', f'{self.vehicle.rolling_resistance}'),
        ]

        if getattr(self, 'erev_results', None) and self.vehicle.vehicle_class == 'erev':
            er = self.erev_results
            data.extend([
                ('EV Miles', f'{er.ev_only_miles:.2f} mi'),
                ('Generator Miles', f'{er.generator_miles:.2f} mi'),
                ('Generator Runtime', f'{er.generator_runtime_seconds/60:.1f} min'),
                ('Generator Energy', f'{er.generator_energy_kwh:.2f} kWh'),
                ('Fuel Used', f'{er.fuel_used_gallons:.3f} gal'),
                ('Final SOC', f'{er.final_soc:.1f}% (min {er.min_soc:.1f}%)'),
                ('MPGe', f'{er.mpge:.1f}'),
                ('Energy per Mile', f'{er.kwh_per_mile:.3f} kWh/mi')
            ])
        
        # Insert data in two columns with alternating row colors
        half = (len(data) + 1) // 2
        for i in range(half):
            tag = 'evenrow' if i % 2 == 0 else 'oddrow'
            left_metric, left_value = data[i]
            if i + half < len(data):
                right_metric, right_value = data[i + half]
                self.results_tree.insert('', 'end', values=(
                    f'{left_metric}: {left_value}',
                    f'{right_metric}: {right_value}'
                ), tags=(tag,))
            else:
                self.results_tree.insert('', 'end', values=(
                    f'{left_metric}: {left_value}', ''
                ), tags=(tag,))
    
    def save_results(self):
        """Save simulation results to files."""
        if self.results is None or self.vehicle is None:
            messagebox.showwarning("No Results", "Please run a simulation first.")
            return
        
        try:
            # Generate timestamp and create output directory
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            output_dir = 'outputs'
            run_output_dir = create_output_directory(output_dir, self.cycle_name, timestamp)
            
            # Generate summary
            summary = generate_summary_text(
                self.vehicle, self.results, self.cycle_name, self.cycle_filepath
            )
            
            # Save text summary
            save_summary_text(summary, run_output_dir)
            
            # Save time-series CSV (wheel-level)
            save_timeseries_csv(self.results, run_output_dir)
            
            # Save plots
            plot_path = os.path.join(run_output_dir, "plots.png")
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            secondary_plot = None
            
            # Save HTML report
            # EREV-specific outputs
            if getattr(self, 'erev_results', None) and self.vehicle.vehicle_class == 'erev':
                erev_plot_path = generate_erev_plots(self.results, self.erev_results, self.cycle_name, run_output_dir)
                secondary_plot = os.path.basename(erev_plot_path)
                # Save EREV time-series (SOC/generator/fuel)
                save_erev_timeseries_csv({
                    'time_trace': self.erev_results.time_trace,
                    'distance_miles_trace': self.erev_results.distance_miles_trace,
                    'soc_trace': self.erev_results.soc_timeline,
                    'generator_output_kw': self.erev_results.generator_output_kw,
                    'generator_on_flags': self.erev_results.generator_on_flags,
                    'fuel_remaining_gal': self.erev_results.fuel_remaining_gal
                }, run_output_dir)
                # Add a small text addendum for generator metrics
                addendum_path = os.path.join(run_output_dir, "erev_addendum.txt")
                with open(addendum_path, 'w') as f:
                    f.write("EREV Generator Summary\n")
                    f.write(f"Generator runtime: {self.erev_results.generator_runtime_seconds/60:.2f} min\n")
                    f.write(f"Generator energy: {self.erev_results.generator_energy_kwh:.2f} kWh\n")
                    f.write(f"Fuel used: {self.erev_results.fuel_used_gallons:.3f} gal\n")
                    f.write(f"EV-only miles: {self.erev_results.ev_only_miles:.2f} mi\n")
                    f.write(f"Generator miles: {self.erev_results.generator_miles:.2f} mi\n")
                    f.write(f"Final SOC: {self.erev_results.final_soc:.1f}% (min {self.erev_results.min_soc:.1f}%)\n")
                extra_note = "\nEREV plots: erev_plots.png\nEREV timeseries: erev_timeseries.csv"

            # BEV-specific secondary plot (efficiency-focused)
            if self.vehicle.vehicle_class == 'bev':
                bev_plot_path = generate_bev_plots(self.results, self.cycle_name, run_output_dir)
                secondary_plot = os.path.basename(bev_plot_path)
                extra_note = ""

            save_summary_html(
                self.vehicle, self.results, self.cycle_name, 
                self.cycle_filepath, run_output_dir, "plots.png",
                erev_results=getattr(self, 'erev_results', None),
                secondary_plot_filename=secondary_plot
            )

            messagebox.showinfo("Success", 
                              f"Results saved to:\n{run_output_dir}{extra_note}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Error saving results:\n{str(e)}")
    
    def load_preset(self):
        """Load preset vehicle configurations from external JSON file."""
        # Load presets from JSON file
        presets_file = "vehicle_presets.json"
        if not os.path.exists(presets_file):
            messagebox.showerror("Error", f"Presets file not found: {presets_file}")
            return
        
        try:
            with open(presets_file, 'r') as f:
                data = json.load(f)
                all_presets = data.get('presets', [])
        except Exception as e:
            messagebox.showerror("Error", f"Error loading presets: {str(e)}")
            return
        
        # Get current vehicle class
        current_class = self.get_vehicle_class_code()
        
        # Filter presets by vehicle class
        filtered_presets = [p for p in all_presets if p.get('vehicle_class') == current_class]
        
        if not filtered_presets:
            messagebox.showinfo("No Presets", 
                              f"No presets found for vehicle class: {self.vehicle_class_dropdown.get()}")
            return
        
        # Create selection dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Preset")
        dialog.geometry("400x500")
        
        ttk.Label(dialog, text=f"Select a {self.vehicle_class_dropdown.get()} preset:", 
                 font=("Arial", 12)).pack(pady=10)
        
        # Scrollable frame for presets
        canvas = tk.Canvas(dialog)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        preset_var = tk.StringVar()
        for preset in filtered_presets:
            ttk.Radiobutton(scrollable_frame, text=preset['name'], variable=preset_var,
                          value=preset['name']).pack(anchor=tk.W, padx=20, pady=5)
        
        canvas.pack(side="left", fill="both", expand=True, padx=10)
        scrollbar.pack(side="right", fill="y")
        
        def apply_preset():
            selected = preset_var.get()
            if not selected:
                return
            
            # Find the selected preset
            preset = next((p for p in filtered_presets if p['name'] == selected), None)
            if not preset:
                return
            
            # Apply common parameters
            self.mass_var.set(str(preset.get("mass", "")))
            self.frontal_area_var.set(str(preset.get("frontal_area", "")))
            self.cd_var.set(str(preset.get("cd", "")))
            self.crr_var.set(str(preset.get("crr", "")))
            
            # Apply class-specific parameters
            if current_class in ['bev', 'erev']:
                self.regen_var.set(str(preset.get("regen", "0.7")))
                self.aux_power_var.set(str(preset.get("aux_power", "0")))
                self.battery_var.set(str(preset.get("battery", "75")))
                self.usable_battery_var.set(str(preset.get("usable_battery_pct", "90")))
            
            if current_class == 'erev':
                self.generator_power_var.set(str(preset.get("generator_power_kw", "100")))
                self.bsfc_var.set(str(preset.get("bsfc_g_kwh", "250")))
                self.fuel_tank_var.set(str(preset.get("fuel_tank_gallons", "10")))
                self.erev_mode_var.set(preset.get("erev_mode", "charge_depleting"))
                self.soc_sustain_var.set(str(preset.get("soc_sustain_pct", "20")))
                self.soc_blended_var.set(str(preset.get("soc_blended_threshold_pct", "30")))
            
            if current_class == 'gasoline':
                self.gas_fuel_tank_var.set(str(preset.get("fuel_tank_gallons", "15")))
                self.fuel_economy_var.set(str(preset.get("fuel_economy_mpg", "25")))
            
            dialog.destroy()
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Apply", command=apply_preset).pack()

    def save_preset(self):
        """Save current vehicle parameters as a preset."""
        presets_file = "vehicle_presets.json"
        
        # Load existing presets
        if os.path.exists(presets_file):
            try:
                with open(presets_file, 'r') as f:
                    data = json.load(f)
                    all_presets = data.get('presets', [])
            except Exception as e:
                messagebox.showerror("Error", f"Error loading presets file: {str(e)}")
                return
        else:
            all_presets = []
        
        # Create dialog for saving
        dialog = tk.Toplevel(self.root)
        dialog.title("Save Preset")
        dialog.geometry("450x300")
        dialog.grab_set()
        
        ttk.Label(dialog, text="Save Vehicle Preset", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Preset name input
        name_frame = ttk.Frame(dialog)
        name_frame.pack(pady=10, padx=20, fill=tk.X)
        ttk.Label(name_frame, text="Preset Name:").pack(side=tk.LEFT, padx=5)
        name_var = tk.StringVar()
        name_entry = ttk.Entry(name_frame, textvariable=name_var, width=30)
        name_entry.pack(side=tk.LEFT, padx=5)
        name_entry.focus()
        
        # Get current vehicle class and existing presets of same class
        current_class = self.get_vehicle_class_code()
        existing_names = [p['name'] for p in all_presets if p.get('vehicle_class') == current_class]
        
        # Show existing presets for this class
        if existing_names:
            ttk.Label(dialog, text=f"Existing {self.vehicle_class_dropdown.get()} presets:", 
                     font=("Arial", 10)).pack(pady=(10, 5))
            
            list_frame = ttk.Frame(dialog)
            list_frame.pack(pady=5, padx=20, fill=tk.BOTH, expand=True)
            
            listbox = tk.Listbox(list_frame, height=6)
            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            listbox.configure(yscrollcommand=scrollbar.set)
            
            for name in existing_names:
                listbox.insert(tk.END, name)
            
            # Double-click to select name
            def on_listbox_select(event):
                selection = listbox.curselection()
                if selection:
                    name_var.set(listbox.get(selection[0]))
            listbox.bind('<<ListboxSelect>>', on_listbox_select)
        
        def save_preset_action():
            preset_name = name_var.get().strip()
            if not preset_name:
                messagebox.showwarning("Invalid Name", "Please enter a preset name.")
                return
            
            # Check if name exists
            existing_preset = next((p for p in all_presets if p['name'] == preset_name and 
                                   p.get('vehicle_class') == current_class), None)
            
            if existing_preset:
                # Confirm overwrite
                response = messagebox.askyesno(
                    "Overwrite Preset?",
                    f"Preset '{preset_name}' already exists. Overwrite it?"
                )
                if not response:
                    return
                # Remove old preset
                all_presets.remove(existing_preset)
            
            # Collect current parameters
            new_preset = {
                "name": preset_name,
                "vehicle_class": current_class,
                "mass": float(self.mass_var.get()),
                "frontal_area": float(self.frontal_area_var.get()),
                "cd": float(self.cd_var.get()),
                "crr": float(self.crr_var.get())
            }
            
            # Add class-specific parameters
            if current_class in ['bev', 'erev']:
                new_preset.update({
                    "regen": float(self.regen_var.get()),
                    "aux_power": float(self.aux_power_var.get()),
                    "battery": float(self.battery_var.get()),
                    "usable_battery_pct": float(self.usable_battery_var.get())
                })
            
            if current_class == 'erev':
                new_preset.update({
                    "generator_power_kw": float(self.generator_power_var.get()),
                    "bsfc_g_kwh": float(self.bsfc_var.get()),
                    "fuel_tank_gallons": float(self.fuel_tank_var.get()),
                    "erev_mode": self.erev_mode_var.get(),
                    "soc_sustain_pct": float(self.soc_sustain_var.get()),
                    "soc_blended_threshold_pct": float(self.soc_blended_var.get())
                })
            
            if current_class == 'gasoline':
                new_preset.update({
                    "fuel_tank_gallons": float(self.gas_fuel_tank_var.get()),
                    "fuel_economy_mpg": float(self.fuel_economy_var.get())
                })
            
            # Add to presets list
            all_presets.append(new_preset)
            
            # Save to file
            try:
                with open(presets_file, 'w') as f:
                    json.dump({"presets": all_presets}, f, indent=2)
                
                messagebox.showinfo("Success", f"Preset '{preset_name}' saved successfully!")
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Error saving preset: {str(e)}")
        
        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=15)
        ttk.Button(button_frame, text="Save", command=save_preset_action, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=12).pack(side=tk.LEFT, padx=5)


class RangeEstimatorGUI:
    """Separate GUI window for range estimation."""
    
    def __init__(self, parent, vehicle, starting_soc=100.0):
        """Initialize the range estimator window."""
        self.vehicle = vehicle
        self.starting_soc = starting_soc
        self.range_results = None
        
        # Create new window
        self.window = tk.Toplevel(parent)
        self.window.title("Range Estimator")
        self.window.geometry("950x700")
        
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title based on vehicle class
        title_text = {
            'bev': "Battery EV Range Estimator",
            'erev': "Extended Range EV Range Estimator",
            'gasoline': "Gasoline Vehicle Range Estimator"
        }.get(vehicle.vehicle_class, "Range Estimator")
        
        title = ttk.Label(main_frame, text=title_text, font=("Arial", 16, "bold"))
        title.pack(pady=(0, 20))
        
        # Vehicle info display
        info_frame = ttk.LabelFrame(main_frame, text="Vehicle Configuration", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Build vehicle info text based on class
        if vehicle.vehicle_class == 'bev':
            vehicle_text = (
                f"Mass: {vehicle.mass} kg  |  Cd: {vehicle.drag_coefficient}  |  "
                f"Battery: {vehicle.battery_capacity} kWh ({vehicle.usable_battery_pct:.0f}% usable)  |  "
                f"Aux Power: {vehicle.auxiliary_power} W"
            )
        elif vehicle.vehicle_class == 'erev':
            vehicle_text = (
                f"Mass: {vehicle.mass} kg  |  Cd: {vehicle.drag_coefficient}  |  "
                f"Battery: {vehicle.battery_capacity} kWh ({vehicle.usable_battery_pct:.0f}% usable)\n"
                f"Generator: {vehicle.generator_power_kw} kW  |  Fuel Tank: {vehicle.fuel_tank_gallons} gal  |  "
                f"BSFC: {vehicle.bsfc_g_kwh} g/kWh  |  Mode: {vehicle.erev_mode}"
            )
        elif vehicle.vehicle_class == 'gasoline':
            vehicle_text = (
                f"Mass: {vehicle.mass} kg  |  Cd: {vehicle.drag_coefficient}  |  "
                f"Fuel Tank: {vehicle.fuel_tank_gallons} gal  |  Fuel Economy: {vehicle.fuel_economy_mpg} MPG"
            )
        else:
            vehicle_text = f"Mass: {vehicle.mass} kg  |  Cd: {vehicle.drag_coefficient}"
        
        ttk.Label(info_frame, text=vehicle_text, justify=tk.CENTER).pack()
        
        # Test configuration
        config_frame = ttk.LabelFrame(main_frame, text="Range Test Configuration", padding="10")
        config_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Test mode selection
        ttk.Label(config_frame, text="Test Mode:", font=("Arial", 10, "bold")).grid(
            row=0, column=0, sticky=tk.W, pady=5, padx=5)
        
        self.test_mode = tk.StringVar(value="constant_70mph")
        
        # Different modes based on vehicle class
        if vehicle.vehicle_class == 'gasoline':
            modes = [
                ("Constant 70 mph", "constant_70mph"),
                ("Constant 75 mph", "highway_75mph"),
                ("EPA City (UDDS)", "UDDS"),
                ("EPA Highway (HWFET)", "HWFET"),
                ("EPA Combined (US06)", "US06"),
            ]
        else:
            modes = [
                ("Multi-Cycle Test (4 City + 2 Hwy + 2 Constant)", "multi_cycle"),
                ("Constant 70 mph (Range)", "constant_70mph"),
                ("Constant 75 mph", "highway_75mph"),
                ("EPA City (UDDS)", "UDDS"),
                ("EPA Highway (HWFET)", "HWFET"),
                ("EPA Combined (US06)", "US06"),
                ("Davis Dam Climb", "davis_dam_real"),
                ("I-70 Colorado Climb", "i70_colorado_climb")
            ]
        
        for i, (text, value) in enumerate(modes):
            ttk.Radiobutton(config_frame, text=text, variable=self.test_mode, 
                          value=value).grid(row=i+1, column=0, sticky=tk.W, padx=20, pady=2)
        
        # Starting SOC for EREV
        if vehicle.vehicle_class == 'erev':
            ttk.Label(config_frame, text="Starting SOC %:", font=("Arial", 10, "bold")).grid(
                row=1, column=1, sticky=tk.W, pady=5, padx=20)
            self.starting_soc_var = tk.StringVar(value=str(starting_soc))
            ttk.Entry(config_frame, textvariable=self.starting_soc_var, width=10).grid(
                row=2, column=1, sticky=tk.W, padx=40, pady=2)
        
        # Run button
        run_frame = ttk.Frame(main_frame)
        run_frame.pack(pady=10)
        ttk.Button(run_frame, text="Calculate Range", command=self.calculate_range,
                  width=20).pack()
        if vehicle.vehicle_class in ['bev', 'erev']:
            ttk.Button(run_frame, text="Save Range Report", command=self.save_range_report,
                      width=20).pack(pady=(6, 0))
        
        # Results display
        results_frame = ttk.LabelFrame(main_frame, text="Range Test Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Results table
        style = ttk.Style()
        style.configure("Results.Treeview", background="white", foreground="black", 
                       fieldbackground="white", rowheight=25)
        style.layout("Results.Treeview", [('Results.Treeview.treearea', {'sticky': 'nswe'})])
        
        columns = ('metric', 'value')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings',
                                        height=14, style="Results.Treeview")
        
        self.results_tree.heading('metric', text='Metric')
        self.results_tree.heading('value', text='Value')
        
        self.results_tree.column('metric', width=400, anchor=tk.W)
        self.results_tree.column('value', width=450, anchor=tk.W)
        
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial message
        self.results_tree.insert('', 'end', values=('Select a test mode and click "Calculate Range"', ''))
    
    def calculate_range(self):
        """Calculate range based on selected test mode and vehicle class."""
        try:
            cycle_name = self.test_mode.get()
            cycle_filepath = os.path.join("drive_cycles", f"{cycle_name}.csv")
            self.last_cycle_name = cycle_name
            
            # For multi-cycle, use a representative cycle for detailed plots
            if cycle_name == "multi_cycle":
                # Use constant_70mph as the representative cycle for plotting
                self.last_cycle_filepath = os.path.join("drive_cycles", "constant_70mph.csv")
            else:
                self.last_cycle_filepath = cycle_filepath
            
            # Check cycle file exists (except for multi_cycle)
            if cycle_name != "multi_cycle" and not os.path.exists(cycle_filepath):
                messagebox.showerror("Error", f"Drive cycle file not found: {cycle_filepath}")
                return
            
            # Dispatch based on vehicle class
            if self.vehicle.vehicle_class == 'gasoline':
                self.range_results = simulate_gasoline(self.vehicle, cycle_filepath)
                self.update_gasoline_results(cycle_name)
                
            elif self.vehicle.vehicle_class == 'erev':
                starting_soc = float(self.starting_soc_var.get())
                if cycle_name == "multi_cycle":
                    # For EREV multi-cycle, we run estimate_erev_range
                    self.range_results = estimate_erev_range(
                        self.vehicle, 
                        os.path.join("drive_cycles", "constant_70mph.csv"),
                        starting_soc
                    )
                else:
                    self.range_results = estimate_erev_range(
                        self.vehicle, cycle_filepath, starting_soc
                    )
                self.update_erev_results(cycle_name)
                
            else:  # BEV
                if cycle_name == "multi_cycle":
                    self.range_results = estimate_range_multi_cycle(self.vehicle)
                else:
                    self.range_results = estimate_range(self.vehicle, cycle_filepath)
                self.update_bev_results(cycle_name)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating range:\\n{str(e)}")

    def save_range_report(self):
        """Save range estimation report (BEV or EREV)."""
        if not self.range_results:
            messagebox.showwarning("No Results", "Run a range calculation first.")
            return

        try:
            cycle_name = getattr(self, 'last_cycle_name', 'range_test')
            cycle_filepath = getattr(self, 'last_cycle_filepath', 'drive_cycles/constant_70mph.csv')
            timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            run_output_dir = create_output_directory('outputs', f"{cycle_name}_range", timestamp)

            if self.vehicle.vehicle_class == 'erev':
                plot_path = generate_erev_range_plots(self.range_results, cycle_name, run_output_dir)
                
                # Generate detailed plots using the stored cycle filepath
                detailed_plot_path = None
                if os.path.exists(cycle_filepath):
                    detailed_plot_path = generate_erev_range_detailed_plots(
                        self.vehicle, cycle_filepath, self.range_results, cycle_name, run_output_dir
                    )
                
                save_erev_timeseries_csv(self.range_results, run_output_dir, filename="erev_range_timeseries.csv")

                summary_path = os.path.join(run_output_dir, "erev_range_summary.txt")
                r = self.range_results
                with open(summary_path, 'w') as f:
                    f.write("EREV Range Estimation Summary\n")
                    f.write(f"Cycle: {cycle_name}\n")
                    f.write(f"Range: {r['range_miles']:.2f} mi ({r['range_km']:.2f} km)\n")
                    f.write(f"EV-only miles: {r['ev_only_miles']:.2f} mi\n")
                    f.write(f"Generator miles: {r['generator_miles']:.2f} mi\n")
                    f.write(f"Fuel used: {r['fuel_used_gallons']:.3f} gal\n")
                    f.write(f"Final SOC: {r['final_soc']:.1f}%\n")
                    f.write(f"MPGe: {r['mpge']:.2f}\n")
                    f.write(f"Energy per mile: {r['kwh_per_mile']:.3f} kWh/mi\n")

                # Generate HTML report
                html_path = save_erev_range_html(
                    self.vehicle,
                    self.range_results,
                    cycle_name,
                    run_output_dir,
                    os.path.basename(plot_path),
                    os.path.basename(detailed_plot_path) if detailed_plot_path else None
                )

                files_list = [
                    f"  • {os.path.basename(html_path)}",
                    f"  • {os.path.basename(plot_path)}"
                ]
                if detailed_plot_path:
                    files_list.append(f"  • {os.path.basename(detailed_plot_path)}")
                files_list.extend([
                    "  • erev_range_timeseries.csv",
                    "  • erev_range_summary.txt"
                ])

                messagebox.showinfo("Saved", 
                    f"EREV range report saved to:\n{run_output_dir}\n\n"
                    f"Files:\n" + "\n".join(files_list))

            elif self.vehicle.vehicle_class == 'bev':
                plot_path = generate_bev_range_plots(self.range_results, cycle_name, run_output_dir)
                
                # Generate detailed plots using the stored cycle filepath
                detailed_plot_path = None
                if os.path.exists(cycle_filepath):
                    detailed_plot_path = generate_bev_range_detailed_plots(
                        self.vehicle, cycle_filepath, self.range_results, cycle_name, run_output_dir
                    )
                
                summary_path = os.path.join(run_output_dir, "bev_range_summary.txt")
                r = self.range_results
                with open(summary_path, 'w') as f:
                    f.write("BEV Range Estimation Summary\n")
                    f.write(f"Cycle: {cycle_name}\n")
                    f.write(f"Range: {r['range_miles']:.2f} mi ({r['range_km']:.2f} km)\n")
                    f.write(f"Energy per mile: {r['energy_per_mile']:.3f} kWh/mi\n")
                    f.write(f"Energy per km: {r['energy_per_km']:.3f} kWh/km\n")
                    f.write(f"Energy used: {r.get('energy_used_kwh',0):.2f} kWh\n")
                    f.write(f"Cycles completed: {r.get('cycles_completed', r.get('total_cycles', 0))}\n")
                    f.write(f"Final SOC: {r['final_soc']:.1f}%\n")

                # Generate HTML report
                html_path = save_bev_range_html(
                    self.vehicle,
                    self.range_results,
                    cycle_name,
                    run_output_dir,
                    os.path.basename(plot_path),
                    os.path.basename(detailed_plot_path) if detailed_plot_path else None
                )

                files_list = [
                    f"  • {os.path.basename(html_path)}",
                    f"  • {os.path.basename(plot_path)}"
                ]
                if detailed_plot_path:
                    files_list.append(f"  • {os.path.basename(detailed_plot_path)}")
                files_list.append("  • bev_range_summary.txt")

                messagebox.showinfo("Saved", 
                    f"BEV range report saved to:\n{run_output_dir}\n\n"
                    f"Files:\n" + "\n".join(files_list))
            else:
                messagebox.showinfo("Not Supported", "Range report export is available for BEV and EREV only.")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving range report:\n{str(e)}")
    
    def update_gasoline_results(self, cycle_name):
        """Update results for gasoline vehicle."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        r = self.range_results
        data = [
            ('Test Cycle', cycle_name),
            ('Fuel Tank', f"{self.vehicle.fuel_tank_gallons:.1f} gallons"),
            ('', ''),
            ('Distance Traveled', f"{r['distance_miles']:.1f} miles"),
            ('Fuel Used', f"{r['fuel_used_gallons']:.2f} gallons"),
            ('Fuel Economy', f"{r['mpg_achieved']:.1f} MPG"),
            ('', ''),
            ('Estimated Full-Tank Range', f"{r['range_miles']:.1f} miles"),
            ('Fuel Remaining', f"{r['fuel_remaining_gallons']:.2f} gallons"),
        ]
        
        self._populate_results(data)
    
    def update_erev_results(self, cycle_name):
        """Update results for EREV."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        r = self.range_results
        data = [
            ('Test Cycle', cycle_name),
            ('EREV Mode', self.vehicle.erev_mode),
            ('', ''),
            ('Total Range', f"{r['range_miles']:.1f} miles  ({r['range_km']:.1f} km)"),
            ('EV-Only Miles', f"{r['ev_only_miles']:.1f} miles"),
            ('Generator Miles', f"{r['generator_miles']:.1f} miles"),
            ('', ''),
            ('Battery Energy Used', f"{r['battery_energy_kwh']:.2f} kWh"),
            ('Fuel Used', f"{r['fuel_used_gallons']:.2f} gallons"),
            ('Final SOC', f"{r['final_soc']:.1f}%"),
            ('', ''),
            ('Efficiency (MPGe)', f"{r['mpge']:.1f} MPGe"),
            ('Energy per Mile', f"{r['kwh_per_mile']:.3f} kWh/mile"),
            ('Cycles Completed', f"{r['cycles_completed']}"),
        ]
        
        self._populate_results(data)
    
    def update_bev_results(self, cycle_name):
        """Update results for BEV."""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        r = self.range_results
        is_multi_cycle = 'cycle_breakdown' in r
        
        usable_kwh = self.vehicle.battery_capacity * (self.vehicle.usable_battery_pct / 100.0)
        
        if is_multi_cycle:
            data = [
                ('Test Type', 'Multi-Cycle Test (4 UDDS + 2 HWFET + 2 Constant 70mph)'),
                ('Battery Capacity', f"{self.vehicle.battery_capacity:.1f} kWh ({self.vehicle.usable_battery_pct:.0f}% usable = {usable_kwh:.1f} kWh)"),
                ('', ''),
                ('Estimated Range', f"{r['range_miles']:.1f} miles  ({r['range_km']:.1f} km)"),
                ('Energy Efficiency', f"{r['energy_per_mile']:.3f} kWh/mile  ({r['energy_per_km']:.3f} kWh/km)"),
                ('Miles per kWh', f"{1/r['energy_per_mile']:.2f} mi/kWh  ({1/r['energy_per_km']:.2f} km/kWh)"),
                ('', ''),
                ('Energy Used', f"{r['energy_used_kwh']:.2f} kWh"),
                ('Full Multi-Cycle Sequences', f"{r['total_cycles']}"),
                ('Final State of Charge', f"{r['final_soc']:.0f}%"),
                ('', ''),
                ('Cycle Breakdown:', ''),
            ]
            
            for cycle_type in ['UDDS', 'HWFET', 'constant_70mph']:
                breakdown = r['cycle_breakdown'][cycle_type]
                label = {'UDDS': 'City (UDDS)', 'HWFET': 'Highway (HWFET)', 'constant_70mph': 'Constant 70mph'}[cycle_type]
                data.append((f"  {label}", 
                           f"{breakdown['count']} cycles, {breakdown['distance_mi']:.1f} mi, {breakdown['energy_kwh']:.2f} kWh"))
        else:
            data = [
                ('Test Cycle', cycle_name),
                ('Battery Capacity', f"{self.vehicle.battery_capacity:.1f} kWh ({self.vehicle.usable_battery_pct:.0f}% usable = {usable_kwh:.1f} kWh)"),
                ('', ''),
                ('Estimated Range', f"{r['range_miles']:.1f} miles  ({r['range_km']:.1f} km)"),
                ('Energy Efficiency', f"{r['energy_per_mile']:.3f} kWh/mile  ({r['energy_per_km']:.3f} kWh/km)"),
                ('Miles per kWh', f"{1/r['energy_per_mile']:.2f} mi/kWh  ({1/r['energy_per_km']:.2f} km/kWh)"),
                ('', ''),
                ('Energy Used', f"{r['energy_used_kwh']:.2f} kWh"),
                ('Full Cycles Completed', f"{r['cycles_completed']}"),
                ('Final State of Charge', f"{r['final_soc']:.0f}%"),
            ]
        
        self._populate_results(data)
    
    def _populate_results(self, data):
        """Populate the results tree with data."""
        for metric, value in data:
            if metric == '':
                self.results_tree.insert('', 'end', values=('', ''), tags=('separator',))
            else:
                self.results_tree.insert('', 'end', values=(metric, value))
        
        self.results_tree.tag_configure('separator', background='#f0f0f0')


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = RoadLoadSimulatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
