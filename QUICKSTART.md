# Quick Start Guide

## Running the GUI

The easiest way to use the simulator is through the graphical interface:

```bash
python gui.py
```

### GUI Features

**Left Panel - Input Parameters:**

**Vehicle Class Selection:**
- Choose between Battery EV, EREV, or Gasoline vehicle types
- Parameter panel automatically adjusts to show relevant inputs for selected class

**Common Parameters (All Vehicle Types):**
- Mass (kg)
- Frontal Area (m²)
- Drag Coefficient (Cd)
- Rolling Resistance Coefficient (Crr)

**Battery EV Parameters:**
- Regen Efficiency (0.0 - 1.0)
- Auxiliary Power (W) - constant power drain for HVAC, electronics, etc.
- Battery Capacity (kWh)
- Usable Battery % (typically 85-95%)

**EREV Additional Parameters:**
- Generator Power (kW) - maximum output of gasoline generator
- BSFC (g/kWh) - Brake Specific Fuel Consumption (typical: 220-300)
- Fuel Tank (gallons)
- EREV Operating Mode:
  - **Charge Depleting**: Pure EV until SOC threshold, then generator sustains
  - **Blended**: Generator assists during high power or low SOC
  - **Hold**: Generator maintains SOC, battery handles peaks
- SOC Sustain % - when to activate charge sustaining mode (typical: 20%)
- SOC Blended % - when to start blended assist (typical: 30%)
- Starting SOC % - initial battery state of charge (typically 100%)

**Gasoline Parameters:**
- Fuel Tank (gallons)
- Fuel Economy (MPG)

**Drive Cycle Selection:**
- Dropdown with all available cycles
- Browse button for custom CSV files
- Speed unit selection (m/s or mph)

**Action Buttons:**
- **Run Simulation**: Execute single-cycle simulation with live plots
- **Estimate Range**: Open dedicated range estimator window
- **Save Results**: Export plots, CSV data, and HTML report
- **Load Preset**: Load pre-configured vehicle from external JSON file (filtered by vehicle class)

**Right Panel - Live Plots:**
- Speed vs. Time
- Power vs. Time (traction/regen shaded)
- Cumulative Energy vs. Time
- Road Grade vs. Time

---

## Range Estimator Window

Click "Estimate Range" to open a separate window for range testing:

**Vehicle Configuration Display:**
Shows current vehicle parameters for reference

**Test Mode Selection:**
- Multi-Cycle Test (4 UDDS + 2 HWFET + 2 Constant 70mph)
- Constant 70 mph (ideal for range testing)
- Constant 75 mph
- EPA City (UDDS)
- EPA Highway (HWFET)
- EPA Combined (US06)
- Davis Dam Climb (sustained grade)
- I-70 Colorado Climb (7% grade)

**Results Display:**
- **BEV**: Total range, energy efficiency (kWh/mi), mi/kWh, cycles completed, final SOC
- **EREV**: Total range, EV-only miles, generator miles, fuel used, MPGe, final SOC
- **Multi-Cycle**: Breakdown by cycle type with individual distances and energy

---

## Vehicle Presets Library

Presets are stored in `vehicle_presets.json` - easily editable without touching code!

### Battery EV Presets
- **Tesla Model 3**: 1847 kg, Cd 0.23, 75 kWh battery
- **Tesla Cybertruck**: 3104 kg, Cd 0.34, 123 kWh battery
- **Rivian R1T**: 3175 kg, Cd 0.32, 135 kWh battery
- **Rivian R1T + Camper**: 4990 kg, Cd 0.60, 135 kWh (towing config)
- **Silverado EV**: 4103 kg, Cd 0.44, 200 kWh battery
- **Scout Traveler BEV**: 2722 kg, Cd 0.36, 85 kWh battery

### EREV Presets
- **Scout Traveler EREV**: 2900 kg, 50 kWh battery, 100 kW generator, 10 gal tank
- **Ram 1500 Ramcharger**: 3500 kg, 92 kWh battery, 130 kW generator, 25 gal tank
- **BMW i3 REx**: 1395 kg, 42 kWh battery, 25 kW generator, 2.4 gal tank

### Gasoline Presets
- **Compact Sedan**: 1400 kg, Cd 0.28, 35 MPG, 13 gal tank
- **Midsize SUV**: 2000 kg, Cd 0.35, 26 MPG, 18 gal tank
- **Full-Size Truck**: 2700 kg, Cd 0.42, 20 MPG, 26 gal tank

**Adding Your Own Presets:**
Edit `vehicle_presets.json` and add your vehicle configuration. The GUI will automatically load it!

---

## Workflow Examples

### Example 1: Compare BEV Range on Different Cycles
1. Select "Battery EV" vehicle class
2. Load "Tesla Model 3" preset
3. Click "Estimate Range"
4. Test "Constant 70 mph" - note highway range
5. Test "UDDS" - note city range
6. Test "Davis Dam Climb" - note range with sustained grade

### Example 2: EREV Power Sizing Analysis
1. Select "EREV" vehicle class
2. Load "Scout Traveler EREV" preset
3. Modify generator power (try 50 kW, 100 kW, 150 kW)
4. Run simulation on "I-70 Colorado Climb"
5. Check results for power deficit events
6. If deficits occur, generator is undersized for that scenario

### Example 3: Multi-Cycle Range Test
1. Select vehicle class and load preset
2. Click "Estimate Range"
3. Select "Multi-Cycle Test"
4. Click "Calculate Range"
5. View combined range and breakdown by cycle type (city/highway/constant)

---

## Running from Command Line

For batch processing or scripting:

### BEV Simulation
```python
from road_load_simulator import VehicleParams, calculate_road_load, load_drive_cycle

# Define Battery EV
vehicle = VehicleParams(
    mass=1847,
    frontal_area=2.22,
    drag_coefficient=0.23,
    rolling_resistance=0.01,
    vehicle_class='bev',
    regen_efficiency=0.75,
    battery_capacity=75,
    usable_battery_pct=90,
    auxiliary_power=500
)

# Load cycle and run simulation
cycle = load_drive_cycle('drive_cycles/UDDS.csv', 'mps')
results = calculate_road_load(vehicle, cycle)

print(f"Peak Power: {results.peak_power_traction / 1000:.1f} kW")
print(f"Total Energy: {results.total_energy / 3.6e6:.3f} kWh")
```

### EREV Simulation
```python
from road_load_simulator import VehicleParams, simulate_erev

# Define EREV
vehicle = VehicleParams(
    mass=2900,
    frontal_area=3.20,
    drag_coefficient=0.36,
    rolling_resistance=0.011,
    vehicle_class='erev',
    regen_efficiency=0.70,
    battery_capacity=50,
    usable_battery_pct=90,
    auxiliary_power=800,
    generator_power_kw=100,
    bsfc_g_kwh=250,
    fuel_tank_gallons=10,
    erev_mode='charge_depleting',
    soc_sustain_pct=20,
    soc_blended_threshold_pct=30
)

# Run EREV simulation
results = simulate_erev(vehicle, 'drive_cycles/UDDS.csv', 'mps', starting_soc=100.0)

print(f"Total Distance: {results.total_distance_miles:.1f} mi")
print(f"EV Only: {results.ev_only_miles:.1f} mi")
print(f"Generator: {results.generator_miles:.1f} mi")
print(f"Fuel Used: {results.fuel_used_gallons:.2f} gal")
print(f"MPGe: {results.mpge:.1f}")
print(f"Power Deficits: {results.power_deficit_count} events")
if results.power_deficit_count > 0:
    print(f"Max Deficit: {results.max_power_deficit_kw:.1f} kW")
```

### Range Estimation
```python
from road_load_simulator import estimate_range, estimate_erev_range

# BEV range
range_results = estimate_range(vehicle, 'drive_cycles/constant_70mph.csv', 'mps')
print(f"BEV Range: {range_results['range_miles']:.1f} miles")
print(f"Efficiency: {range_results['energy_per_mile']:.3f} kWh/mi")

# EREV range
erev_range = estimate_erev_range(erev_vehicle, 'drive_cycles/constant_70mph.csv', 'mps')
print(f"EREV Total Range: {erev_range['range_miles']:.1f} miles")
print(f"  EV portion: {erev_range['ev_only_miles']:.1f} miles")
print(f"  Generator portion: {erev_range['generator_miles']:.1f} miles")
print(f"  Fuel used: {erev_range['fuel_used_gallons']:.2f} gallons")
```

---

## Creating Custom Drive Cycles

Drive cycle CSV format:
```csv
time,speed,grade
0,0.0,0.0
1,5.0,0.0
2,10.0,1.5
3,15.0,2.0
...
```

- **time**: Elapsed time in seconds (monotonically increasing)
- **speed**: Vehicle speed in m/s (or mph if speed_unit='mph' specified)
- **grade**: Road grade in percent (positive = uphill, negative = downhill)

Place your CSV in the `drive_cycles/` folder and it will automatically appear in the GUI dropdown.

**Example: Creating a custom highway cycle**
```python
import csv
import numpy as np

# 10 minutes at constant 75 mph (33.53 m/s) on flat road
time = np.arange(0, 601, 1)  # 0 to 600 seconds
speed = np.full(len(time), 33.53)  # constant 75 mph
grade = np.zeros(len(time))  # flat

with open('drive_cycles/my_highway_cycle.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time', 'speed', 'grade'])
    for t, s, g in zip(time, speed, grade):
        writer.writerow([t, s, g])
```

---

## Output Files

Each simulation creates a timestamped subfolder in `outputs/` with:

### Regular Simulation
- **`summary.txt`** - Text summary with all metrics (power, energy, distance, etc.)
- **`plots.png`** - 4-panel visualization:
  - Speed profile
  - Power demand (with traction/regen shading)
  - Cumulative energy
  - Road grade profile
- **`timeseries.csv`** - Full time-series data with all calculated values for further analysis
- **`report.html`** - Interactive HTML report with embedded plots and formatted results

### Range Analysis
For **BEV range** estimations:
- **`bev_range_report.html`** - Interactive HTML report with range metrics, efficiency data, and embedded plots
- **`bev_range_plots.png`** - Summary bar charts (range and energy)
- **`bev_range_detailed_plots.png`** - Detailed 4-panel plots (speed, power, energy, grade vs distance)
- **`bev_range_summary.txt`** - Text summary

For **EREV range** estimations:
- **`erev_range_report.html`** - Comprehensive HTML report with EV/generator breakdown, fuel consumption, and embedded plots
- **`erev_range_plots.png`** - Multi-panel plots (SOC, generator output, fuel remaining, distance)
- **`erev_range_detailed_plots.png`** - Detailed 4-panel plots (speed, power, energy, grade vs distance)
- **`erev_range_timeseries.csv`** - Time-series data with SOC and fuel levels
- **`erev_range_summary.txt`** - Text summary

Example output location: `outputs/UDDS_2025-12-24_143022/`

**For EREV simulations**, results include additional metrics:
- EV-only distance vs generator-assisted distance
- Fuel consumption and efficiency (MPGe)
- SOC timeline
- Power deficit summary (count and maximum shortfall)

---

## Vehicle Presets

### Tesla Model 3
- Mass: 1847 kg
- Frontal Area: 2.22 m²
- Cd: 0.23
- Crr: 0.01
- Regen: 75%

### Compact Car
- Mass: 1200 kg
- Frontal Area: 2.0 m²
- Cd: 0.30
- Crr: 0.011
- Regen: 0% (no regenerative braking)

### SUV
- Mass: 2200 kg
- Frontal Area: 3.0 m²
- Cd: 0.35
- Crr: 0.012
- Regen: 0%

### Truck
- Mass: 2500 kg
- Frontal Area: 3.5 m²
- Cd: 0.40
- Crr: 0.013
- Regen: 0%

---

## Troubleshooting

**GUI won't start:**
- Ensure tkinter is installed: `python -m tkinter`
- On Linux, you may need: `sudo apt-get install python3-tk`

**Drive cycle not found:**
- Verify the CSV file is in the `drive_cycles/` folder
- Check that the file format matches the expected structure (time, speed, grade columns)

**Invalid parameters:**
- All numeric fields must contain valid numbers
- Regen efficiency must be between 0.0 and 1.0
- Mass, frontal area, and coefficients must be positive
- BSFC typically ranges 220-300 g/kWh
- SOC thresholds should be: sustain < blended < starting

**Presets not loading:**
- Ensure `vehicle_presets.json` exists in the project root directory
- Check JSON syntax is valid (use a JSON validator if needed)
- Verify `vehicle_class` field matches: "bev", "erev", or "gasoline"

**EREV power deficits:**
- This means generator can't provide enough power for the drive cycle
- Solutions:
  - Increase `generator_power_kw`
  - Reduce vehicle mass or drag
  - Use a less demanding drive cycle
  - Switch to "blended" mode for better power distribution

**Range estimator shows unexpected results:**
- Verify starting SOC is set correctly (typically 100%)
- Check usable battery percentage (typically 90%)
- For EREV, ensure fuel tank has capacity
- Multi-cycle tests may take longer to compute

---

## Tips

- **For quick comparisons**: Use the GUI to run multiple simulations with different parameters
- **For batch analysis**: Use the command-line interface with Python scripts
- **Custom cycles**: Create your own drive cycles based on GPS data or desired test conditions
- **Real-world validation**: Compare results with vehicle specifications and real-world measurements
- **EREV generator sizing**: Run steep grade cycles and check for power deficit events to validate generator is adequately sized
- **Range testing**: Use the Multi-Cycle Test for EPA-style combined range estimates
- **Edit presets easily**: Modify `vehicle_presets.json` to add your own vehicle configurations
- **Operating modes**: 
  - Use "Charge Depleting" for maximum EV range before generator kicks in
  - Use "Hold" mode to preserve battery for later (e.g., save EV for city driving after highway)
  - Use "Blended" mode for optimal power distribution on demanding routes

---

## Understanding EREV Results

**Power Deficit Events:**
- Indicates instances where power demand exceeds (generator + battery capability)
- If deficits occur, consider:
  - Increasing generator power
  - Reducing vehicle mass/drag
  - Avoiding that specific drive cycle profile
  
**MPGe (Miles Per Gallon Equivalent):**
- Accounts for both electrical energy (from battery) and fuel energy
- 33.7 kWh = 1 gallon gasoline equivalent
- Higher is better

**SOC Management:**
- Watch final SOC to see if battery is fully depleted
- Adjust SOC thresholds to optimize EV vs generator usage
- Lower sustain threshold = more EV miles, but risks power deficits on steep grades

---

## Advanced Features

**Auxiliary Power Drain:**
Constant power draw for HVAC, lights, electronics. Typical values:
- Compact car: 300-500 W
- Midsize sedan: 500-800 W
- Truck/SUV: 800-1500 W
- With camper/accessories: 1500-3000 W

**Usable Battery Percentage:**
Most EVs protect battery health by limiting usable capacity:
- Conservative: 85% (longer battery life)
- Typical: 90%
- Aggressive: 95% (maximum range)

**BSFC Values:**
Generator efficiency varies by technology:
- Modern direct-injection ICE: 220-250 g/kWh
- Older port-injection ICE: 250-280 g/kWh
- Small range extenders: 260-300 g/kWh
