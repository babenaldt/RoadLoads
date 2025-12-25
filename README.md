# Road Load Simulator

A Python-based vehicle road load simulation tool for analyzing power requirements, energy consumption, and vehicle dynamics across various drive cycles. Supports Battery Electric Vehicles (BEV), Extended Range Electric Vehicles (EREV), and gasoline vehicles.

## Overview

This simulator calculates the power and energy requirements for a vehicle traveling through a drive cycle with time-varying speed and grade. It accounts for aerodynamic drag, rolling resistance, road grade, vehicle acceleration, and powertrain-specific characteristics.

### Features

- **Graphical User Interface** - Easy-to-use GUI with live plotting and separate range estimator window
- **Multiple Powertrain Types**:
  - **Battery EV (BEV)**: Pure electric with regenerative braking, battery capacity, auxiliary loads
  - **Extended Range EV (EREV)**: Battery + gasoline generator with three operating modes
  - **Gasoline**: Traditional ICE vehicle with fuel economy modeling
- **Modular Design** - Separate physics model, GUI, and visualization components
- **External Preset Library** - Easy-to-edit JSON file for vehicle configurations
- **Multiple Drive Cycles** - EPA test cycles (UDDS, HWFET, US06, NYC), real-world climbs (Davis Dam, I-70 Colorado), constant speed tests (70/75 mph), and multi-cycle range testing
- **Range Estimation** - Dedicated interface for estimating vehicle range over various test cycles
- **EREV Operating Modes**:
  - **Charge Depleting**: Pure EV until SOC threshold, then generator sustains
  - **Blended**: Generator assists when power demand is high or SOC drops
  - **Hold**: Generator maintains SOC, battery covers power peaks
- **Real-time Visualization** - Interactive plots for speed, power, energy, grade, and SOC (for EREV)
- **Comprehensive Outputs** - Text summaries, plots, CSV data, HTML reports, and EREV power deficit analysis
- **Flexible Input** - Support for custom drive cycles, configurable SOC thresholds, BSFC, and vehicle parameters

## Quick Start

**Launch the GUI:**
```bash
python gui.py
```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage instructions.

---

## Mathematical Model

### Road Load Equation

The total tractive force required at the wheels is the sum of four components:

$$F_{total} = F_{aero} + F_{rolling} + F_{grade} + F_{accel}$$

---

### 1. Aerodynamic Drag Force

Air resistance opposing vehicle motion:

$$F_{aero} = \frac{1}{2} \rho \cdot C_d \cdot A \cdot v^2$$

Where:
- $F_{aero}$ = Aerodynamic drag force (N)
- $\rho$ = Air density (1.225 kg/m³ at sea level, 15°C)
- $C_d$ = Coefficient of drag (dimensionless)
- $A$ = Frontal area (m²)
- $v$ = Vehicle velocity (m/s)

**Physical Interpretation**: Drag force increases quadratically with speed. A vehicle traveling at 30 m/s (~67 mph) experiences 4× the drag force compared to 15 m/s (~34 mph).

---

### 2. Rolling Resistance Force

Friction between tires and road surface:

$$F_{rolling} = C_{rr} \cdot m \cdot g \cdot \cos(\theta)$$

Where:
- $F_{rolling}$ = Rolling resistance force (N)
- $C_{rr}$ = Rolling resistance coefficient (typically 0.008-0.015 for car tires)
- $m$ = Vehicle mass (kg)
- $g$ = Gravitational acceleration (9.81 m/s²)
- $\theta$ = Road grade angle (radians)
- $\cos(\theta) \approx 1$ for small grades

**Grade Conversion**: 
$$\theta = \arctan\left(\frac{\text{grade\%}}{100}\right)$$

For small grades (< 10%), $\cos(\theta) \approx 1$ is a reasonable approximation.

---

### 3. Grade Force (Gravitational Component)

Force due to climbing or descending:

$$F_{grade} = m \cdot g \cdot \sin(\theta)$$

Where:
- $F_{grade}$ = Grade force (N)
- $\theta$ = Road grade angle (radians)

**Sign Convention**: 
- Positive grade (uphill): $F_{grade} > 0$ (opposes motion)
- Negative grade (downhill): $F_{grade} < 0$ (assists motion)

**Approximation for small grades**:
$$\sin(\theta) \approx \frac{\text{grade\%}}{100} \quad \text{for } |\theta| < 10°$$

---

### 4. Acceleration Force (Inertial)

Force required to change vehicle velocity:

$$F_{accel} = m \cdot a$$

Where:
- $F_{accel}$ = Acceleration force (N)
- $a$ = Vehicle acceleration (m/s²)

Acceleration is calculated from the drive cycle speed profile:
$$a(t) = \frac{v(t+\Delta t) - v(t)}{\Delta t}$$

**Sign Convention**:
- Positive acceleration: Vehicle speeds up ($F_{accel} > 0$)
- Negative acceleration (braking): Vehicle slows down ($F_{accel} < 0$)

---

### Power Calculation

Instantaneous power required at the wheels:

$$P(t) = F_{total}(t) \cdot v(t)$$

Where:
- $P(t)$ = Power (Watts)
- $v(t)$ = Velocity at time $t$ (m/s)

**Unit Conversions**:
- 1 kW = 1000 W
- 1 hp = 745.7 W
- Power in kW: $P_{kW} = \frac{P_W}{1000}$
- Power in hp: $P_{hp} = P_W \times 0.00134102$

---

### Regenerative Braking

When braking ($P < 0$), regenerative braking recovers a portion of the kinetic energy:

$$P_{regen} = P_{braking} \times \eta_{regen}$$

Where:
- $P_{regen}$ = Regenerated power (W, negative value)
- $P_{braking}$ = Braking power demand (W, negative value)
- $\eta_{regen}$ = Regenerative braking efficiency (0.0 - 1.0, typically 0.7-0.9 for EVs)

**Efficiency Losses**: The $(1 - \eta_{regen})$ fraction is lost to:
- Electrical losses in motor/generator
- Battery charging inefficiency
- Mechanical losses

**Applied Formula**:

For traction (accelerating):
$$P_{applied}(t) = P(t) \quad \text{when } P(t) \geq 0$$

For braking (regenerating):
$$P_{applied}(t) = P(t) \times \eta_{regen} \quad \text{when } P(t) < 0$$

---

### Energy Calculation

Total energy consumption over a drive cycle:

$$E_{total} = \int_0^{T} P(t) \, dt$$

For discrete time steps:
$$E_{total} = \sum_{i=1}^{N} P(t_i) \cdot \Delta t$$

Where:
- $E_{total}$ = Total energy (Joules)
- $T$ = Total cycle duration (seconds)
- $\Delta t$ = Time step (seconds)

**Separated Energy Components**:

Traction energy (positive power only):
$$E_{traction} = \sum_{P(t_i) > 0} P(t_i) \cdot \Delta t$$

Regenerated energy (negative power only):
$$E_{regen} = -\sum_{P(t_i) < 0} P(t_i) \cdot \Delta t$$

Net energy:
$$E_{net} = E_{traction} - E_{regen}$$

**Unit Conversions**:
- 1 kWh = 3.6 × 10⁶ J
- 1 MJ = 10⁶ J
- Energy in kWh: $E_{kWh} = \frac{E_J}{3.6 \times 10^6}$

---

### Key Performance Metrics

**Peak Traction Power**:
$$P_{peak,traction} = \max_{t \in [0,T]} \{P(t) \mid P(t) > 0\}$$

**Peak Regenerative Power**:
$$P_{peak,regen} = \max_{t \in [0,T]} \{|P(t)| \mid P(t) < 0\}$$

**Average Power**:
$$P_{avg} = \frac{1}{T} \int_0^{T} P(t) \, dt = \frac{E_{net}}{T}$$

**Energy Recovery Percentage**:
$$\text{Recovery\%} = \frac{E_{regen}}{E_{traction}} \times 100\%$$

---

## Input Parameters

### Vehicle Parameters (BEV)
| Parameter | Symbol | Units | Typical Range |
|-----------|--------|-------|---------------|
| Mass | $m$ | kg | 1000 - 3000 |
| Frontal Area | $A$ | m² | 2.0 - 4.0 |
| Drag Coefficient | $C_d$ | - | 0.20 - 0.40 |
| Rolling Resistance | $C_{rr}$ | - | 0.008 - 0.015 |
| Regen Efficiency | $\eta_{regen}$ | - | 0.0 - 0.95 |
| Battery Capacity | - | kWh | 50 - 200 |
| Usable Battery | - | % | 85 - 95 |
| Auxiliary Power | - | W | 0 - 2000 |

### Additional EREV Parameters
| Parameter | Units | Typical Range | Description |
|-----------|-------|---------------|-------------|
| Generator Power | kW | 25 - 150 | Maximum generator output |
| BSFC | g/kWh | 220 - 300 | Brake Specific Fuel Consumption |
| Fuel Tank | gallons | 2 - 30 | Gasoline tank capacity |
| SOC Sustain Threshold | % | 15 - 25 | When to activate charge sustaining |
| SOC Blended Threshold | % | 25 - 40 | When to activate blended mode |

### Gasoline Vehicle Parameters
| Parameter | Units | Typical Range |
|-----------|-------|---------------|
| Fuel Tank | gallons | 10 - 30 |
| Fuel Economy | MPG | 15 - 40 |

### Drive Cycle Format
CSV file with three columns:
- `time` (seconds)
- `speed` (**must be in m/s** - meters per second)
- `grade` (percent, e.g., 6.0 for 6% grade)

**Note:** All EPA drive cycles (UDDS, HWFET, US06, etc.) are in m/s format. Results are displayed in both metric and imperial units, but all input parameters use metric units.

---

## Usage Example

### Battery EV Simulation
```python
from road_load_simulator import VehicleParams, calculate_road_load, load_drive_cycle

# Define BEV vehicle
vehicle = VehicleParams(
    mass=1847,                  # kg (e.g., Tesla Model 3)
    frontal_area=2.22,          # m²
    drag_coefficient=0.23,      # Cd
    rolling_resistance=0.01,
    vehicle_class='bev',
    regen_efficiency=0.75,      # 75% regen efficiency
    battery_capacity=75,        # kWh
    usable_battery_pct=90,      # 90% usable
    auxiliary_power=500         # 500W aux load
)

# Run simulation
cycle = load_drive_cycle('drive_cycles/UDDS.csv')
results = calculate_road_load(vehicle, cycle)

# Access results
print(f"Peak Power: {results.peak_power_traction / 1000:.1f} kW")
print(f"Total Energy: {results.total_energy / 3.6e6:.3f} kWh")
```

### EREV Simulation
```python
from road_load_simulator import VehicleParams, simulate_erev

# Define EREV vehicle (e.g., Scout Traveler EREV)
vehicle = VehicleParams(
    mass=2900,
    frontal_area=3.20,
    drag_coefficient=0.36,
    rolling_resistance=0.011,
    vehicle_class='erev',
    regen_efficiency=0.70,
    battery_capacity=50,        # kWh
    usable_battery_pct=90,
    auxiliary_power=800,
    generator_power_kw=100,     # 100 kW generator
    bsfc_g_kwh=250,             # Brake Specific Fuel Consumption
    fuel_tank_gallons=10,
    erev_mode='charge_depleting',
    soc_sustain_pct=20,         # Sustain at 20% SOC
    soc_blended_threshold_pct=30
)

# Run EREV simulation
results = simulate_erev(
    vehicle=vehicle,
    cycle_filepath='drive_cycles/UDDS.csv',
    starting_soc=100.0
)

print(f"EV-only miles: {results.ev_only_miles:.1f}")
print(f"Fuel used: {results.fuel_used_gallons:.2f} gal")
print(f"MPGe: {results.mpge:.1f}")
print(f"Power deficit events: {results.power_deficit_count}")
```

### Range Estimation
```python
from road_load_simulator import estimate_range, estimate_erev_range

# BEV range
bev_range = estimate_range(vehicle, 'drive_cycles/constant_70mph.csv')
print(f"BEV Range: {bev_range['range_miles']:.1f} miles")

# EREV range (uses both battery and fuel)
erev_range = estimate_erev_range(erev_vehicle, 'drive_cycles/constant_70mph.csv')
print(f"EREV Range: {erev_range['range_miles']:.1f} miles")
print(f"  EV portion: {erev_range['ev_only_miles']:.1f} miles")
print(f"  Generator portion: {erev_range['generator_miles']:.1f} miles")
```

---

## Output Files

Each simulation run creates a timestamped subfolder in `outputs/` containing:

1. **`summary.txt`** - Text summary with key metrics (SI and Imperial units)
2. **`plots.png`** - 4-panel figure showing:
   - Speed vs. Time
   - Power vs. Time (with traction/regen shading)
   - Cumulative Energy vs. Time
   - Grade vs. Time
3. **`timeseries.csv`** - Full time-series data with all calculated values
4. **`report.html`** - Interactive HTML report with embedded plots

---

## Available Drive Cycles

| Cycle | Description | Duration | Distance | Characteristics |
|-------|-------------|----------|----------|-----------------|
| **UDDS** | Urban Dynamometer Driving Schedule | ~23 min | ~7.5 mi | City driving, frequent stops |
| **HWFET** | Highway Fuel Economy Test | ~13 min | ~10.3 mi | Highway cruise, minimal stops |
| **US06** | Supplemental FTP | ~10 min | ~8.0 mi | Aggressive driving, high speed |
| **NYC** | New York City Cycle | ~10 min | ~1.2 mi | Dense urban, very low speed |
| **Davis Dam** | Real-world grade test | ~14 min | ~13.9 mi | Sustained 4-7% climb, 3000 ft gain |
| **I-70 Colorado** | Mountain climb | ~9 min | ~10 mi | Constant 7% grade at 65 mph |
| **Constant 70 mph** | Highway range test | ~8.5 min | ~10 mi | Flat highway cruise |
| **Constant 75 mph** | Highway range test | ~8 min | ~10 mi | Flat highway cruise |
| **Multi-Cycle Test** | EPA combined | Variable | Variable | 4 city + 2 highway + 2 constant speed |

---

## Dependencies

```bash
pip install numpy matplotlib
```

---

## Physical Validation

### Typical Results for a Compact EV (e.g., Tesla Model 3)
- **UDDS Cycle**:
  - Peak power: ~60-80 kW
  - Energy: ~2-3 kWh
  - Average power: ~8-10 kW
  
- **Highway (65 mph constant)**:
  - Steady-state power: ~15-20 kW
  - Dominated by aerodynamic drag ($F_{aero} \propto v^2$)

- **Davis Dam Climb**:
  - Peak power: ~50-70 kW (grade + aero)
  - Energy: ~12-18 kWh (long sustained climb)
  - Grade force dominates: $F_{grade} = m \cdot g \cdot \sin(\theta)$

### Rule of Thumb Estimates
**Power for steady speed on flat ground**:
$$P \approx \frac{1}{2} \rho C_d A v^3 + C_{rr} m g v$$

**Highway cruise (30 m/s ≈ 67 mph)**:
- Aero: ~12-18 kW (dominates at high speed)
- Rolling: ~3-4 kW
- Total: ~15-22 kW

**Climbing power (additional)**:
$$P_{climb} = m \cdot g \cdot v \cdot \sin(\theta) \approx m \cdot g \cdot v \cdot \frac{\text{grade\%}}{100}$$

For 2000 kg at 30 m/s on 6% grade:
$$P_{climb} \approx 2000 \times 9.81 \times 30 \times 0.06 \approx 35 \text{ kW}$$

---

## References

1. **SAE J2951** - Drive Quality Evaluation for Chassis Dynamometer Testing
2. **EPA Federal Test Procedure** - Vehicle emissions and fuel economy testing
3. **Gillespie, T.D.** - *Fundamentals of Vehicle Dynamics* (1992)
4. **Wong, J.Y.** - *Theory of Ground Vehicles* (2008)

---

## License

This project is provided as-is for educational and research purposes.

---

## Author

Generated with GitHub Copilot
Date: December 2025
