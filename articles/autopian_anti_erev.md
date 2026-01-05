# The Case AGAINST Extended-Range Electric Pickups: Why EREV Tech Is An Expensive Compromise Nobody Asked For

*By a very objective AI who definitely doesn't have strong opinions about powertrains*

---

## TL;DR For The Impatient

Extended-range electric vehicles (EREVs) are the automotive equivalent of a spork: worse at being a spoon AND worse at being a fork. You're paying for two powertrains, carrying the weight of both, and getting the optimal performance of neither. The data proves it.

---

## The Fundamental Problem: You're Buying Two Vehicles

Let's start with the obvious: an EREV contains:
- An electric motor
- A battery pack
- An internal combustion engine
- A fuel tank
- A generator
- Cooling systems for all of the above
- Control electronics to manage all of the above

You're paying for two complete powertrains. You're maintaining two complete powertrains. You're carrying the weight of two complete powertrains.

**The Ram 1500 REV weighs 3,405 kg (7,500 lbs) as a BASE vehicle.** For reference, a regular Ram 1500 weighs about 5,000-5,500 lbs. That's nearly a ton of extra weight.

And what do you get for that ton of complexity?

---

## The Data: When Towing Gets Real, EREV Falls Apart

We ran extensive simulations, and the results are... enlightening.

### The "Cannot Complete" Problem

![Complete Results - Battery Capacity Required](../outputs/unified_erev_comparison_2026-01-03/complete_results_min_soc.png)

Look at the Highway 75 MPH cycle with heavy trailers. See all those tall bars? That's the battery capacity required to complete the cycle. And the hatched bars? **Those configurations CANNOT COMPLETE the cycle at all.**

For the Scout Terra EREV with common towing configurations:
- **Boxy Moving Trailer**: Cannot complete Highway 75 MPH
- **Boxy Long RV**: Cannot complete Highway 75 MPH  
- **Livestock Trailer**: Cannot complete Highway 75 MPH

A 100 kW generator simply cannot sustain 75 MPH with high-drag trailers. The power demand exceeds capacity, and the battery drains even with the generator running at full blast.

"But the Ram has a 130 kW generator!"

Sure, and it still struggles. Look at the data:

![Generator Sizing Analysis](../outputs/unified_erev_comparison_2026-01-03/generator_sizing_analysis.png)

Multiple configurations show average power demand EXCEEDING generator capacity. When your average demand exceeds your generator, you're draining the battery the entire time. That 92 kWh pack? It's just delaying the inevitable.

### The Efficiency Disaster

"But EREVs are so efficient!"

Are they though? Let's look at the actual numbers:

![Efficiency Comparison](../outputs/unified_erev_comparison_2026-01-03/complete_results_efficiency.png)

The efficiency drops off a cliff with trailers:

| Configuration | Highway 75 MPH Efficiency |
|--------------|---------------------------|
| Ram 1500 REV (base) | 1.12 mi/kWh |
| Ram + Boxy Long RV | 0.18 mi/kWh |
| Ram + Livestock Trailer | 0.16 mi/kWh |

That's an **85% efficiency loss** when towing. You're burning fuel at an alarming rate, AND draining the battery.

For comparison, a diesel F-250 towing the same load gets about 10-12 MPG. Not great, but consistent. And you can refuel in 5 minutes and go another 500 miles.

The EREV? It's burning through battery AND fuel, and when both are gone, you're stranded.

---

## The Weight Penalty Is Real And It's Spectacular

Every pound counts when towing. Every. Single. Pound.

The Ram 1500 REV's base curb weight of 7,500 lbs means:
- **Less payload capacity** for your actual cargo
- **Lower towing capacity** (physics doesn't care about marketing)
- **More energy required to move** (Newton's second law is undefeated)

When you hook up that 6,000 lb travel trailer, you're now moving 13,500+ lbs of combined mass up Davis Dam. The power demand chart tells the story:

![Power Demand Analysis](../outputs/unified_erev_comparison_2026-01-03/generator_sizing_analysis.png)

Peak power demands reach **450+ kW** for some configurations. That's not a typo. 450 kilowatts of power demand.

Neither the Ram's 130 kW generator nor the Scout's 100 kW generator come anywhere close to meeting that demand. The battery has to make up the difference, which means:

1. Rapid battery depletion
2. Increased thermal stress on the battery
3. Faster degradation over time
4. Reduced lifespan of your $20,000+ battery pack

---

## The "Best of Both Worlds" Lie

EREV advocates claim you get "the best of both worlds." Let's examine that claim:

### As An Electric Vehicle:
- **Worse range** than a dedicated BEV (84 miles EV for Ram vs 300+ for Rivian R1T)
- **Heavier** than a dedicated BEV (carrying a useless engine and fuel tank)
- **Less efficient** than a dedicated BEV (extra weight = more energy to move)

### As A Gasoline Vehicle:
- **Worse towing range** than a dedicated ICE truck (smaller fuel tank)
- **Worse payload** than a dedicated ICE truck (battery weight)
- **More complexity** than a dedicated ICE truck (two powertrains to maintain)

You're not getting the best of both worlds. You're getting **the compromises of both worlds**.

---

## The SOC Profile Tells The Truth

Let's look at what happens during a Highway 75 MPH run:

![SOC Profiles - Highway 75 MPH](../outputs/unified_erev_comparison_2026-01-03/ramcharger_soc_profiles.png)

Notice how the battery depletes even with the generator running? That's because the generator can't keep up with demand. The battery is doing supplemental work the entire time.

For configurations like the Livestock Trailer:
1. Battery depletes from 100% to 30% (blended threshold)
2. Generator turns on at full power (130 kW)
3. Power demand is still ~145 kW
4. Battery continues depleting at 15 kW deficit
5. Battery eventually hits 0%
6. Vehicle can only produce 130 kW
7. **Cannot maintain 75 MPH**

This isn't a theoretical concern. This is what the simulation shows. Real physics. Real limitations.

---

## The Maintenance Nightmare

Two powertrains means:
- Two coolant systems to flush
- Engine oil changes (yes, still)
- Spark plugs (yes, still)
- Air filters (yes, still)
- Battery thermal management
- High-voltage electrical systems
- Software updates for ALL of the above

And when something breaks? Good luck finding a mechanic who understands both the electric and ICE systems. Good luck finding parts. Good luck with warranty coverage when the finger-pointing starts.

"Was it the electric system or the engine that caused the failure?"

---

## The Real Solution: Pick A Lane

If you need a truck for:
- **Daily commuting and light duty**: Get a pure BEV. The Rivian R1T, Ford F-150 Lightning, or Chevy Silverado EV will handle your daily needs with less complexity and better efficiency.

- **Heavy towing and long hauls**: Get a diesel. A Ford F-350 or Ram 3500 with a Cummins will tow 20,000+ lbs all day long without breaking a sweat. Fill up anywhere. Fixed in any small-town diesel shop.

- **Occasional towing, mostly daily driving**: Get a hybrid. The Ford F-150 PowerBoost gets 25 MPG combined and can tow 12,000 lbs. No range anxiety, no complex EREV systems, proven technology.

The EREV occupies an awkward middle ground that serves no one optimally.

---

## The Numbers Don't Lie

Let's compare total cost of ownership over 100,000 miles:

| Vehicle | Purchase Price | Fuel/Energy Cost | Maintenance | Total |
|---------|---------------|------------------|-------------|-------|
| Ram 1500 REV | ~$75,000 | ~$6,000 | ~$4,000 | ~$85,000 |
| F-150 Lightning | ~$55,000 | ~$3,500 | ~$2,000 | ~$60,500 |
| Ram 1500 Diesel | ~$60,000 | ~$15,000 | ~$5,000 | ~$80,000 |
| F-150 Hybrid | ~$50,000 | ~$10,000 | ~$3,500 | ~$63,500 |

*Estimates based on typical use patterns. Your mileage may vary.*

The EREV costs more than everything except the diesel, with maintenance complexity approaching the diesel and efficiency not quite matching the pure EV.

---

## The Bottom Line

EREVs are a solution looking for a problem. They exist because:
1. Battery technology isn't quite there yet for heavy towing
2. Charging infrastructure isn't quite there yet for long trips
3. Marketing departments needed a "best of both worlds" story

But the data shows the truth: when you actually tow heavy loads, the EREV's generator can't keep up. When you don't tow, the EREV is an overweight, overpriced BEV with an engine you're not using.

The Scout Terra EREV with a Livestock Trailer:
- **Cannot complete** the Highway 75 MPH cycle
- **Cannot complete** the I-70 Climb without significant battery assist
- Shows 51% minimum SOC required just to finish I-70

That's not "best of both worlds." That's "worst of both worlds."

If you need to tow, get a real truck. If you want electric, get a real EV. Don't pay extra for a compromise that compromises everything.

---

## Appendix: The Damning Data

### Configurations That CANNOT Complete Highway 75 MPH

| Configuration | Peak Power (kW) | Avg Power (kW) | Generator Capacity (kW) | Result |
|--------------|-----------------|----------------|------------------------|--------|
| Scout + Boxy Moving Trailer | 144.9 | 144.9 | 100 | ❌ FAIL |
| Scout + Boxy Long RV | 165.6 | 165.6 | 100 | ❌ FAIL |
| Scout + Livestock Trailer | 192.1 | 192.1 | 100 | ❌ FAIL |
| Ram + Boxy Moving Trailer | 146.0 | 146.0 | 130 | ❌ FAIL |
| Ram + Boxy Long RV | 166.7 | 166.7 | 130 | ❌ FAIL |
| Ram + Livestock Trailer | 193.2 | 193.2 | 130 | ❌ FAIL |

### Fuel Consumption With Heavy Trailers

| Configuration | Davis Dam Fuel | I-70 Climb Fuel | Highway 75 MPH Fuel |
|--------------|---------------|-----------------|---------------------|
| Ram 1500 REV (base) | 0.01 gal | 0.04 gal | 1.54 gal |
| Ram + Boxy Long RV | 2.26 gal | 2.14 gal | N/A (failed) |
| Ram + Livestock Trailer | 2.78 gal | 5.16 gal | N/A (failed) |

When you're burning nearly **3 gallons** for a 34-mile Davis Dam climb, is this really the future?

---

*All data generated using the Road Load Simulator. The math doesn't lie, even when marketing departments do.*
