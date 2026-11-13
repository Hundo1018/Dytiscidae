# A14 — Power survey: sustained sources, burst sources for a crossing, run-ups, and drives other than motor and crank

Survey date 2026-11-13. Web search and page fetches only; no project code, no simulation. Follows
`A13_hardware.md` (cells, drives, reference machines) and `A11_flapping_design.md` (flapping machines, one wing
in two media); rows those reports hold are cited as "A13" / "A11" and not repeated. Written for decisions 2 and 6
of `experiments/flapper/PLAN.md` ("The user's decisions"): only the first take-off may be assisted, and the
power source and the drive are open.

The machine in view: 0.3–0.85 kg, about 1 m span, up to about 5.5 Hz, 15 minutes per fill in each medium, fresh
water, built at home in Taiwan from parts bought online.

Labels, per number (A13's, plus the class of the source):

- **VERIFIED** — seen today in the fetched page or PDF (PDFs the fetch tool could not read were read with
  `pdftotext`). Followed by the source class: *paper* (peer-reviewed), *maker* (manufacturer sheet or manual),
  *test* (magazine bench test), *press*, *shop*, *forum*.
- **SNIPPET** — only in a search-result summary or a secondary page; same class tags.
- **INFERRED** — arithmetic of this report on labelled numbers; formula and inputs are shown.
- **UNSOURCED** — general knowledge that no page fetched today confirms. Used twice, marked each time.
- **UNKNOWN** — not found; what was tried is in the last section.

"Demonstrated" below means a machine did it and the mass is given. "Arithmetic" means nothing more than that
the numbers allow it.

---

## 0. What a crossing needs, as arithmetic (INFERRED)

Minimum flight speed. The brief sets 6–9 m/s (A11: FAAV flies ≈ 6 m/s; A13: Robo Raven 6.7 m/s). Other values
found: E-Flap flies at 2–6 m/s at 0.51 kg (A13, VERIFIED paper); RoboFalcon 2.0 reached 4–6 m/s at 0.8 kg
(C.1 below); the 100 g fixed-wing AquaMAV needs 8.5 m/s (B.2). The table uses 6 and 9 m/s; every entry scales
linearly (impulse) or quadratically (energy) to another speed.

Impulse J = m·v, kinetic energy E = ½·m·v², equivalent vertical jump height h = v²/(2g), g = 9.81 m/s². Losses
(gravity during the push, drag, water leaving with the body) are not included, so these are floors.

| body mass | J at 6 m/s | J at 9 m/s | E at 6 m/s | E at 9 m/s |
|---|---|---|---|---|
| 0.3 kg | 1.8 N·s | 2.7 N·s | 5.4 J | 12.2 J |
| 0.5 kg | 3.0 N·s | 4.5 N·s | 9.0 J | 20.3 J |
| 0.8 kg | 4.8 N·s | 7.2 N·s | 14.4 J | 32.4 J |

h = 1.83 m for 6 m/s, 4.13 m for 9 m/s. In electrical terms 32.4 J is 0.009 Wh: **the energy of a crossing is
nothing; what is scarce is force and the hardware that delivers it in under a second.**

---

## A. Sustained sources (15 minutes)

### A.1 Lithium cells other than the A13 baseline

A13's bound: a high-drain 21700 (Molicel P45B) delivers 195 Wh/kg at the cell and 182 Wh/kg in a welded pack on
a 15-minute (4C) discharge, and 1.9 kW/kg at its continuous limit.

| item | numbers | URL | label |
|---|---|---|---|
| Molicel INR-21700-P50B (successor of the P45B) | 18.0 Wh typical, 17.5 Wh minimum; 71 g maximum; 60 A continuous (the sheet ties current limits to a temperature cut-off); DC impedance 12.8 mΩ at 50% charge; stated 260 Wh/kg | https://www.fsaeonline.com/content/Molicel-TDS--INR-21700-P50B-80122.pdf | VERIFIED maker |
| P50B capacity and price | 5.0 Ah typical, 4.85 Ah minimum; US$22.99 per cell at one US shop; sold by FPV shops in pairs | https://imrbatteries.com/products/molicel-21700-p50b-5000mah-60a-battery ; https://www.getfpv.com/molicel-p50b-21700-5000mah-60a-3-6v-li-ion-battery-2pcs.html | SNIPPET shop |
| P50B at 4C | 18.0 / 0.071 = 254 Wh/kg at low rate. At 4C = 20 A: V = 3.6 − 20 × 0.0128 = 3.34 V. Capacity at 20 A is not on the sheet's text; borrowing the P45B's ratio (4.10/4.5 = 0.911): 4.56 Ah × 3.34 V = 15.2 Wh = **214 Wh/kg**, 10% above the P45B's 195. Pack at A13's 0.96: 206 Wh/kg | | INFERRED (capacity ratio borrowed) |
| Upgrade Energy "Red V4" 4S1P 5 Ah pack | 250 Wh/kg stated; 25 A continuous, 50 A for 10 s. 5 Ah × 14.4 V = 72 Wh, hence ≈ 288 g, which is four 71 g cells: this is a P50B-class 21700 pack, not a new chemistry | https://www.unmannedsystemstechnology.com/company/upgrade-energy/red-v4-4s1p-5ah-xt60/ (403 on fetch) | SNIPPET maker; mass INFERRED |
| FPV LiPo pouch pack, ManiaX 4S 1500 mAh | 22.2 Wh, 185 g (reviewer's scale) = 120 Wh/kg; the 1300 mAh version 19.24 Wh, 159 g = 121 Wh/kg | https://www.firstquadcopter.com/reviews/maniax-power-4s-fpv-lipo-review/ | SNIPPET press |
| FPV LiPo, Tattu R-Line 6S 1300 mAh (V4 130C, V6 160C) | 203 g ± 20 g; 1.3 Ah × 22.2 V = 28.9 Wh = 142 Wh/kg; the maker's reseller states 160 Wh/kg for the V6 | https://www.racedayquads.com/products/tattu-r-line-version-4-0-22-2v-6s-1300mah-130c-lipo-battery-xt60 ; https://gensace.de/products/taa13006s16st | SNIPPET shop |
| FPV LiPo, other 4S 1500 mAh packs | 157–209 g for 22.2 Wh: 106–141 Wh/kg | search summary of shop listings | SNIPPET shop |
| FPV LiPo, power | Nameplate: 160C × 1.3 Ah = 208 A × 22.2 V = 4.6 kW from 203 g = 23 kW/kg. Vendor C ratings are not test results; no independent test at a stated C rate was found | | INFERRED on shop figures |
| Tattu / Grepow semi-solid packs ("300 Wh/kg" series) | 6S 22 Ah: ≈ 1807 g, €433; 6S 30 Ah: ≈ 2415 g, €591. 22 Ah × 22.2 V = 488 Wh / 1.807 kg = **270 Wh/kg at the pack**; rated 5C–10C | https://gensace.de/blogs/news/tattu-semi-solid-batteries-four-energy-density-levels-and-flexible-customization ; https://www.nextfpv.com/products/tattu-semi-solid-state-300wh-kg-22000mah-5c-22-2v-6s1p-g-tech-lipo-battery-pack-with-xt90s-f | SNIPPET shop; Wh/kg INFERRED |
| Semi-solid, smallest pack found | 4S 8 Ah, 400 g, 296 Wh/kg stated, **1C continuous**: 118 Wh, six times the machine's need, and too slow for a 15-minute discharge. Listings of the larger packs were mostly "sold out" | https://grepow.com/semi-solid-state-battery/300wh-kg-series-high-energy-density-battery-pack.html | SNIPPET maker |
| Amprius SiCore (silicon anode) | up to 400 Wh/kg; a 370 Wh/kg high-power cell rated 10C without cooling; shipped to drone manufacturers from a pilot line; no public price per cell | https://electrek.co/2025/01/14/amprius-unveils-high-power-sicore-cell-for-evs-drones-more/ ; https://amprius.com/amprius-ships-high-performance-sicore-cells-to-multiple-drone-customers-from-u-s-pilot-line/ | SNIPPET press |
| Amprius cells in a retail pack | Upgrade Energy lists a "Red V4 4S1P 5 Ah Amprius SA124" pack; mass, rate, price not read (403) | https://www.unmannedsystemstechnology.com/company/upgrade-energy/red-v4-4s1p-5ah-amprius-sa124/ | UNKNOWN beyond the title |

Read of A.1 (INFERRED):

- **Pouch LiPo is worse than the baseline on energy, not better.** FPV packs measure 106–142 Wh/kg at the pack
  against 182 Wh/kg for welded P45B at 4C. What LiPo buys is power (tens of kW/kg on the nameplate against
  1.9 kW/kg) and a free shape. For 15 minutes at 4C the high-drain cylindrical cell stays the better sustained
  source.
- **One step is buyable today: the P50B**, about +10% at 4C (214 against 195 Wh/kg at the cell), at about US$23.
- **Nothing clearly better is buyable in a 20–80 Wh size.** Semi-solid packs reach 270 Wh/kg at the pack but
  start at 118 Wh and 1C; the 5C–10C versions start near 490 Wh (1.8 kg). Silicon-anode cells at 370–400 Wh/kg
  exist and are sold to manufacturers; a hobby channel was not confirmed.

### A.2 Small combustion engines

| item | numbers | URL | label |
|---|---|---|---|
| Cox Tee Dee .049 (0.819 cm³ glow two-stroke) | peak 0.105 bhp (78 W) at 22,000 rpm on about 30% nitromethane; ran on test propellers from about 12,000 to beyond 21,000 rpm; "considerable kick-back" on hand starting | https://coxengines.ca/cox/www.mh-aerotools.de/airfoils/documents/cox_test_teedee_049_aeromodeller_november_1962.pdf | VERIFIED test (Aeromodeller, 1962) |
| Tee Dee .049 mass | 1.48 oz = 42 g | https://mh-aerotools.de/airfoils/documents/cox_teedee_15_049_instructions.pdf | VERIFIED maker |
| Cox Black Widow .049 (reed valve, integral tank) | peak "nearly .08 b.h.p." (≈ 60 W) at 15,000–15,550 rpm; maximum torque just over 6 oz·in (0.042 N·m) near 9,000 rpm; 64 g checked weight with tank; 0.55 bhp/lb as tested; 15% nitromethane | https://www.mh-aerotools.de/airfoils/documents/cox_test_black_widow_aeromodeller_august_1974.pdf | VERIFIED test (Aeromodeller, 1974) |
| Cox Tee Dee .010 (0.16 cm³) | 13 g without tank, 15.3 g with; 27,400 rpm on the maker's 3 in propeller; power: the testers could not measure torque at this size | https://www.mh-aerotools.de/airfoils/documents/cox_test_teedee_010_aeromodeller_october_1979.pdf | VERIFIED test (Aeromodeller, 1979) |
| O.S. MAX-15LA (2.49 cm³ glow two-stroke, radio-control carburettor) | 0.31 kW (0.41 hp) at 17,000 rpm; 138 g; **2,500–18,000 rpm**; €100–115 | https://lindinger.at/en/AIRPLANES/GAS-GLOW-POWERPLANTS/Engines/OS-MAX-15LA-RC-WITH-SILENCER/34660 | VERIFIED shop (maker's figures) |
| O.S. MAX-10LA (1.76 cm³) | manual suggests a 7×4 propeller and a 70 cm³ tank; power and mass not read | https://manualshive.com/en/manual/o-s-engine/max-10la/3120672?page=9 | SNIPPET maker |
| Smallest petrol engine found, NGH GT9 Pro (9 cm³) | 1.2 hp (895 W) at 13,000 rpm; 435 g plus a 100 g ignition unit | https://motionrc.com/products/ngh-gt9-pro-9cc-two-stroke-engine-ngh-gt9pro | SNIPPET shop |
| Petrol engine against the machine | 535 g of engine and ignition is 63–178% of a 0.3–0.85 kg machine. No petrol engine under 9 cm³ was found | | INFERRED |
| Fuel use, sport .049 on its integral tank | 5 cm³ in about 2 min; 8 cm³ in about 3.5 min (same owner, same propeller); others report 3–4 min on the 5 cm³ tank | https://coxengineforum.com/t13718-run-time-for-babebee-049 ; https://www.rcuniverse.com/forum/glow-engines-114/1010131-49-engine-run-times.html | SNIPPET forum |
| Fuel use, rule of thumb | two-strokes at full throttle: 1.5 fl oz per minute per cubic inch of displacement | https://www.rcuniverse.com/forum/glow-engines/4753706-fuel-consumption-print.html | SNIPPET forum |
| Fuel use, derived | .049: 5/2 to 8/3.5 = 2.3–2.5 cm³/min measured; the rule gives 0.049 × 1.5 × 29.6 = 2.2 cm³/min. .15: 0.15 × 1.5 × 29.6 = 6.7 cm³/min. With fuel at 0.88 g/cm³ (65% methanol 0.79, 15% nitromethane 1.14, 20% castor oil 0.96; densities are handbook values, UNSOURCED today): **.049 ≈ 2.0–2.2 g/min, .15 ≈ 5.9 g/min at full throttle** | | INFERRED on forum figures |
| Specific fuel consumption, derived | .049 at 60 W: 2.1 × 60 / 0.060 = 2,100 g/kWh. .15 at its rated 310 W: 5.9 × 60 / 0.31 = 1,140 g/kWh. Shaft energy per kg of fuel: 480–880 Wh/kg. A P45B pack through a drive of 65% efficiency (E-Flap: 130 W mechanical estimated from 200 W electrical, A13): 182 × 0.65 = 118 Wh/kg. **Fuel carries 4–7.5 times the shaft energy per kilogram, not an order of magnitude** | | INFERRED |
| Why so little | Seven glow two-strokes of 15–450 g (0.16–7.5 cm³) on a dynamometer: incomplete combustion is 60–70% of the energy loss, then heat transfer, exhaust enthalpy, friction; the thesis names 15% overall efficiency as a target, not a result | https://drum.lib.umd.edu/items/a1920304-5162-42cd-9b2a-01904ee6dcd6 | VERIFIED paper (thesis abstract; tables not read) |
| Throttle | 15LA: 2,500–18,000 rpm by carburettor (above). Tee Dee .049 as tested: no throttle; speed was set by the propeller load. A Tee Dee .010 with an added throttle idled at 4,400 and peaked at 27,200 rpm | 15LA and Aeromodeller URLs above; https://library.modelaviation.com/article/engine-shop-200506 | VERIFIED / SNIPPET press |
| Restart | Glow engines are started by spinning the shaft with the plug heated from a battery (Tee Dee: by hand or spring starter). An on-board restart system at this size: UNKNOWN | Aeromodeller URLs above | VERIFIED test for the method |
| Water | A two-stroke under water takes water into the cylinder; liquid does not compress, so cranking can bend a rod (marine source). Model-boat owners report restarting a glow engine after less than 30 s under water, and after about 2 min with a drain-and-flush routine by hand; one maker's manual: hold the boat bow-up to drain the exhaust, restart, run 30 s. Nobody reports a restart without hands | https://boatingmag.com/boats/how-to-prevent-hydrolock/ ; https://www.rcuniverse.com/forum/speed-rc-nitro-boats-117/9637946-hydro-lock-question.html ; https://manualshive.com/en/manual/arrow-shark/x-hydro/2976660?page=21 | SNIPPET press / forum / maker |
| Vibration of a single-cylinder engine on a flapping airframe | UNKNOWN (no measurement found) | — | UNKNOWN |
| CO₂ expansion motors (free-flight class) | Gasparin GMW 160: 1.57 cm³, 16.8 g without propeller. A small Gasparin motor at part throttle runs "more than 10 minutes" in models whose propeller and filled tank weigh a quarter of an ounce (7 g). Shaft power of any of them: UNKNOWN | https://craftsmanshipmuseum.com/exhibit/gasparin-gmw-160-co2-motor/ ; https://library.modelaviation.com/article/engine-shop-200303 | SNIPPET press |
| CO₂ as a sustained source, ceiling | Ideal isothermal work of gas expanding from 57 bar to 1 bar: R·T·ln(57) = 188.9 × 293 × 4.04 = 224 kJ/kg = 62 Wh/kg of gas. A 12 g cartridge: 0.75 Wh at most, in a 41–43.5 g steel body (B.1) = 17–18 Wh per kg carried: a tenth of the battery | | INFERRED (ideal gas) |
| Compressed air | air at 300 bar: about 50 Wh per litre and 372 g per litre, i.e. 134 Wh per kg of air before the tank; a tank-inclusive figure and a motor of this size: UNKNOWN | https://newmars.com/forums/viewtopic.php?id=10883 | SNIPPET forum |

Engines in ornithopters:

| item | numbers | URL | label |
|---|---|---|---|
| Lippisch's students, 1930s | engine-powered model ornithopters, "flights over 16 minutes"; mass and span not given | https://ornithopter.org/mobile/unmanned.html | VERIFIED press (society history page) |
| Percival Spencer's "Seagulls", 1950s | engine-driven, from a model on an .02 cu in (0.33 cm³) engine to one of 8 ft span; mass and duration not given | same | VERIFIED press |
| Spencer Orniplane, 1961 | 2.3 m span, 3.4 kg, 5.7 cm³ two-stroke, radio-controlled biplane layout (to cut fuselage oscillation); duration not given | https://en.wikipedia.org/wiki/Ornithopter | SNIPPET |
| Trouvé, 1890 | twelve gunpowder charges fired into a Bourdon tube; flew 70 m | https://ornithopter.org/mobile/unmanned.html | VERIFIED press |
| An engine ornithopter under 1 kg with both mass and duration stated | not found | — | UNKNOWN |

System mass for 15 minutes of shaft power (INFERRED; every assumption is in the row):

| shaft power | engine system | battery system | difference |
|---|---|---|---|
| inputs | Engine mass from the table. Fuel between "proportional to power" (lower) and "full-throttle rate throughout" (upper). Tank 10 g (**assumed**; no tank mass was found). Reduction 51 g (E-Flap transmission: 71 g less the 20 g motor, A13); a 17,000–22,000 rpm engine needs about 55:1 to 67:1 for 5.5 Hz (330 rpm), inside A13's 24:1–97.8:1 | Electrical energy = P × 0.25 h / 0.65; pack at 182 Wh/kg (A13); motor 20 g, ESC 5 g, the same 51 g reduction | |
| 50 W | Tee Dee .049 (78 W peak): 42 + fuel 2.1 g/min × 15 = 32 g + 10 + 51 = **135 g** | 19.2 Wh → 106 g + 76 g = **182 g** (A13's 3S minimum of 138 g of cells would make it 220 g) | engine −47 g |
| 100 W | 15LA (310 W rated): 138 + fuel 28–88 g + 10 + 51 = **227–287 g** | 38.5 Wh → 211 g + 76 g = **287 g** | engine 0 to −60 g |
| 150 W | 15LA: 138 + fuel 43–88 g + 10 + 51 = **242–287 g** | 57.7 Wh → 317 g + 76 g = **393 g** | engine −106 to −151 g |

With P50B cells at 206 Wh/kg the battery column becomes 169, 263 and 356 g.

Read of A.2 (INFERRED):

- A hobby glow engine of the right size exists and is cheap (42–138 g, 60–310 W). At 15 minutes its fixed
  mass eats most of the fuel advantage: about 50 g saved at 50 W, 0–60 g at 100 W, 100–150 g at 150 W.
- The engine does not run under water, so a machine that also swims carries an electric drive and cells for
  the water leg as well (A13: swimming is 3–16% of the flight energy). That second drive is at least the 76 g
  of motor, controller and reduction unless the two share one gearbox, which nothing found has done.
- No engine-powered ornithopter under 1 kg with a stated mass and duration was found, no on-board restart, no
  hands-off recovery from immersion, and no vibration measurement. Each is an open bench question, not a
  parts-list question.

### A.3 Anything else at this scale

| item | numbers | URL | label |
|---|---|---|---|
| Rubber | Specific energy of a wound rubber motor: about 6–9 kJ/kg, i.e. about 2 Wh/kg (**UNSOURCED**: general knowledge, the one search returned no figure). At 2 Wh/kg the 22 Wh that 15 minutes of flight takes (A13) would be 11 kg of rubber. The indoor ornithopter duration record of 21 min 44 s (Roy White) belongs to models of about a gram (class mass not on the page) | https://ornithopter.org/mobile/unmanned.html | VERIFIED press for the record; energy UNSOURCED; 11 kg INFERRED |
| Hydrogen fuel cell, smallest module found | Horizon AEROSTAK 250: 250 W rated, 720 g, plus a 200–250 g pressure regulator, before the cylinder | https://www.unmannedsystemstechnology.com/wp-content/uploads/2022/09/H3D_Brochure_AEROSTAK_compressed.pdf | SNIPPET maker |
| Fuel cell against the machine | 920 g before the hydrogen cylinder exceeds the whole machine (0.3–0.85 kg). A 50–150 W module with its mass: not found | | INFERRED |
| Supercapacitors | 1–10 Wh/kg, 500–10,000 W/kg (review table) | https://arxiv.org/pdf/1903.03740 | SNIPPET paper |
| Supercapacitors, read | At 5 Wh/kg, 22 Wh would be 4.4 kg: not a sustained source. As burst: 32 J is 0.009 Wh, i.e. 2 g at 5 Wh/kg; but a P45B already gives 132 W per 70 g cell (A13), so nothing is gained | | INFERRED |

---

## B. Burst sources for a crossing

### B.1 CO₂ cartridges as cold gas

| item | numbers | URL | label |
|---|---|---|---|
| 12 g cartridge | 31.5 g empty, 43.5 g full (hobby wiki); measured full: Daisy 41.1 g, Gamo 41.7 g with labels; measured fill of one brand 11.7 g, not 12 g; about 19 mm × 83 mm | https://hardairmagazine.com/reviews/how-much-co2-does-a-12-gram-co2-cartridge-contain-heres-the-answer/ ; https://spudfiles.com/spud_wiki/index.php/CO2_cartridge (403; via search summary) | SNIPPET press / forum |
| 8 g and 16 g cartridges, full and empty mass | UNKNOWN (two searches) | — | UNKNOWN |
| Vapour pressure | 57 bar at 20 °C, changing by about 1.25 bar/°C; the capsule cools as liquid evaporates | B.2 paper | VERIFIED paper |
| Specific impulse of CO₂ as a cold gas | 67 s theoretical, 61 s measured (table copied from a propulsion reference; conditions on the row differ from a room-temperature cartridge) | https://en.wikipedia.org/wiki/Cold_gas_thruster | SNIPPET |
| Impulse per cartridge, ceiling | J = m_gas × Isp × g: 8 g: 0.008 × 61 × 9.81 = 4.8 N·s; 12 g: 7.2 N·s; 16 g: 9.6 N·s. Ceiling only: a pierced cartridge has no nozzle, cools itself and can spit liquid. Measured impulse of a pierced cartridge: UNKNOWN | | INFERRED |
| Speed given, ceiling (Δv = J / m, device mass and losses ignored) | 8 g: 16.0 / 9.6 / 6.0 m/s for 0.3 / 0.5 / 0.8 kg. 12 g: 23.9 / 14.4 / 9.0 m/s. 16 g: 31.9 / 19.1 / 12.0 m/s | | INFERRED |
| Device mass | 12 g cartridge 41–43.5 g, plus a piercing valve and holder whose mass was not found. 42 g is 14% / 8% / 5% of a 0.3 / 0.5 / 0.8 kg body, before the valve | | INFERRED |
| Uses per fill; re-arm | One per cartridge. Re-arming on board means either one cartridge and one piercer per crossing, or a valve that meters one cartridge into several shots (B.2 built such a valve for a 5 ml tank; nobody has metered a cartridge on a flying machine) | | INFERRED |
| Precedent as pure cold-gas thrust on a flying or swimming robot | not found | — | UNKNOWN |

### B.2 CO₂ driving a water jet (Imperial College AquaMAV)

| item | numbers | URL | label |
|---|---|---|---|
| AquaMAV jet thruster (Siddall & Kovac, IEEE/ASME Trans. Mechatronics 22(1), 2017) | Robot 100.8 g with batteries, electronics, folding wings; **thruster 41.9 g** without electronics (40% of the robot, "twice as heavy as is required by design standards" for test safety); 5 ml tank of CO₂ at 57 bar released into a water tank that stays below 10 bar; peak thrust 5 N; exit **above 11 m/s**; minimum flight speed 8.5 m/s; fails to reach it if started deeper than 0.8 body lengths; launched at 30°, 50°, 70° in a tank and from a lake; only limited gliding achieved | https://nubot-uaav.github.io/static/pdf/squid-like%20aavs/Fast_Aquatic_Escape_With_a_Jet_Thruster.pdf | VERIFIED paper |
| Valve and re-arm | Shape-memory-alloy wire lifts a poppet 1.6 mm against 24 N; a 7.4 V 200 mAh battery is "sufficient for over 150 actuations" at 2 mAh per launch; over 100 actuations with no loss; the tank is charged from a canister by hand between shots; the water tank must refill | same | VERIFIED paper |
| Shots per fill | **One.** No on-board recharge | same | VERIFIED paper |
| Earlier prototypes | a 70 g vehicle to 11 m/s in 0.3 s (57 bar); a 33.5 g compressed-air jumper | https://spiral.imperial.ac.uk/entities/publication/71639ef3-e6cc-425d-89ec-35d86d2f93e8 | SNIPPET paper |
| Impulse delivered | at least m·v = 0.1008 × 11 = 1.11 N·s (gravity and drag during the push not added) | | INFERRED |
| Gas per shot | 5 ml of saturated CO₂ vapour at 57 bar and 20 °C: about 194 kg/m³ × 5 × 10⁻⁶ m³ ≈ 1.0 g (density is a handbook value, UNSOURCED today; more if liquid entered the tank) | | INFERRED |
| Scaled to this machine | If thruster mass scales with impulse (**assumption**): 41.9 g × (m·v / 1.11 N·s), i.e. 41.6% of body mass × v/11. At 6 m/s: 23% of the body (68 / 113 / 181 g for 0.3 / 0.5 / 0.8 kg). At 9 m/s: 34% (102 / 170 / 272 g). Half of that if the paper's "twice as heavy as required" is taken up | | INFERRED |
| One 12 g cartridge through such a jet | 12 g / 1.0 g = about 12 AquaMAV-sized shots = 13 N·s in total, against 7.2 N·s as cold gas. For 0.5 kg at 6 m/s (3.0 N·s) that is about 4 exits per cartridge on paper. Not demonstrated by anyone | | INFERRED |
| Works on land | No: the reaction mass is water taken in while afloat. On land the water would have to be carried | same | VERIFIED paper (principle) |
| Squid-like vehicle, Beihang (Hou et al., 2019) | compressed air in a nose cylinder; 10–20 m through the air; "can only fire its jet once"; the authors discuss liquid CO₂ for repeats; mass not found | https://spectrum.ieee.org/robot-squid-and-robot-scallop-showcase-bioinspired-underwater-propulsion | SNIPPET press |

### B.3 Combustion and detonation

| item | numbers | URL | label |
|---|---|---|---|
| Calcium-carbide jet robot (Zufferey et al., Science Robotics 2019) | **160 g**; water reacts with carbide powder to make acetylene, ignited in a chamber, driving a water jet; force 25 times the robot's weight; flights up to 26 m; "just 0.2 g of calcium carbide"; one moving part (a small pump); 22 flights in tests | https://www.imperial.ac.uk/news/192826/flying-fish-robot-propel-itself-water/ | VERIFIED press (university) |
| same, further figures | peak thrust up to 51 N; 11 m/s at launch; about 50 mg of carbide and three drops of water per launch, 0.2 g "enough for multiple jumps" (so about 4 per load, INFERRED); a glider with no control surfaces yet; one secondary page says the tank takes 20 minutes to refill | https://spectrum.ieee.org/water-jet-powered-drone-takes-off-with-explosions ; https://oceanbites.org/a-high-flying-aquatic-robot/ | VERIFIED press / SNIPPET |
| Impulse delivered | at least 0.160 × 11 = 1.76 N·s. 51 N / (0.160 × 9.81) = 32 body weights at the peak, against "25 times" in the release | | INFERRED |
| Scaled to this machine | Needs 1.0–4.1 times that impulse (1.8–7.2 N·s). Fuel mass is negligible at any scale; chamber mass and its scaling: UNKNOWN (not in the pages read) | | INFERRED |
| Works on land | No: water is both reactant and reaction mass | | VERIFIED press (principle) |
| Hybrid RoboBee (Chen et al., Science Robotics 2017) | 175 mg; electrolytic plates make oxyhydrogen in a chamber, the gas first floats the wings clear of the surface, a spark then ignites it; the robot "cannot yet fly immediately" after the exit; exit speed not in the pages read | https://www.engadget.com/2017-10-26-harvard-robobee-can-fly-in-and-out-of-water.html ; A11 section B | SNIPPET press |
| RoboBee, relevance | 175 mg against 300–850 g is a factor of 1,700–4,900 in mass; the result that carries over is the order of events (buoyancy first, then impulse), not a size | | INFERRED |

### B.4 Springs and jumping

| item | numbers | URL | label |
|---|---|---|---|
| Record spring jumper (Hawkes et al., Nature 604, 2022) | 30 g; 30 cm tall; jumped 32.9 ± 0.7 m; specific energy 1,075 J/kg; 0 to over 28 m/s in 9 ms; a small motor ratchets energy into a carbon-bow and rubber spring over many turns | https://www.wevolver.com/specs/engineered-jumper-reaches-heights-over-30-meters ; https://www.theregister.com/2022/05/01/humanmade_hopper_outleap_rival_robots/ | SNIPPET press (of the paper) |
| MultiMo-Bat (Woodward & Sitti, IJRR 2014) | 115.6 g; jumps over 3 m and glides; a later thesis: 100–162 g, over 4.5 m | https://is.mpg.de/publications/woodward2014multimo ; https://kilthub.cmu.edu/articles/thesis/MultiMo-Bat_Biologically_Inspired_Integrated_Multi-Modal_Locomotion/6720758 | SNIPPET paper |
| Salto (Berkeley) | 100 g; average hop height 0.65 m in a later comparison | https://arxiv.org/html/2312.08301v1 | SNIPPET paper |
| EPFL jumpers and jump-glider | 7–14.3 g, up to 27 times their own size; jump-glider 16.5 g | https://infoscience.epfl.ch/record/148680 | SNIPPET paper |
| Stanford jump-glider | 30 g, 70 cm wing; goes less high than a ballistic jumper of the same mass and energy, but farther | https://bdml.stanford.edu/pmwiki/pmwiki.php/Main/JumpGliding | SNIPPET press |
| RAVEN (Shin et al., Nature 636, 2024) | **0.62 kg** fixed-wing with bird-like legs (springs and motors); walks, hops, jumps onto a 26 cm step, jumps into flight; jumping take-off was more energy-efficient than a propeller-only take-off. It has a propeller. Take-off speed from the jump: not in the pages read | https://www.scimex.org/newsfeed/bird-robot-shows-how-flying-machines-could-one-day-move-better-on-the-ground ; https://www.wevolver.com/article/bird-inspired-drone-can-jump-for-take-off | SNIPPET press (of the paper) |
| Vampire bat | 23 g; 2.4 m/s vertical in under 30 ms, from the forelimbs | A11 section C | A11 |
| Specific energy delivered as height, g·h | Hawkes 9.81 × 32.9 = 323 J/kg (of the 1,075 stored); MultiMo-Bat 9.81 × 3 = 29 J/kg, equal to a launch at √(2 × 9.81 × 3) = 7.7 m/s; Salto 6.4 J/kg (3.6 m/s) | | INFERRED |
| Mechanism mass for this machine, floor | Energy needed 5.4–32.4 J (section 0). At the record 1,075 J per kg of jumper: 5–30 g. At MultiMo-Bat's 29 J/kg the whole 0.5 kg body would have to be the jumper to reach 7.7 m/s. No jumper above 0.2 kg with a stated launch speed was found except RAVEN, whose speed was not read | | INFERRED |
| Force a jump needs | To reach v over a leg stroke s the mean acceleration is v²/(2s). For s = 0.10 m: 18 g at 6 m/s, 41 g at 9 m/s, i.e. 54 / 90 / 144 N at 6 m/s and 122 / 203 / 324 N at 9 m/s for 0.3 / 0.5 / 0.8 kg. A jump to the bat's 2.4 m/s over 0.10 m is 2.9 g and reaches h = 0.29 m | | INFERRED |
| Uses per fill; re-arm | Unlimited: a motor rewinds the spring (Hawkes, RAVEN, MultiMo-Bat all do); 32 J is 0.009 Wh per jump | rows above | SNIPPET |
| From water | A jump needs something to push on. A spring jump from open water by a machine of this size: not found | — | UNKNOWN |

### B.5 Short electric boost through the same drive

| item | numbers | URL | label |
|---|---|---|---|
| RoboFalcon 2.0 (Science Advances 2025) | **About 800 g, 1.2 m span, wing loading 3.64 kg/m²**; flapping amplitude 85°, up to 7.5 Hz (about 7 Hz in the take-off trial); **peak power up to 400 W in take-off**; airspeed about 4 m/s, 6 m/s with the centre of mass moved forward; two legs hold the body at about 45°, **the wings alone lift it off** with a flap-sweep-fold stroke; trials on a safety tether from a 15 m ceiling, cleared a 0.5 m threshold; no tail, pitch control fails above about 3 m/s; hover not achieved; the authors call the take-off not energy-efficient. The paper states that other bird-scale flapping robots with self-take-off "still have to rely on jumping, catapult, or other assistive devices" | https://pmc.ncbi.nlm.nih.gov/articles/PMC12407090/ | VERIFIED paper (first 100,000 characters read) |
| RoboFalcon, derived | 400 / 0.8 = 500 W per kg of machine at the peak, against 392 W/kg for E-Flap's peak and 122–175 W/kg in cruise (A13) | | INFERRED |
| Osprey-inspired eagle-scale flapper (ICRA 2024) | 870 g, 1.8 m span, self-take-off with an auxiliary platform that sets the initial pitch | https://ieeexplore.ieee.org/document/10610958 | SNIPPET paper |
| What the cells allow | At 500 W/kg the body needs 150 / 250 / 400 W for 0.3 / 0.5 / 0.8 kg; for 5 s that is 0.21 / 0.35 / 0.56 Wh. A P45B gives 132 W at its continuous limit (A13), so 2 / 2 / 4 cells cover it with no burst rating invoked; an FPV LiPo of 203 g has a 4.6 kW nameplate (A.1). **The sustained pack already holds the burst; a separate electric burst source buys nothing** | | INFERRED |
| What the drive survives | E-Flap's 71 g drive: estimated 130 W mechanical peak, "close to 200 W" electrical at full throttle, hinge load estimated up to 6.5 N·m (A13). Overload ratings of a 20 g motor, a printed gearbox or a crank for 3–5 s: UNKNOWN. RoboFalcon's drive mass: not in the part read | A13 | UNKNOWN |
| Uses per fill | Each 5 s at 500 W/kg costs 0.7 Wh per kg of machine, 1.6% of the 43.8 Wh/kg that 15 minutes of flight takes (A13) | | INFERRED |

### B.6 Small solid rocket motors

| item | numbers | URL | label |
|---|---|---|---|
| Estes A8-3 | total impulse 2.5 N·s | https://courses.washington.edu/engr100/All_Sections/Rocket/HTML%20Handouts/01b_hnd_engine_chart.pdf | SNIPPET maker chart |
| Estes B6-4 | 5.0 N·s; 6 N average thrust | https://www.wardsci.com/store/product/43574834/estes-standard-rocket-engines | SNIPPET shop |
| Estes C6-5 | 10.0 N·s; 6 N average, 15.3 N maximum; 24.1 g (another chart: 25.8 g); propellant 12.2–12.5 g | https://hobbywholesale.com/toys-games-puzzles/outdoor-fun/rockets/engines/rocket-c6-5-std-3-est-est10024.html | SNIPPET shop |
| Estes D12-5 | 20.0 N·s; 12 N average, 32.9 N maximum; 43.1–45.6 g; propellant 24.2–24.9 g | https://estesrockets.com/product/001567-d12-5-engines | SNIPPET maker |
| Burn time | impulse / average thrust: 10 / 6 = 1.7 s (C6), 20 / 12 = 1.7 s (D12) | | INFERRED |
| Speed given (Δv = J / m, horizontal, losses ignored) | A8: 8.3 / 5.0 / 3.1 m/s for 0.3 / 0.5 / 0.8 kg. B6: 16.7 / 10.0 / 6.3. C6: 33 / 20 / 12.5. D12: 67 / 40 / 25 | | INFERRED |
| Thrust against weight | Weights are 2.9 / 4.9 / 7.8 N. A C6's 6 N average lifts the 0.3 kg body, barely the 0.5 kg one, not the 0.8 kg one; a D12's 12 N lifts all three | | INFERRED |
| Uses per fill; re-arm | One per motor, each with its own igniter; no reload in the field without hands | | INFERRED |
| After immersion | Black-powder motor and electric igniter under water: UNKNOWN (not searched beyond the rows above) | — | UNKNOWN |
| Taiwan | Whether a hobbyist may buy or fire model-rocket motors: **not found**. One Chinese-language search returned only the Space Development Act draft (penalties for unlicensed launches of space vehicles) and research-rocket rules; neither says whether a 25 g black-powder motor is covered, and no Taiwanese shop listing came up | https://news.pts.org.tw/article/513658 | UNKNOWN |
| Precedent on a flapping or amphibious robot | not found | — | UNKNOWN |

### B.7 The burst sources side by side (INFERRED from B.1–B.6)

| source | reaches 6 m/s? (0.3 / 0.5 / 0.8 kg) | reaches 9 m/s? | basis |
|---|---|---|---|
| 12 g CO₂, cold gas | yes / yes / yes | yes / yes / just (9.0) | ceiling arithmetic, no demonstrated machine |
| CO₂ water jet | yes if the thruster is 23% of the body (12% at the paper's "half") | 34% (17%) | demonstrated at 0.10 kg and 11 m/s; scaled by assumption |
| Carbide combustion jet | needs 1.0–2.7 times the demonstrated impulse | 1.5–4.1 times | demonstrated at 0.16 kg and 11 m/s; chamber scaling unknown |
| Spring jump | 18 g over a 10 cm stroke | 41 g | demonstrated 7.7 m/s equivalent at 0.116 kg; 0.62 kg jumps into flight at a speed not read |
| Electric boost through the wings | demonstrated lift-off at 0.8 kg and 500 W/kg, speed 4–6 m/s | not shown | one machine, tethered |
| Estes C6 | yes / yes / yes (12.5) | yes / yes / yes | arithmetic; 0.8 kg cannot be lifted vertically by it |

---

## C. Run-ups that need no thrust above weight

### C.1 Machines

| item | numbers | URL | label |
|---|---|---|---|
| BOLT (Peterson, Birkmeyer, Dudley, Fearing; IROS 2011) | **11.4 g**, 28 cm span; legs and flapping wings on one drive; runs at 1.5 m/s at 12.5 Hz; reaches 1 m/s in 0.14 s; leaves the ground at 2.5 m/s after just under 2 m at 18 Hz, or at 1.75 m/s after 1 m with the board moved back; an earlier version "could only takeoff by running from a one meter or higher ledge" | https://people.eecs.berkeley.edu/~ronf/PAPERS/kpeterson-iros11.pdf | VERIFIED paper |
| H2Bird launched from a running robot (Rose et al., ICRA 2015) | 13.2 g ornithopter, 26.5 cm span, **cannot take off from rest**; carried by a 30 g legged robot that runs up to 2.7 m/s; launches succeeded from about 1.2 m/s; failures appeared again near 1.5 m/s | https://people.eecs.berkeley.edu/~ronf/PAPERS/crose-icra15.pdf | VERIFIED paper (text extract; mass of the legged robot as printed) |
| RoboFalcon 2.0 | 0.8 kg, standing start at 45°, no run (B.5) | B.5 | VERIFIED paper |
| Ornibibot | a small flapping robot said to take off from the ground unassisted; no numbers on the page | https://labiybafakh.github.io/projects/ornibibot-v1.0 | SNIPPET |
| Hobby ornithopter rising off the ground on wheels | one Model Aviation article describes a rise-off-ground model with carbon gear and plywood wheels; mass, run and thrust not given | https://library.modelaviation.com/article/micro-flying-200611 | SNIPPET press |
| Manned ornithopters | Schmid (1942) is reported to have taken off unassisted, perhaps from a wheeled dolly; the Toronto machine (2006) bounced during its ground roll and used a jet boost | https://ornithopter.org/history.manned.shtml | SNIPPET press |
| A flapping machine of 0.1–1 kg that took off from wheels or from water after a run, with mass and thrust stated | **not found** | — | UNKNOWN |
| FAAV water exit | under 300 g, from 0.5 m depth at 70° pitch, flapping only (the one flapping water exit on record) | A11 section B | A11 |

### C.2 Birds

| item | numbers | URL | label |
|---|---|---|---|
| Common loon | take-off run on water "usually" 200–600 ft (61–183 m); shorter into wind; cannot take off from land | https://loon.org/looncam-blogs/loon-cam-2019/getting-airborne-why-loons-dont-visit-your-birdbath/ | SNIPPET press (conservation group) |
| Swans | about 30 yards (27 m) of run from land, anecdotal | https://boards.straightdope.com/t/can-swans-take-off-from-dry-land/787534?page=2 | SNIPPET forum |
| Western and Clark's grebes, "rushing" (Clifton, Hedrick, Biewener; J. Exp. Biol. 218, 2015) | 0.7–1.8 kg; 16–20 steps per second over 5–20 m for about 4 s (up to 7 s); the foot slap alone gives up to 30–55% of weight support. This is a display, not a take-off; the wings are held out | https://sicb.org/?p=21861 ; https://zendy.io/pdf-viewer/10.1242/jeb.118745 | SNIPPET paper |
| Speed, thrust as a fraction of weight, or wing share of lift during any bird's water run | UNKNOWN | — | UNKNOWN |

### C.3 Planing

| item | numbers | URL | label |
|---|---|---|---|
| Water resistance of floats | peaks at the "hump", just before planing; the thrust margin and the acceleration are smallest there; larger floats lower the hump and raise the resistance near lift-off | https://ntrs.nasa.gov/api/citations/19930082208/downloads/19930082208.pdf ; https://pilotfriend.com/training/flight_training/seaplanes/takeoff.htm | SNIPPET paper / press |
| Load over resistance at the hump | 4.8–6.2 in towing-tank comparisons of planing-tail against conventional hulls (a blog quoting NACA work); 2.67 and 2.9 in two NACA reports on other configurations (one with hydrofoils) | https://maltehoeltken.substack.com/p/planing-tail-versus-conventional ; search summary of NACA reports on ntrs.nasa.gov (which report holds 2.67 and which 2.9 was not established) | SNIPPET press / paper |
| Hump drag as a fraction of weight | 1/6.2 to 1/4.8 = 0.16–0.21; 1/2.9 to 1/2.67 = 0.34–0.37. All for full-size hull forms; a figure for a hull of 0.3–0.8 kg: UNKNOWN | | INFERRED |
| Against a flapper's thrust | Robo Raven II: thrust 1.72–2.32 N at 0.30–0.38 kg (A13), i.e. T/W = 1.72 / (0.382 × 9.81) = 0.46 to 2.32 / (0.302 × 9.81) = 0.78. That is above 0.16–0.37, **if** the thrust measured in flight tests holds at low speed and **if** the hull exists and weighs nothing; neither is shown | | INFERRED |
| Length of a run | s = v² / (2a), a = (T/W − r)·g. With T/W = 0.46 and resistance r = 0.2 of weight: a = 2.55 m/s², s = 7.1 m to 6 m/s and 15.9 m to 9 m/s; with T/W = 0.78: 3.2 m and 7.1 m. Wing lift during the run, which unloads the hull or wheels, is not counted | | INFERRED |

Read of C (INFERRED): the run-up is physically open (a thrust of half the weight exceeds the hump on paper)
and biologically common (loons, swans), but as a machine it exists only at 11–13 g. At 0.3–0.85 kg the only
unassisted take-off found is RoboFalcon's standing start on 500 W/kg.

---

## D. Drives other than one motor and a fixed crank

| item | numbers | URL | label |
|---|---|---|---|
| Cable-driven variable-amplitude drive (Cal Poly thesis, ornithopter "K1") | brushless motor; amplitude variable in flight; up to 6 Hz at 90° peak-to-peak in flight tests; mass not in the summary | https://digitalcommons.calpoly.edu/theses/3083 | SNIPPET paper (thesis) |
| Two motors on one wing pair, amplitude set by their phase difference | patent; no machine or mass given | https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8333342 | SNIPPET (patent) |
| Amplitude-variable mechanism for a flapping micro air vehicle (NWPU) | title only | https://pure.nwpu.edu.cn/en/publications/integrated-design-and-analysis-of-an-amplitude-variable-flapping-/fingerprints | SNIPPET |
| Crank with slotted dual lever (quick-return stroke) | bench study at micro-air-vehicle size | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9680486/ | SNIPPET paper |
| Flap-sweep-fold wing on one drive (RoboFalcon 2.0) | 0.8 kg; sweep amplitude adjustable 5–25° and used for pitch control; this is the fore-aft degree of freedom that A11 (items 8 and 10) says water thrust and wing-walking need | B.5 URL | VERIFIED paper |
| One servo per wing, no crank | Robo Raven, 0.29–0.38 kg, 4 Hz; amplitude, frequency and left/right difference free; power per kg 11–83 times below a geared drive | A13 B.2 | A13 |
| One crank, compliant wing, same frequency in air and water | FAAV, under 300 g, about 5 Hz in both; the wing's bending cuts the stroke in water | A11 section B | A11 |
| Clutch or gear shift between an air ratio and a water ratio | **not found** in any flapping machine | — | UNKNOWN |
| Direct-drive resonant flapping at 50–150 W | **not found** above 12 g (Purdue, A11). Spring-wing studies are insect-scale or bench models; one abstract states there is "little direct experimental evidence" that springs reduce flight energy; another finds series-elastic elements may hurt control; a third argues internal damping can cancel the gain | https://arxiv.org/html/2403.03398v1 ; https://sicb.org/abstracts/resonance-properties-of-insect-inspired-series-elastic-flapping-wings/ ; https://ar5iv.arxiv.org/html/2012.05428 | SNIPPET paper |
| Pneumatic flapping actuator (off the same CO₂) | **not found** at any size in two searches | — | UNKNOWN |
| Legs as a second drive | RAVEN: 0.62 kg including legs that walk, hop and jump (B.4); leg mass not read | B.4 | SNIPPET |

Read of D (INFERRED): of the options a hobbyist could build, three have a flying precedent near this mass:
a servo per wing (0.3–0.4 kg), a crank whose amplitude is cut by wing compliance (under 0.3 kg, both media),
and a crank with an added sweep (0.8 kg, air only). Gear shifts, resonant direct drives and pneumatic
actuators have none.

---

## Summary table

"System mass" is for the duty named in the row. "After immersion" means: usable again without hands after the
machine has been under water.

| option | role | system mass for the stated duty | uses per fill | works after immersion | hobby-buyable | strongest precedent (mass) |
|---|---|---|---|---|---|---|
| Molicel P45B pack + motor + reduction (A13 baseline) | sustained | 182 / 287 / 393 g for 50 / 100 / 150 W shaft, 15 min (INFERRED) | recharge | yes if sealed (FAAV swims with a waterproof motor, A11) | yes | E-Flap, 0.51 kg, 15 min with 93 g more battery (A13) |
| Molicel P50B pack, same drive | sustained | 169 / 263 / 356 g (INFERRED) | recharge | same | yes, about US$23 per cell | none yet in a flapper |
| FPV LiPo pouch pack | sustained (worse), burst (ample) | 106–142 Wh/kg at the pack: about 1.3–1.7 times the baseline pack mass | recharge | same | yes | Flapper Nimble+, 0.10–0.13 kg (A11) |
| Semi-solid pack (Tattu/Grepow) | sustained | smallest found 400 g (118 Wh, 1C); 5C packs from 1.8 kg | recharge | unknown | not in this size; larger packs listed, often sold out | none in a flapper |
| Silicon-anode cells (Amprius) | sustained | 370–400 Wh/kg at the cell (press) | recharge | unknown | no (sold to manufacturers; one pack maker lists them) | drone manufacturers; none in a flapper |
| Glow engine .049 (Tee Dee) + fuel + reduction | sustained, air only | 135 g for 50 W shaft, 15 min (INFERRED) | refuel | no hands-off case found; restarts by hand after < 30 s under water (forum) | yes | model ornithopters of the 1930s–50s, mass not stated; Orniplane 3.4 kg |
| Glow engine .15 (O.S. 15LA) + fuel + reduction | sustained, air only | 227–287 g for 100 W; 242–287 g for 150 W (INFERRED) | refuel | same | yes, €100–115 | same |
| Petrol engine (9 cm³) | sustained | 535 g engine and ignition alone | refuel | unknown | yes | none at this mass |
| CO₂ expansion motor | sustained | at most 17–18 Wh per kg of cartridge (ideal) | one cartridge | unknown | motors are collectors' items; cartridges yes | free-flight models of a few grams |
| Hydrogen fuel cell | sustained | 920 g before the cylinder (250 W module) | refill | unknown | no (industrial) | none under several kg |
| Rubber | — | about 2 Wh/kg (UNSOURCED): 11 kg for 22 Wh | rewind | yes | yes | indoor models of about a gram, 21 min 44 s |
| Supercapacitor | burst only | 1–10 Wh/kg; adds nothing to a pack that already gives 1.9 kW/kg | recharge from the pack | unknown | yes | none found |
| 12 g CO₂ cartridge, cold gas | burst | 41–43.5 g plus a piercing valve (mass unknown); ceiling 7.2 N·s | 1 per cartridge | cartridge yes; valve unknown | yes (bicycle and airgun cartridges) | none as thrust on a robot |
| CO₂ water jet (AquaMAV type) | burst, water exit only | 41.9 g for 1.11 N·s; scaled: 23–34% of body mass, half if lightened (INFERRED) | 1 as built; about 4 per 12 g cartridge on paper | yes, it is built for it | parts yes; the valve is home-made (SMA wire, 7075 tank) | AquaMAV, 0.10 kg, > 11 m/s |
| Carbide / acetylene combustion jet | burst, water exit only | fuel 50–200 mg; chamber mass unknown | about 4 per 0.2 g load (INFERRED from press); 22 flights in tests | yes, it is built for it | calcium carbide is sold for lamps (availability in Taiwan not checked) | Imperial robot, 0.16 kg, 11 m/s, 26 m |
| Oxyhydrogen from electrolysis | burst, water exit only | not scalable from 175 mg | repeatable (water and electricity) | yes | — | RoboBee, 175 mg |
| Spring / leg jump | burst, ground only | floor 5–30 g at the record's 1,075 J/kg; realistic mass unknown | unlimited (motor rewinds) | unknown | yes (carbon, latex, small gearmotor) | RAVEN, 0.62 kg (speed not read); MultiMo-Bat, 0.116 kg, 7.7 m/s equivalent |
| Electric boost through the wings | burst, ground (and FAAV-style water exit) | no added source; the drive must take about 500 W per kg of machine | about 1.6% of the 15-minute energy per 5 s | yes if the drive is sealed | yes | RoboFalcon 2.0, 0.8 kg, 400 W peak, tethered lift-off |
| Estes C6 / D12 rocket motor | burst | 24–26 g for 10 N·s; 43–46 g for 20 N·s | 1 per motor | unknown | legality and supply in Taiwan not found | none on a flapper |
| Take-off run (wheels or planing hull) | no burst source | hull or wheels (mass unknown); needs T/W above 0.16–0.37 at the hump | unlimited | yes | yes | BOLT, 0.011 kg, 1–2 m run; nothing at 0.1–1 kg |

---

## What this survey could not establish

- **Measured specific fuel consumption of a glow engine.** The University of Maryland dynamometer work on
  0.16–7.5 cm³ engines was reached only as an abstract (DTIC, CORE and the repository bitstream returned HTML
  or JSON instead of the PDF). The fuel rates here come from forum run times and a forum rule of thumb, and the
  part-throttle rate is bracketed, not known.
- **Shaft power of the Tee Dee .010 and of any CO₂ expansion motor**; mass and power of the O.S. 10LA (manual
  pages did not give them in text); any glow engine's fuel-tank mass (10 g is assumed in A.2).
- **Engine vibration on a flapping frame, on-board restart, and hands-off recovery after immersion.** No source
  measures the first, shows the second, or reports the third.
- **An engine-powered ornithopter under 1 kg with both mass and duration stated.**
- **LiPo and P50B energy actually delivered at 4C**: no independent discharge test was found for either; the
  P50B figure borrows the P45B's capacity ratio, the LiPo figures are nameplate.
- **Whether semi-solid or silicon-anode packs of 20–80 Wh can be bought by an individual in Taiwan**, and at
  what price. The Upgrade Energy pages returned 403.
- **Measured thrust and impulse of a pierced 8, 12 or 16 g CO₂ cartridge**, and the full and empty mass of the
  8 g and 16 g sizes. The cold-gas figures are ceilings from a tabulated specific impulse.
- **The AquaMAV's total impulse, CO₂ mass per shot and thrust duration** (the paper's figures were not in the
  extracted text; 1.11 N·s is m·v, 1.0 g is from a handbook density), and **the carbide robot's chamber mass,
  carbide per launch and launches per load** (press only; the Science Robotics paper was not reached, and the
  press gives both "0.2 g" and "50 mg").
- **RoboBee's exit speed; RAVEN's take-off speed, leg mass and energy numbers** (Nature redirected to a login;
  EPFL's page returned 404); **the EPFL 7 g jumper's height**; **the Beihang squid vehicle's mass**.
- **Overload limits of a small brushless motor, a printed gearbox and a crank for 3–5 s**, and RoboFalcon 2.0's
  drive mass, battery and lift in numbers (the last 11,780 characters of the paper were not read).
- **Model-rocket motors in Taiwan**: legality, supply, and behaviour after immersion.
- **A take-off run by a flapping machine of 0.1–1 kg**, from wheels or from water; **rolling or planing
  resistance of a hull or wheel set at 0.3–0.8 kg**; **speed and thrust fraction in any bird's water run**
  (the grebe paper returned 403; loon and swan distances are from a conservation blog and a forum).
- **A clutch or gear shift, a resonant direct drive at 50–150 W, or a pneumatic flapping actuator** in any
  machine near this size.
- **Rubber's specific energy from a source** (the figure used is general knowledge), a tank-inclusive figure
  for compressed air, and a fuel cell of 50–150 W with its mass.
- **Glow fuel, carbide and CO₂ cartridge supply in Taiwan** was not checked beyond one cartridge listing on a
  Taiwanese shopping site appearing in a search.
