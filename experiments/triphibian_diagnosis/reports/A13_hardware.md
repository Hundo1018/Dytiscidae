# A13 — Hardware survey: what bought parts deliver (cells, flapping drives, structure, electronics, reference machines)

Survey date 2026-11-12. Web search and page fetches only; no code, no simulation, nothing installed. Follows
`A9_lit_hardware.md` (section 3) and `A11_flapping_design.md` (sections A–D); rows those reports already hold are
referred to as "A9" / "A11" and not repeated.

Purpose: bounds for a simulated design that its owner will build at home (3-D printer, parts bought online):
a flapping machine that flies, swims with the same wings and walks on them, 15 minutes per medium, on 18650 or
21700 cells.

Labels, per number:

- **VERIFIED** — seen today in the fetched manufacturer page, manufacturer datasheet (some are hosted by
  resellers; the URL given is the one fetched) or paper. PDFs that the fetch tool saved but could not read were
  read with `pdftotext`.
- **SNIPPET** — search-result summary, shop listing without datasheet, or a secondary source.
- **INFERRED** — arithmetic on labelled numbers; the arithmetic is shown. Conversions used: 1 kgf·cm =
  0.0981 N·m; 1 oz·in = 0.00706 N·m; a servo speed of t seconds per 60° is ω₀ = (π/3)/t rad/s.
- **UNKNOWN** — not found; what was tried is in "Not found".

---

## A. Cells

### A.1 Datasheet values

| item | numbers | URL | label |
|---|---|---|---|
| Samsung INR18650-35E (high-energy 18650; Samsung SDI "Technical Report", June 2015) | Typical 3450 mAh, min 3350 mAh (0.2C); nominal 3.60 V; mass max 50.0 g; discharge max 8.0 A ("not for cycle"); measured 3482 mAh / 12.62 Wh at 0.2C; at 3.5 A 3468 mAh / 11.94 Wh; at 7.0 A 3429 mAh / 11.37 Wh; at 10.0 A 3379 mAh / 10.98 Wh (23 °C); initial AC-IR 19.8 mΩ, DC-IR 34.5 mΩ (20.9 / 37.8 mΩ after 20 days at 60 °C) | https://static.rapidonline.com/pdf/10-5181_v1.pdf | VERIFIED |
| LG INR18650 MJ1 (high-energy 18650; LG spec LRB-PS-CY3500_MJ1 rev 1, 2016) | Nominal 3500 mAh, min 3350 mAh; nominal 3.635 V; max discharge 10 A; mass max 49.0 g; initial AC impedance ≤ 40 mΩ | https://cdn.shopify.com/s/files/1/0721/2761/1190/files/LG-MJ1.pdf | VERIFIED |
| Panasonic/Sanyo NCR18650GA (high-energy 18650; Sanyo data sheet) | Rated 3300 mAh (min at 20 °C), min 3350 / typ 3450 mAh at 25 °C; 3.6 V; max discharge 8 A; mass max 49.5 g with tube; rate curves drawn for 2–10 A (no table). Internal resistance: not on the sheet. Shops quote "10 A" | https://eshop.fulgurbattman.cz/priloha.php?ak=1446457 | VERIFIED (10 A: SNIPPET) |
| Samsung INR21700-50E (high-energy 21700; Samsung SDI spec INR21700-50E2 v1.0) | Standard discharge capacity ≥ 4900 mAh, rated ≥ 4753 mAh; nominal 3.63 V; max discharge 9.8 A continuous, 14.7 A "not for continuous discharge"; mass max 69.5 g; AC impedance ≤ 14.0 ± 5.0 mΩ; relative capacity 100% at 0.2C, 97% at 1C (4.9 A), 95% at 2C (9.8 A) | https://shop.aceongroup.com/cdn/shop/files/RLII-CSI-1618_Samsung_50E_DATASHEET.pdf | VERIFIED |
| LG INR21700 M50 (high-energy 21700) | Nominal 5.0 Ah (min 4.85 Ah), 3.63 V, 18.20 Wh (min 17.60 Wh); mass 68 ± 1 g; stated 267 Wh/kg; DCIR 30 ± 6 mΩ (30 s, 0.5C), ACIR ≤ 25 mΩ; max continuous discharge 1.5C at 5–60 °C; energy 14.6 Wh at 0 °C, 12.7 Wh at −10 °C | https://www.batterydesign.net/lg-21700-m50/ | SNIPPET (secondary page quoting the LG sheet) |
| Molicel INR-18650-P28A (high-drain 18650; product data sheet) | Typical 2800 mAh / 10.3 Wh, min 2700 mAh / 9.56 Wh; 3.6 V; max discharge 35 A; AC (1 kHz) 20 mΩ max, DC (10 A, 1 s) 20 mΩ; mass 46 g typical; stated 219 Wh/kg; rate curves at 0.56, 2.8, 10, 20, 30 A (figure, no table) | https://www.rosebatteries.com/hubfs/INR18650P28A-V2-80093.pdf | VERIFIED |
| Molicel INR-21700-P42A (high-drain 21700; product data sheet) | Typical 4200 mAh / 15.5 Wh, min 4000 mAh / 14.7 Wh; 3.6 V; continuous discharge 45 A; AC 10 mΩ, DC (10 A, 1 s) 16 mΩ typical; mass 70 g max; stated 230 Wh/kg; rate curves at 0.84, 4.2, 10, 20, 30 A (figure, no table) | https://www.molicel.com/wp-content/uploads/INR21700P42A-V4-80092.pdf | VERIFIED |
| Molicel INR-21700-P45B (high-drain 21700; cell specification INR21700P45B-01 ver. 0.2, a pre-release revision) | Typical 4.5 Ah, rated 4.35 Ah; 3.6 V; max continuous discharge 45 A; AC (1 kHz) ≤ 8 mΩ at shipping charge; mass 70.0 g max; capacity ≥ 4.35 Ah at 0.9 A, ≥ 4.20 Ah at 4.5 A, ≥ 4.15 Ah at 10 A, ≥ 4.10 Ah at 20 A (23 °C, to 2.5 V) | https://www.kronium.cz/uploads/p45b-datasheet.pdf | VERIFIED |
| P45B, DC resistance | "< 15 mΩ DCR at 10 s" (Molicel presentation) | search summary; source PDF not opened | SNIPPET |
| Samsung INR18650-30Q (high-drain 18650; Samsung SDI spec v0.0) | Nominal ≥ 3000 mAh; standard rated ≥ 2900 mAh at 10 A; 3.6 V; max continuous discharge 15 A (25 °C); mass max 48.0 g; AC impedance ≤ 26 mΩ; relative capacity (100% = 2900 mAh at 10 A): 97% at 15 A, 95% at 20 A | https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/196/samsung_5F00_30q_5F00_data_5F00_sheet.pdf | VERIFIED |
| Samsung INR21700-40T (high-drain 21700; Samsung SDI spec v0.0) | Standard ≥ 4000 mAh; rated ≥ 3900 mAh at 10 A; 3.6 V; max continuous 35 A without temperature cut, 45 A with an 80 °C cut; mass max 70.0 g; AC impedance ≤ 12 mΩ; relative capacity (100% = 3900 mAh at 10 A): 95% at 20 A, 93% at 30 A, 90% at 40 A | https://www.piekarz.pl/pdf/50840 | VERIFIED |
| Sony/Murata US18650VTC6 (high-drain 18650; Sony technical information rev 0.1, 2015) | Nominal 3130 mAh / 11.3 Wh (0.2C, to 2.0 V); rated 3000 mAh / 10.8 Wh; at 1C 3007 mAh / 10.8 Wh; at 10 A 2940 mAh / 10.5 Wh (to 2.5 V); 3.6 V; mass 46.5 g average; cycle test at 7.5 A and 15 A. Impedance value and a continuous-current rating: not in the sheet's text. Shops quote 15 A (30 A with an 80 °C cut) | https://www.kronium.cz/uploads/SONY_US18650VTC6.pdf | VERIFIED (ratings: SNIPPET) |

### A.2 Derived per cell (INFERRED)

Method. Specific energy = datasheet energy (or nominal Ah × nominal V) ÷ datasheet mass; where the sheet gives a
maximum mass the result is a lower bound. Continuous power = I_max × V_load, with V_load = V_nom − I_max × R and R
the datasheet DC resistance (AC limit where no DC value is given; this overstates V_load), or the measured mean
voltage where the sheet gives Wh at a rate. "15-minute energy" = what the cell can deliver in 0.25 h: at 4C if
4C ≤ I_max (capacity derated by the sheet's rate table where one exists), otherwise I_max × 0.25 h × V_load.

| cell | specific energy, low rate | continuous power at I_max (per cell; per kg) | 15-minute energy (per cell; per kg) | arithmetic |
|---|---|---|---|---|
| 35E | 12.62 / 0.050 = 252 Wh/kg | 8 A × 3.29 V ≈ 26 W; 530 W/kg | 4C = 13.8 A > 8 A: 8 × 0.25 × 3.29 = 6.6 Wh (52% of 12.62); 132 Wh/kg | mean V: 11.37/3.429 = 3.32 V at 7 A, 10.98/3.379 = 3.25 V at 10 A |
| MJ1 | 3.5 × 3.635 = 12.72 Wh; 260 Wh/kg | 10 A × (3.635 − 10 × 0.040) = 32 W; 660 W/kg | 4C = 14 A > 10 A: 10 × 0.25 × 3.23 = 8.1 Wh; 165 Wh/kg | R = AC limit |
| NCR18650GA | 3.45 × 3.6 = 12.42 Wh; 251 Wh/kg | 8 A × ≈3.3 V ≈ 26 W; 530 W/kg | 6.6 Wh; 133 Wh/kg | R unknown; 35E's measured voltage borrowed |
| 50E | 4.9 × 3.63 = 17.79 Wh; 256 Wh/kg | 9.8 A × (3.63 − 9.8 × 0.019) = 34 W; 490 W/kg | 4C = 19.6 A > 9.8 A: 9.8 × 0.25 × 3.44 = 8.4 Wh; 121 Wh/kg | R = AC upper limit 19 mΩ |
| M50 | 18.20 / 0.068 = 268 Wh/kg | 1.5C = 7.5 A × (3.63 − 7.5 × 0.030) = 26 W; 375 W/kg | 7.5 × 0.25 × 3.40 = 6.4 Wh; 94 Wh/kg | |
| P28A | 10.3 / 0.046 = 224 Wh/kg (sheet: 219) | 35 A × (3.6 − 35 × 0.020) = 102 W; 2200 W/kg | 4C = 11.2 A: 2.8 × 0.97 × (3.6 − 11.2 × 0.020) = 9.2 Wh; 199 Wh/kg | 0.97 capacity factor assumed from the 30Q/40T tables |
| P42A | 15.5 / 0.070 = 221 Wh/kg (sheet: 230, i.e. 67.4 g) | 45 A × (3.6 − 45 × 0.016) = 130 W; 1850 W/kg | 4C = 16.8 A: 4.2 × 0.95 × 3.33 = 13.3 Wh; 190 Wh/kg | 0.95 assumed |
| P45B | 4.5 × 3.6 = 16.2 Wh; 231 Wh/kg | 45 A × (3.6 − 45 × 0.015) = 132 W; 1880 W/kg | 4C = 18 A: 4.10 Ah × (3.6 − 18 × 0.015) = 13.7 Wh; 195 Wh/kg | 4.10 Ah is the sheet's 20 A floor; R = 15 mΩ (SNIPPET) |
| 30Q | 3.0 × 3.6 = 10.8 Wh; 225 Wh/kg | 15 A × (3.6 − 15 × 0.026) = 48 W; 1000 W/kg | 4C = 12 A: 2.81 Ah × 3.29 V = 9.2 Wh; 193 Wh/kg | 2.81 = 0.97 × 2.9 |
| 40T | 4.0 × 3.6 = 14.4 Wh; 206 Wh/kg | 35 A × (3.6 − 35 × 0.012) = 111 W; 1590 W/kg | 4C = 16 A: 3.71 Ah × 3.41 V = 12.6 Wh; 180 Wh/kg | 3.71 = 0.95 × 3.9 |
| VTC6 | 11.3 / 0.0465 = 243 Wh/kg | 15 A × ≈3.5 V ≈ 52 W; 1100 W/kg (rating SNIPPET) | at 10 A (3.4C): 10.5 Wh measured; 226 Wh/kg | 10.5/2.94 = 3.57 V mean at 10 A as printed |

Read of the table (INFERRED): on a 15-minute discharge the high-energy cells are current-limited to 35–64% of
their nominal energy (94–165 Wh/kg), and the high-drain cells deliver 180–226 Wh/kg. All figures are to the
datasheet cut-off (2.5 V; 2.0 V for the VTC6 nominal row), at 23–25 °C, new cells.

### A.3 What a pack carries besides cells

| item | numbers | URL | label |
|---|---|---|---|
| Upgrade Energy Dark Lithium V2, 6S1P, Molicel P42A, spot-welded, XT60 | 420 g; 4200 mAh; 7.2 × 4.2 × 6.4 cm; 45 A continuous | https://www.lumenier.com/products/upgrade-energy-dark-lithium-v2-21700-4200mah-6s-22-2v-25c-li-ion-battery-xt60 | VERIFIED |
| same pack, overhead | 6 × 70 g (sheet maximum) = 420 g, overhead 0; with 67.4 g per cell (15.5 Wh ÷ 230 Wh/kg) cells = 404 g, overhead 16 g = 4% | | INFERRED |
| iFlight Fullsend 6S1P 4000 mAh (EVE 40PL 21700) | 430 g, XT60H + balance plug | https://iflight-rc.eu/en/products/fullsend-6s-4000mah-li-ion-battery | SNIPPET |
| Auline 6S 3000 mAh (18650) | 314 g, i.e. 52.3 g per cell position | https://oscarliang.com/auline-v45-6s-3000mah-s70-4500mah-li-ion-battery/ | SNIPPET |
| Keystone 1042 / 1042P single-18650 holder (nylon) | 9–10 g (distributor compliance data) | https://my.element14.com/keystone/1042p/batt-holder-18650-x-1-smd/dp/4049848 | SNIPPET |
| holder against weld | 9.5 g on a 46 g cell = 21% added; 15-minute specific energy of a P28A falls from 199 to 9.2 / 0.0555 = 166 Wh/kg | | INFERRED |
| 2S 20 A protection board (HX-2S-D20) | 4 g; 44.8 × 18.9 × 3.5 mm | search summary of shop listings | SNIPPET |
| 3S 20 A protection boards | 4.3 g (59 × 20 × 3.4 mm), 8 g, 12 g on three listings | https://mantech.co.za/Datasheets/Products/BMS-20A-3S-S_SGT.pdf | SNIPPET |
| pure nickel strip 0.15 × 8 mm | 8.9 g/cm³ × 0.012 cm² × 100 cm = 10.7 g per metre (density is the handbook value, not fetched) | | INFERRED |

---

## B. Drives for flapping

### B.1 Brushless motor, gearbox and crank (what flying ornithopters of 0.3–1.2 kg use)

| item | numbers | URL | label |
|---|---|---|---|
| E-Flap drive (Zufferey et al., RA-L 2021) | Motor 20 g, 1130 rpm/V, selected as "150 W"; two stages 1:6 and 1:7 = 42:1; four-bar linkage; motor plus transmission 71 g; flapping amplitude 30–50°; max 5.5 Hz; estimated hinge load up to 6.5 N·m; estimated peak 130 W; at maximum throttle "close to 12 A or close to 200 W"; battery 16.5 V, 450 mAh; vehicle 0.510 kg, span 1.5 m, wing area 0.438 m² | https://zenodo.org/records/4917204/files/Eagle_Inspired_Flying_Robot_RAL_Final.pdf | VERIFIED |
| E-Flap drive, derived | 130 W / 0.071 kg = 1.8 kW per kg of drive (estimated mechanical peak); 200 W / 0.071 = 2.8 kW/kg electrical; 150 W / 0.020 kg = 7.5 kW/kg motor alone; drive is 71 / 510 = 13.9% of the vehicle | | INFERRED |
| MIT "Phoenix" (Jackowski, MIT thesis 2009) | 1.2 kg; span 1.8 m (2.0 m by scaling, shortened); PJS 3D 1200 outrunner, 848 rpm/V, torque constant 0.01126 N·m/A; two-stage 24:1; design output 17.7 N·m at 4 Hz (240 rpm, 5760 rpm at the motor); measured peak motor torque ≈ 0.35 N·m at ≈ 70% throttle; mean electrical power in flight 147 W at ≈ 4 Hz and ≈ 4 m/s; ESC Jeti Advance 30, 35 g, "about 300 watts at 11.4 volts"; battery 3S 2200 mAh | https://groups.csail.mit.edu/robotics-center/public_papers/Jackowski09.pdf | VERIFIED |
| Phoenix, derived | Output-shaft torque in flight 0.35 × 24 = 8.4 N·m peak; 147 / 1.2 = 122 W/kg | | INFERRED |
| "Kestrel" (a commercial Kinkade-pattern ornithopter measured in the same thesis) | 395 g; wing area 0.22 m²; span 1.17 m ("the 1.17 meter wingspan vehicle"; that this is the Kestrel is INFERRED from 1.8 m / 1.56 = 1.15 m); brushed motor; 97.8:1 gearbox; 6 Hz; maximum motor torque 0.00832 N·m, stated as 4.43 N·m at the shoulder "through the 97.8:1 ratio gearbox and the flapping linkage" (gearbox alone: 0.00832 × 97.8 = 0.81 N·m, so the stated value includes a linkage factor of 5.4 that the text does not derive). Shoulder torque taken to scale with span³: 3.79× for 1.56× span = 7.78 N·m | same | VERIFIED (0.81 N·m and 5.4: INFERRED) |
| BirdKit ornithopter gearbox (hobby product, discontinued) | Ratios 28:1 to 42:1 by five pinions (16–24 teeth); gearbox 3.1 oz (88 g); for models of about 0.5 kg and 4–7 ft span; one brushless motor "rated at 100 watts"; 35° flapping range; 3S LiPo | https://nathan.ornithopter.org/models.gbx.html | VERIFIED |
| Hobby design rule (ornithopter.org) | "about 100 watts per kilogram of flying weight"; gear so the motor runs at about 80% of no-load speed; 30:1 typical, 100:1 in some cases; flights "twenty minutes or less" | https://ornithopter.org/mobile/how.html | VERIFIED |
| Kinkade Slow Hawk 2 | "Fusion 380" brushed motor with assembled gearbox; 2S 1200 mAh 7.4 V; 9–18 min per charge; ESC ≥ 8 A; 5–20 mph. Span, mass, gear ratio: UNKNOWN | https://manualmachine.com/kinkade/slowhawk2/21133233-instruction-manual | SNIPPET |
| Kinkade Park Hawk | span 42 in (press). Mass, drive: UNKNOWN | https://www.cbsnews.com/news/watch-out-amazing-mechanical-bird-takes-flight | SNIPPET |
| HIT-Hawk (Pan et al. 2021) | High-speed brushless motor, 81:1 reduction, four-link RSSR spatial linkage | search summary of the paper | SNIPPET |
| Festo SmartBird | Motor "Compact 135, brushless"; 2 digital servos for wing torsion, 45° in 0.03 s; electromechanical efficiency ≈ 45%. Gear ratio, flapping frequency: UNKNOWN | https://www.festo.com/PDF_Flip/corp/smartbird_en/files/assets/seo/page54.html (403 on fetch) | SNIPPET |
| Festo BionicSwift | 42 g, length 44.5 cm, span 68 cm; one brushless motor, two servos, gearbox. Frequency, flight time: UNKNOWN | https://www.festo.com/en/bionicswift | SNIPPET |
| Dove (NWPU) | 220 g, 50 cm span, 30 min (2018 paper); a later review lists 245 g, 60 cm, 20 min, 10 Hz | https://pure.nwpu.edu.cn/en/publications/dove-a-biomimetic-flapping-wing-micro-air-vehicle/ | SNIPPET |
| Smaller geared flappers (KUBeetle-S 28:1, T3 25.4:1, DelFly, NUS 16:1) | in A11 section A | A11 | — |

### B.2 Hobby servos, 9–81 g

No manufacturer page fetched publishes a torque-speed curve; every power figure below is from the two end points,
P_max = τ_stall × ω₀ / 4 (a straight torque-speed line assumed). f₃₀ is the highest frequency of a ±30° sinusoid
whose peak speed equals the no-load speed, f₃₀ = ω₀ / (2π × 0.5236 rad); at the maximum-power point (half speed,
half stall torque) the frequency is half of f₃₀.

| item | numbers (mass; voltage; stall torque; no-load speed) | P_max, per kg, f₃₀ (INFERRED) | waterproof | URL | label |
|---|---|---|---|---|---|
| TowerPro SG90 | 9 g (10.6 g with lead); 4.8 V; 1.8 kgf·cm = 0.177 N·m (listings range 1.2–1.8); 0.10 s/60° = 10.5 rad/s | 0.46 W; 51 W/kg; 3.2 Hz | no | https://docs.circuit.rocks/motors/sg90-servo/ | SNIPPET |
| TowerPro MG90S | 13.4 g; 6 V; 2.2 kgf·cm = 0.216 N·m; 0.08 s/60° = 13.1 rad/s | 0.71 W; 53 W/kg; 4.0 Hz | no | https://www.servodatabase.com/servo/towerpro/mg90s | SNIPPET |
| KST X08H Plus (coreless, steel gears) | 9 g; 3.8–8.4 V; 2.4 kg·cm at 3.8 V, 3.85 at 6.0 V, 4.84 at 7.4 V, 5.30 at 8.4 V (0.520 N·m); 0.18 / 0.15 / 0.12 / 0.09 s/60° (11.6 rad/s at 8.4 V); travel ±60° | 1.51 W; 168 W/kg; 3.5 Hz (8.4 V). 0.34 W; 1.8 Hz (3.8 V, one cell) | not stated | https://cdn.shopify.com/s/files/1/0759/8009/files/KST_X08_20210614.pdf | VERIFIED |
| KST DS215MG V8.0 (coreless, aluminium case) | 20 g ± 10%; 6.0–8.4 V; 3.1 / 3.7 / 4.0 kgf·cm (0.392 N·m at 8.4 V); 0.060 / 0.055 / 0.050 s/60° (20.9 rad/s); travel ±70° | 2.05 W; 103 W/kg; 6.4 Hz | not stated | https://www.flyingtech.co.uk/wp-content/uploads/2025/06/DS215MG_V8.0Technical_Specifcation.pdf | VERIFIED |
| Robotis Dynamixel XL330-M288 (smart servo, 288.4:1) | 18 g; 3.7–6.0 V; stall 0.42 N·m (1.11 A) at 3.7 V, 0.52 N·m (1.47 A) at 5.0 V, 0.60 N·m (1.74 A) at 6.0 V; no-load 76 / 103 / 123 rev/min | 1.40 W; 78 W/kg; 3.3 Hz (5 V). 1.93 W; 107 W/kg; 3.9 Hz (6 V) | no rating on the page | https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/ | VERIFIED |
| Hitec HS-5086WP (micro, metal gear) | 28.5 g; 4.8–6.0 V; 3.0–3.6 kg·cm (0.353 N·m); 0.18–0.15 s/60° (7.0 rad/s); idle 3 mA, running 280 mA, stall 1.8 A | 0.62 W; 22 W/kg; 2.1 Hz | IP67; page defines 7 as "Immersion up to 1M" | https://hitecrcd.com/hs-5086wp-metal-gear-micro-digital-waterproof-servo/ | VERIFIED |
| Hitec D646WP (standard size, metal gear) | 61.0 g; 3.5–8.4 V; 11.6 kg·cm = 1.138 N·m at 7.4 V; 0.16 s/60° = 6.5 rad/s; idle 30 mA, running 330 mA, stall 2.2 A | 1.86 W; 31 W/kg; 2.0 Hz | IP67 | https://hitecrcd.com/products/servos/digital/d-series/d646wp/product | VERIFIED |
| DSServo DS3218 | 60 g; 4.8–6.8 V; 19 kg·cm at 5 V, 21.5 kg·cm (2.11 N·m) at 6.8 V; 0.16 / 0.14 s/60° (7.5 rad/s) | 3.94 W; 66 W/kg; 2.3 Hz | "IP66" on listings; one buyer review reports it is not waterproof | https://druav.com/products/dsservo-ds3218-digital-servo | SNIPPET |
| Savox SW-1210SG+ | 71 g; 277.7 oz·in at 6 V, 444.4 oz·in (3.14 N·m) at 7.4 V; 0.13 s/60° (8.1 rad/s) | 6.3 W; 89 W/kg; 2.4 Hz | sold as waterproof; rating not confirmed | https://servodatabase.com/servo/savox/sw1210sgp | SNIPPET |
| Savox SB-2290SG (brushless, steel gear) | 81 g; 50 kg·cm (4.90 N·m) at 7.4 V; 0.13 s/60° | 9.9 W; 122 W/kg; 2.4 Hz | "water resistant"; the SW-2290SG variant is listed IP67 | https://www.rynos.se/internt/artiklar/internal_documents/SB-2290SG_Spec-Sheet.pdf | SNIPPET |
| Futaba S9352HV (fast standard servo) | 72 g; 6.0–7.4 V; 305.6 oz·in (2.16 N·m) at 7.4 V; 0.06 s/60° (17.5 rad/s), 0.07 at 6.0 V | 9.4 W; 131 W/kg; 5.3 Hz | no | https://servodatabase.com/servo/futaba/s9352hv | SNIPPET |
| MKS HBL980 (helicopter tail servo) | 70.3 g; 6.2 kg·cm (0.608 N·m) at 8.4 V; 0.030 s/60° (34.9 rad/s), 0.036 at 7.4 V | 5.3 W; 75 W/kg; 10.6 Hz | no | https://www.motionrc.com/collections/mks-rc-servos/products/mks-hbl980-brushless-titanium-gear-high-speed-digital-servo-mks-hbl980 | SNIPPET |
| Hitec HS-645MG, Dynamixel XM430-W350 | 22 and 60 W/kg | A9 section 3 | — | A9 | — |

Servo-driven flapping that flew:

| item | numbers | URL | label |
|---|---|---|---|
| Robo Raven I (UMD; Bruck & Gupta retrospective, Biomimetics 2023) | One programmable digital servo per wing, no crank; 291.6 g with battery (264.5 g without) in Table 1, 285.0 g without battery and 328.8 g maximum take-off in Table 3; span 1.1–1.17 m; 4 Hz; 6.7 m/s; design target ≥ 0.20 m² of wing at 3.5 Hz. Servo model, battery, flapping amplitude: not in the article | https://pmc.ncbi.nlm.nih.gov/articles/PMC10603970/ | VERIFIED |
| Robo Raven II, flight tests of four wings | Span 1.33 m; 301.6 g without battery, 382.0 g maximum take-off; average lift 2.59–3.47 N, thrust 1.72–2.32 N; average power 13.6–17.6 W; average torque 0.99–1.17 N·m; maximum plunge velocity 12.6–14.7 rad/s. Whether power and torque are per wing or for both, and electrical or mechanical, was not established from the part of the page read | same | VERIFIED |
| Robo Raven V | Propellers added: thrust +261% at full signal; payload 43.8 g to 272.9 g; mass 438.1 g without battery, 711.0 g maximum take-off | same | VERIFIED |

### B.3 Other ways to flap at 3–10 Hz at this size

| item | numbers | URL | label |
|---|---|---|---|
| Direct-drive gimbal motor, iPower GM2804 | 42.3 g; 12N14P; "0.5–1 kg·cm" = 0.05–0.10 N·m; ≤ 25 W on one listing | https://shop.iflight.com/ipower-gm2804-gimbal-motor-pro1153 | SNIPPET |
| Direct-drive gimbal motor, iPower GM3506 | 80.3 g with encoder; 24N22P; "1.5 kg·cm" = 0.15 N·m (another listing: 0.6–1.0 kg·cm) | https://shop.iflight.com/ipower-motor-gm3506-brushless-gimbal-motor-pro967 | SNIPPET |
| Gimbal motor against the torque a 1.2–1.8 m wing needs | 0.05–0.15 N·m available against 4.4–8.4 N·m at the shoulder (B.1): short by 30–170×, so a reduction of that order would still be needed | | INFERRED |
| Crank-rocker / four-bar on a geared motor | E-Flap (5.5 Hz at 0.51 kg), Phoenix (4 Hz at 1.2 kg), Kestrel (6 Hz at 0.395 kg): B.1. Cloud Owl 4–6 Hz at 0.98 kg, 45° amplitude (SNIPPET, section E) | B.1 | — |
| One motor and a crankshaft, both media | FAAV, < 300 g, ≈ 5 Hz in air and water: A9, A11 | A9 | — |
| Cam, string and resonant direct drives | KU Leuven cam 40 Hz at 3.4 g; KUBeetle pulley-string 18 Hz at 16 g; Purdue spring-resonant direct drive ≈ 34 Hz at 12 g: all in A11 section A. No cam, string or resonant drive was found at 0.2–1 kg | A11 | — |
| Servo per wing, no crank | Robo Raven, 4 Hz at 0.29–0.38 kg: B.2 | B.2 | — |

---

## C. Structure and skin

### C.1 Printing filaments (manufacturer data sheets; printed specimens, X-Y = in the layer plane, Z = across layers)

| item | numbers | URL | label |
|---|---|---|---|
| PLA (Bambu PLA Basic, TDS V2) | 1.24 g/cm³; Young's modulus 2680 ± 130 MPa (X-Y), 2160 ± 90 (Z); tensile strength 39 ± 2 / 35 ± 3 MPa; bending strength 76 / 68 MPa; elongation 12.2 / 7.5%; HDT 57 °C (0.45 MPa); saturated water absorption 0.43% (25 °C, 55% RH) | https://cdn.shopify.com/s/files/1/0380/5149/files/Bambu_PLA_Basic_Technical_Data_Sheet.pdf | VERIFIED |
| PETG (Prusament PETG, Prusa Polymers data sheet) | 1.27 g/cm³; tensile modulus 1.5 ± 0.1 GPa (two flat orientations), 1.4 ± 0.1 (third); tensile yield 47 ± 2, 50 ± 1, 30 ± 5 MPa by orientation; filament yield 46 ± 1 MPa; HDT 68 °C (0.45 MPa); moisture absorption 0.2% in 24 h, 0.3% in 7 days, 0.3% in 4 weeks | https://cdn.shopify.com/s/files/1/1745/6181/files/PETG_TechSheet_ENG.pdf | VERIFIED, with a caveat: the PDF's text layer is font-substituted; digits were decoded with the substitution fixed by the rows "1.27 / ISO 1183", "ISO 527", "ISO 75" |
| Carbon-filled nylon (Bambu PAHT-CF, TDS V2; dry state) | 1.06 g/cm³; Young's modulus 3860 ± 230 MPa (X-Y), 2180 ± 130 (Z); tensile strength 92 ± 7 / 47 ± 5 MPa; bending modulus 4230 / 1820 MPa; bending strength 125 / 61 MPa; HDT 194 °C (0.45 MPa); saturated water absorption 0.88% (25 °C, 55% RH) | https://uwaterloo.ca/rapid-prototyping-centre/sites/default/files/uploads/documents/bambu_paht-cf_technical_data_sheet_v2.pdf | VERIFIED |
| Carbon-filled nylon (Prusament PA11 Carbon Fiber, Polish-language TDS) | 1.11 g/cm³; tensile strength 42 ± 1 and 49 ± 2 MPa, tensile modulus 2.5 ± 0.1 and 3.3 ± 0.1 GPa, flexural strength 63 ± 2 and 103 ± 3 MPa, flexural modulus 3.0 ± 0.1 and 6.2 ± 0.3 GPa (two print orientations, column headings not read); filament tensile 61 ± 3 MPa; interlayer adhesion 20 ± 5 MPa; HDT 192 °C (0.45 MPa) | https://download.kamami.pl/p1201932-prusament-pa11cf-karta-danych-technicznych.pdf | VERIFIED |
| TPU (Bambu TPU 95A, TDS) | 1.20 g/cm³; Shore 95A; Young's modulus 9.2 ± 0.4 MPa (X-Y), 7.8 ± 0.5 (Z); tensile strength 29.6 / 23.2 MPa; elongation > 700% / > 500%; saturated water absorption 1.16% | https://cdn.shopify.com/s/files/1/0645/5876/0155/files/Bambu_TPU_95A_Technical_Data_Sheet.pdf | VERIFIED |
| ABS / ASA, unfilled PA12 | UNKNOWN (data sheets not retrieved) | — | UNKNOWN |

The water-absorption figures above are at 55% relative humidity or short immersion; none of the sheets gives
properties after immersion, and the PAHT-CF mechanical values are stated for the dry state only.

### C.2 Pultruded carbon rod and tube as sold to hobbyists

| item | numbers | URL | label |
|---|---|---|---|
| Goodwinds rods < 0.156 in, standard modulus (minimum properties) | Tensile modulus 134 GPa, tensile strength 2.34 GPa, compressive modulus 131 GPa, compressive strength 1.90 GPa, fibre volume 65%, strain to failure 1.30% | https://www.aircraftspruce.com/catalog/pdf/Goodwinds%20RodsTubes%20Specs.pdf | VERIFIED |
| Goodwinds rod mass per metre | 0.94 mm 1.10 g/m; 1.32 mm 2.20 g/m; 1.60 mm 3.30 g/m; 2.49 mm 7.72 g/m; 3.18 mm 13.24 g/m; 3.96 mm 19.85 g/m | same | VERIFIED |
| Goodwinds rods > 0.156 in and tubes | Rods: tensile modulus 138 GPa, flexural modulus 131 GPa, flexural strength 1.83 GPa, 1.5 g/cm³. Tubes: tensile modulus 134 GPa, flexural modulus 127 GPa, flexural strength 1.37 GPa, 1.5 g/cm³ | same | VERIFIED |
| Easy Composites pultruded 3 mm rod, 1 m | 0.011 kg gross | https://www.easycomposites.co.uk/3mm-carbon-fibre-rod | VERIFIED |
| Easy Composites pultruded tube 6 mm OD / 4 mm ID, 1 m | 0.0235 kg gross; wall 1 mm; tolerance ± 0.1 mm; Tg onset 80 °C | https://www.easycomposites.co.uk/6mm-pultruded-carbon-fibre-tube | VERIFIED |
| Easy Composites pultrusion data sheet (all rods, tubes, strips) | Tensile modulus lengthways 28–40 GPa; flexural modulus lengthways 20–30 GPa; tensile strength 400–500 MPa; flexural strength 250–400 MPa; compressive strength 200–320 MPa; density 1.30–1.50 g/cm³; maximum service temperature 80 °C; "typical", "not to be used for specifications" | https://media.easycomposites.eu/datasheets/EC-TDS-Carbon-Fibre-Pultrusions.pdf | VERIFIED |
| The two suppliers against each other | Density from the listed masses: 11 g / (π × 0.15² × 100 cm³) = 1.56 g/cm³ and 23.5 / (π × (0.3² − 0.2²) × 100) = 1.50 g/cm³ (Easy Composites); 13.24 / (π × 0.159² × 100) = 1.67 g/cm³ (Goodwinds). Stated tensile modulus differs by 134 / 34 = 3.9× at about the same density. Bending stiffness of a 6/4 tube, EI with I = π(6⁴ − 4⁴)/64 mm⁴ = 51.05 mm⁴: 6.8 N·m² at 134 GPa, 1.7 N·m² at 34 GPa | | INFERRED |
| E-Flap's choice | 6 mm woven carbon tube as spar, 1.5 mm carbon chord rods every 15 cm | B.1 URL | VERIFIED |

### C.3 Skins

| item | numbers | URL | label |
|---|---|---|---|
| Ripstop nylon used on the E-Flap ("Mirai") | 48 g/m²; "highly tear resistant"; joined with cyanoacrylate or ripstop tape | B.1 URL | VERIFIED |
| Polyester film, Mylar A 50–125 µm (DuPont Teijin sheet) | Tensile strength 180–190 MPa (MD), 200–210 (TD); modulus 3600–3800 MPa (MD), 3800–4100 (TD); elongation 140–150% (MD), 120–130% (TD); usable −70 to 150 °C; also sold at 12–36 µm | https://media.easycomposites.co.uk/datasheets/Mylar_A_50_to_125mic.pdf | VERIFIED |
| Mylar A 12 µm | Tensile strength 200 MPa (MD), 220 MPa (TD); 48–50 gauge grades: modulus 550 kpsi (3.8 GPa), Graves tear 0.7 lb | search summary of DuPont Teijin sheets | SNIPPET |
| Areal mass of PET film | 12 µm × 1.39 g/cm³ = 16.7 g/m²; 23 µm = 32 g/m²; 50 µm = 70 g/m² (density is the handbook value for PET, not on the fetched sheet) | | INFERRED |
| Icarex PC31 (coated polyester ripstop, kite cloth) | "31 g" per unit area; the one listing found prints g/yd², the trade name implies g/m² | https://modelemax.pl/en/icarex/36600-icarex-pc31-fabric-width-140-cm-price-per-1-running-meter-red | SNIPPET |
| Kite and paraglider ripstop | Listings at 40–60 g/m²; "Skytex 27", PU-coated nylon, 27 g/m²; "0.75 oz" spinnaker nylon measured above 1 oz/yd² (34 g/m²) finished | https://kitebuilder.com/wiki/pmwiki.php/Reference/RipstopNylonFabric.html ; https://www.extremtextil.de/en/ripstop-nylon-f.-paraglider-skytex-27-pu-coated-27g-sqm-special-price/73085.ARCTIC | SNIPPET |
| MIT Phoenix skin | "12 oz polycarbonate coated ripstop polyester" as printed (the figure is implausible for kite cloth and is probably ½ oz); Dacron reinforcement glued where the wing meets spars and frame | B.1 URL | VERIFIED as printed |
| TPU film | 0.10 mm × 1.20 g/cm³ = 120 g/m²; modulus 9.2 MPa (printed TPU 95A, C.1) against 3800 MPa for PET film: 410× more compliant | | INFERRED |
| Thin films and membranes on small flappers | 5 µm polyester (T3), 10 µm Mylar (KUBeetle-S), 15 µm polyester (KU Leuven), 40 g/m² PE-laminated non-woven (NUS), 56 µm silicone (Bat Bot) | A11 section A | — |
| Latex membrane | Used on a NASA/UF flexible-wing MAV with a carbon leading edge and battens | https://ntrs.nasa.gov/api/citations/20040031588/downloads/20040031588.pdf | SNIPPET |
| Glue weight | Caltech's carbon rod, Mylar and glue wings: "glue added significant weight" | https://www.ornithopter.org/archive/caltech.pdf | SNIPPET |

### C.4 Whole wings at 0.5–1.5 m span

| item | numbers | URL | label |
|---|---|---|---|
| E-Flap wing set | 82 g for 1.5 m span, 0.36 m root chord, 0.29 m mean chord, 0.438 m²; "55 gram per meter"; built "in less than 3 hours with only standard 3D-printing equipment"; PLA leading and trailing edge pieces, Dyneema tensioner, 3-D printed ring connectors | B.1 URL | VERIFIED |
| E-Flap wing, derived | 82 / 0.438 = 187 g/m² complete; skin 0.438 × 48 = 21 g (26%); frame and fittings 61 g (74%); wing set is 82 / 510 = 16% of the vehicle | | INFERRED |
| Kestrel | 0.22 m² at 395 g: wing loading 1.78 kg/m² (17.5 N/m²) | B.1 URL | VERIFIED |
| E-Flap wing loading | 0.510 / 0.438 = 1.16 kg/m² (11.4 N/m²) empty; 2.35 kg/m² at the 1.03 kg it flew with full payload | | INFERRED |
| Robo Raven, SmartBird, Phoenix wing mass | UNKNOWN | — | UNKNOWN |

---

## D. Electronics and waterproofing

| item | numbers | URL | label |
|---|---|---|---|
| ESC for a 1.2 kg ornithopter (Jeti Advance 30) | 35 g; 30 A; "about 300 watts at 11.4 volts"; shut down above 30 A in flight tests | B.1 Phoenix URL | VERIFIED |
| Single 35 A BLHeli_32 / AM32 ESCs, 3–6S | 3.4 g (HGLRC T-Rex, without wires), 4.4–4.5 g bare and 8.1 g with wires (DYS Aria), 4.8 g (Lumenier), 6.5 g (Flycolor Raptor 5), 6.7 g (T-Motor F35A) | https://oscarliang.com/dys-aria-35a-esc/ ; https://www.getfpv.com/tiger-motor-f35a-3-6s-esc-blheli-32.html | SNIPPET |
| Flight controller with IMU and barometer (Matek F405-WMN) | 10 g; 31 × 26 × 16.5 mm; ICM42688-P IMU, SPL06-001 barometer; 6–30 V input; BEC 5 V 1.5 A. The sibling F405-WMO: 9 g, 90 A current sensor | https://rotorama.com/product/matek-f405-wmn | SNIPPET |
| E-Flap avionics | Carrier PCB 10 g (microcontroller, power regulation for 6 servos) plus a NanoPi companion computer and a VectorNav VN-200; three power buses, servo bus at 5–7 V | B.1 E-Flap URL | VERIFIED |
| Phoenix avionics (2009) | Computer boards 43.5 g, IMU 25.8 g, servo controller 5 g, servo multiplexer about 30 g | B.1 Phoenix URL | VERIFIED |
| Depth sensor, Blue Robotics Bar30 (MS5837-30BA in an aluminium M10 bulkhead penetrator) | 14 g in air; 0–30 bar (about 300 m); resolution 0.2 mbar (2 mm of fresh water); absolute accuracy ± 200 mbar (± 2.04 m) at 0–45 °C; 2.5–5.5 V, I²C at 0x76, peak 1.25 mA; maximum continuous submersion 24 h, the gel must dry at least 2 h daily; only the front face is sealed | https://bluerobotics.com/store/electronics/bar30-sensor-r1/ | VERIFIED |
| Depth sensor, Blue Robotics Bar02 (MS5837-02BA) | 10 m limit (2 bar absolute); resolution 0.16 mm of water; worst-case accuracy ± 4 mbar (4 cm); same drying requirement | https://www.robotshop.com/products/bluerobotics-bar02-ultra-high-resolution-10m-depth-pressure-sensor | SNIPPET |
| Receiver mass | UNKNOWN | — | UNKNOWN |
| Waterproof servo rating | Hitec HS-5086WP and D646WP: IP67, defined on the page as dust tight and "Immersion up to 1M" | B.2 URLs | VERIFIED |
| Brushless motors flooded | Blue Robotics staff: "brushless motors can run underwater"; their M100/M200 "have special coatings and components to do a better job of preventing corrosion than stock brushless motors"; no lifetime given | https://discuss.bluerobotics.com/t/water-proofing-motors/4649 | VERIFIED (forum statement by the vendor) |
| Potting | Blue Robotics engineer: potting is "not a reliable solution" as they had done it; failure attributed to delamination between compound and anodised penetrator under temperature swings; the potted BlueESC was discontinued | https://discuss.bluerobotics.com/t/potting-escs-why-was-the-blueesc-discontinued/10214 | SNIPPET |
| Sealing a servo that was not sold waterproof | Blue Robotics reply: "seems extremely challenging" | https://discuss.bluerobotics.com/t/waterproofing-advice-for-a-non-br-servo-motor-in-a-mudflat-robot-any-guidance-appreciated/23220 | SNIPPET |
| Conformal coating on flight electronics | Hobby sources: protects against "light moisture, condensation, and brief splashes"; not a substitute for a sealed build; coating inside motors can seize bearings; open bearings are the part that rusts; salt water is the main threat | https://oscarliang.com/waterproofing-drone-electronics ; https://www.unmannedtechshop.co.uk/blogs/knowledge-base/fpv-drone-conformal-coating-weatherproof-rain-uk-guide | SNIPPET |
| A flapping machine that swims | FAAV: one "waterproof motor", tested in a tank and a lake; method not described on the pages read | A9, A11 | — |
| Magnetic couplings, O-ring shaft seals, grease-filled servos | No source with a depth or duration figure was found | — | UNKNOWN |

---

## E. Reference machines with measured endurance or power

### E.1 Flapping flyers

| item | numbers | W per kg of vehicle (INFERRED) | URL | label |
|---|---|---|---|---|
| E-Flap | 0.510 kg; 1.5 m; 16.5 V, 450 mAh; 5 min in manual free flight; "an additional 93 g of battery capacity" gave 15 min; cruise 3.5–4 Hz, max 5.5 Hz; 2–6 m/s; flew with up to 520 g payload | 16.5 × 0.45 = 7.43 Wh in 5 min = 89 W = 175 W/kg (upper bound: assumes the pack was emptied). 15 min at 89 W = 22.3 Wh; the added 14.9 Wh in 93 g is 160 Wh/kg, consistent with a LiPo. At 0.603 kg: 148 W/kg. Peak 200 W = 392 W/kg | B.1 URL | VERIFIED |
| MIT Phoenix | 1.2 kg; 1.8 m; 147 W mean; ≈ 4 Hz; ≈ 4 m/s; 3S 2200 mAh | 147 / 1.2 = 122 W/kg; 11.1 V × 2.2 Ah = 24.4 Wh would last 10 min | B.1 URL | VERIFIED |
| Festo SmartBird | 0.450 kg; span 2.00 m (1.96 m on another Festo page); length 1.07 m; 23 W; LiPo 2 cells 7.4 V 450 mAh. One press article reads 23 W as take-off and 12 W as flight | 23 / 0.45 = 51 W/kg (27 W/kg at 12 W); 3.33 Wh ÷ 23 W = 8.7 min | https://www.festo.com/PDF_Flip/corp/smartbird_en/files/assets/seo/page54.html ; https://www.controleng.com/robotic-bird-flies-at-pack-expo-2012/ | SNIPPET |
| HIT-Hawk (Pan et al. 2021) | 2.0 m (2.3 m in other sources); 1.15 kg take-off; 65 min on 3S 4300 mAh | 11.1 × 4.3 = 47.7 Wh in 65 min = 44 W = 38 W/kg | https://doi.org/10.1016/j.birob.2021.100002 (not fetched; search summary of the abstract) | SNIPPET |
| HIT-Phoenix | 2.3 m; 0.86 kg; 8 min on 3S 800 mAh | 8.88 Wh in 8 min = 67 W = 77 W/kg | same | SNIPPET |
| Cloud Owl (NWPU, Appl. Sci. 2023) | 1.82 m; 0.98 kg; 4–6 Hz; 45°; 6–13 m/s; 154 min on 16.8 V, 10.5 Ah | 176 Wh (155 Wh at 14.8 V) in 154 min = 61–69 W. A 155–176 Wh battery cannot weigh less than about 0.6 kg at 268 Wh/kg, so 0.98 kg is unlikely to be the all-up mass: W/kg not computed | https://mdpi.com/2076-3417/13/9/5667 (403 on fetch) | SNIPPET |
| Robo Raven I / II | 0.29–0.38 kg; 1.1–1.33 m; 4 Hz; 5.6–6.7 m/s; average power 13.6–17.6 W in Robo Raven II flight tests | 36–46 W/kg if the figure is the vehicle's total; 71–92 W/kg if it is per wing | B.2 URL | VERIFIED (ambiguity stated in B.2) |
| Kinkade Slow Hawk 2 | 2S 1200 mAh; 9–18 min | 8.88 Wh in 9–18 min = 30–59 W; mass UNKNOWN | B.1 URL | SNIPPET |
| Dove | 220 g; 50 cm; 30 min | UNKNOWN | B.1 URL | SNIPPET |
| MetaFly | < 10 g; span 29 cm; 8 min per charge; range 100 m | UNKNOWN | https://newatlas.com/metafly-flying-model-insect/58765 | SNIPPET |
| Hobby rule | 100 W/kg | 100 W/kg | B.1 URL | VERIFIED |
| Hover-capable flappers | Flapper Nimble+ ≤ 210 W/kg at 127 g; KUBeetle-S 209–222 W/kg at 16 g | A11 section A | A11 | — |

Electrical power per kg of vehicle in forward flapping flight, as found (INFERRED): 38 and 51 W/kg (2 m span,
0.45–1.15 kg, both SNIPPET), 122 W/kg (1.8 m, 1.2 kg, VERIFIED), 148–175 W/kg (1.5 m, 0.51–0.60 kg, VERIFIED,
upper bound), 100 W/kg (hobby rule). Hover: about 210–220 W/kg (A11).

### E.2 Swimmers and flipper machines

| item | numbers | W per kg (INFERRED) | URL | label |
|---|---|---|---|---|
| Tunabot (Zhu et al., Sci. Robot. 2019), tail-beat swimmer | 0.306 kg; 255 mm; up to 15 Hz and 4.0 body lengths/s; range ≈ 9.1 km at 0.4 m/s or ≈ 4.2 km at 1.0 m/s on a 10 Wh battery; cost of transport is lowest near 1.6 body lengths/s; tested on a power tether | 10 Wh over 9100 m at 0.4 m/s (6.32 h) = 1.58 W = 5.2 W/kg; over 4200 m at 1.0 m/s (1.17 h) = 8.6 W = 28 W/kg | https://sites.harvard.edu/glauder/files/2022/03/Zhu.etal_.201918987.pdf | VERIFIED |
| "Crush" sea-turtle robot (arXiv 2602.21389), 10 motors, flippers | 0.208 m/s (0.32 body lengths/s); mean 24.0 W in steady swimming; cost of transport 1.18 | mass from COT = P/(m g v): 24.0 / (1.18 × 9.81 × 0.208) = 10.0 kg; 2.4 W/kg | https://arxiv.org/pdf/2602.21389 | SNIPPET |
| Prosthetic-flipper turtle robot (Garcia et al. 2023) | 0.487 m/s average with a 6 m tether, 0.618 m/s calculated maximum | UNKNOWN | https://openrepository.aut.ac.nz/items/f35e44a8-99f3-409c-98c8-441562e4655b | SNIPPET |
| FAAV, ZS-Puffin, RoboBee | A9 sections 2–3, A11 section B | A9, A11 | — | — |
| Flipper walkers (FlipperBot 790 g, MuddyBot, DALER, T3 ground mode) | speeds in A11 section C; electrical power on land: UNKNOWN for all | A11 | — | — |

---

## What a designer can take from this

**Everything in this section is this report's inference, not a finding of any source.** Each limit names the
rows it rests on.

1. **Cell family for a 15-minute high-power leg.** A high-drain cell gives 180–226 Wh/kg in 15 minutes; a
   high-energy cell is current-limited to 94–165 Wh/kg (A.2: 35E 132, MJ1 165, 50E 121, M50 94 against P28A 199,
   P42A 190, P45B 195, 30Q 193, 40T 180, VTC6 226 at 10 A). Use e_cell = 190 Wh/kg at 4C as the bound, to the
   datasheet cut-off and for new cells at 23 °C.
2. **Pack-level specific energy.** Welded soft pack: 190 × 0.96 = 182 Wh/kg (A.3: 420 g pack on 404 g of cells).
   Plastic holders: 9.2 Wh / (46 + 9.5 g) = 166 Wh/kg. A protection board adds 4–12 g per pack.
3. **Power per cell.** Sustained for 15 minutes: 37 W (P28A, 30Q), 51–55 W (40T, P42A, P45B). At the datasheet
   continuous maximum: 48 W (30Q), 102 W (P28A), 111 W (40T), 130–132 W (P42A, P45B), i.e. 1.0–2.2 kW/kg, at a
   terminal voltage of 2.9–3.2 V (A.2).
4. **Flight power bracket for the simulation.** Forward flapping flight: 122 W/kg (Phoenix) to 148–175 W/kg
   (E-Flap) are the verified measurements at 0.5–1.2 kg; 38–51 W/kg (HIT-Hawk, SmartBird; 2 m span) are
   unverified lower readings; hover-capable machines sit at 210–220 W/kg (A11). A design that needs less than
   38 W/kg or reports more than 2.2 kW per kg of battery is outside what was found (E.1, A.2).
5. **Battery mass fraction for 15 minutes of flight.** fraction = 0.25 h × w / 182 Wh/kg: 5.2% at 38 W/kg,
   7.0% at 51, 13.7% at 100, 16.8% at 122, 24.0% at 175, 28.8% at 210, 35.0% at 255 W/kg (E.1; limit 2).
6. **All-up mass one 21700 can fly for 15 minutes.** m = 13.65 Wh / (0.25 h × w), with a P45B at 4C: 0.55 kg at
   100 W/kg, 0.45 kg at 122, 0.31 kg at 175, 0.26 kg at 210; 1.07–1.44 kg at the unverified 51–38 W/kg
   (A.2, E.1). The cell alone is 70 g, so at 175 W/kg it is 22% of a 0.31 kg machine.
7. **Series count sets a floor on battery mass.** The two verified drives run on 16.5 V (E-Flap) and 11.4 V
   (Phoenix); one cell under a 4C load gives 3.3 V. Three P28A in series: 138 g of cells, 27.6 Wh in 15 minutes,
   111 W sustained. If the battery may be 24–35% of the machine (limit 5 at 175–255 W/kg), the smallest machine
   with a 3S 18650 pack is 138 × 1.04 / 0.35 = 0.41 kg to 138 × 1.04 / 0.24 = 0.60 kg. Four 21700 in series are
   280 g of cells (A.1, B.1).
8. **Swimming and walking are small against flight in the energy budget, where measured.** 15 minutes of flight
   at 175 W/kg is 43.8 Wh per kg of machine; 15 minutes of swimming at the Tunabot's 5.2–28 W/kg is 1.3–7.0 Wh/kg,
   3–16% of it (E.2). No electrical power figure for a flipper-supported walker was found, so the land leg has no
   measured bound.
9. **Geared brushless motor and crank: mass, torque, frequency.** 71 g of motor and transmission delivers an
   estimated 130 W mechanical peak (1.8 kW per kg of drive), 6.5 N·m at the hinge and up to 5.5 Hz on a 1.5 m,
   0.51 kg machine, i.e. 14% of its mass (E-Flap). A 24:1 drive on a 1.2 kg, 1.8 m machine saw 0.35 × 24 = 8.4 N·m
   at 4 Hz and was designed for 17.7 N·m (Phoenix). A hobby gearbox alone for a 0.5 kg model weighed 88 g
   (BirdKit). Reductions in use: 24:1 to 97.8:1 (B.1).
10. **Shoulder torque against span.** With the thesis's span³ rule anchored on the Kestrel (4.43 N·m at 1.17 m
    span, 6 Hz, 1.78 kg/m²), τ = 4.43 × (b / 1.17)³: 0.60 N·m at 0.6 m, 1.4 N·m at 0.8 m, 2.8 N·m at 1.0 m,
    4.8 N·m at 1.2 m, 9.3 N·m at 1.5 m. The E-Flap's own estimate at 1.5 m is 6.5 N·m, 70% of the scaled value,
    so the rule holds to about 30% between these two machines; the anchor itself contains an underived linkage factor (B.1).
11. **A hobby servo as the flapping actuator.** P_max = τ_stall ω₀ / 4 gives 0.46–9.9 W per servo of 9–81 g,
    22–168 W/kg, against 1.8 kW/kg for the geared drive of limit 9: 11–83× less per kilogram (1830/168, 1830/22). A ±30° sinusoid
    is tracked unloaded up to 2.0–6.4 Hz (10.6 Hz for one 70 g tail servo with 0.61 N·m), and at half of that at
    the maximum-power point. Two 72 g servos of the fastest standard class give 2 × 9.4 = 18.8 W and
    2 × 2.16 N·m stall; the one servo-per-wing machine that flew weighed 0.29–0.38 kg, flapped at 4 Hz and drew
    13.6–17.6 W on average (B.2). The servos sold as waterproof (IP67, 1 m) are the weakest per kilogram:
    0.62 W at 28.5 g and 1.86 W at 61 g, 22–31 W/kg, 2.0–2.1 Hz unloaded at ±30°, hence about 1 Hz at maximum
    power.
12. **Structure, skin and fixed electronics.** A complete wing set built with printed fittings, carbon and
    48 g/m² ripstop weighed 187 g per m² of wing and 16% of the machine; the skin is 26% of that (C.4).
    Spar stiffness is uncertain by 3.9× between two suppliers' data for the same product type (EI of a 6/4 tube:
    6.8 or 1.7 N·m², 23.5 g/m), so the simulation should sweep it or the builder should measure the tube bought
    (C.2). Printed parts: 1.06–1.27 g/cm³, modulus 1.5–3.9 GPa in the layer plane, strength 39–92 MPa in the
    plane and 30–47 MPa across layers for the rigid grades, water uptake 0.2–1.2% (C.1). Fixed electronics found: flight controller
    10 g, 35 A ESC 3.4–8.1 g, depth sensor 14 g, receiver not found; 27–32 g before wiring and sealing (D).
    Nothing found supports a depth rating beyond 1 m for a bought servo, and no source gives a lifetime for
    home-made sealing (D).

---

## Not found

- **Torque-speed curves for hobby servos.** None on the KST, Hitec, Robotis or reseller pages fetched (the Robotis
  page lists a performance graph, but only stall torque and no-load speed came through as text). All servo power
  limits are from two end points. Savox manufacturer pages returned HTTP 429 twice; Futaba's page returned 404.
- **Robo Raven's servo model, battery, flapping amplitude, wing mass, and whether Table 2's power is per wing.**
  The retrospective (PMC10603970) does not name them in the first 100,000 characters read; the UMD technical
  report (core.ac.uk) returned 403.
- **Festo SmartBird and BionicSwift details** (gear ratio, flapping frequency, servo masses, BionicSwift flight
  time): Festo's brochure and media PDFs returned 403; what is given is from search summaries.
- **Kinkade Park Hawk / Slow Hawk span, mass, gear ratio and motor data.** Manual-hosting sites were behind bot
  protection; no manufacturer page was found.
- **An open-hardware ornithopter of 0.2–1 kg with a bill of materials and measured mass and flight time.** Two
  searches returned build logs without numbers (Hackster, Instructables, a Notion page); none was opened.
- **A catalogue of small brushless motors with mass, Kv and rated power** beyond the two motors in B.1 was not
  compiled; the Phoenix motor's mass is not in the thesis text read.
- **HIT-Hawk, HIT-Phoenix, Cloud Owl and the HIT power-modelling paper**: MDPI and the abstract mirror returned
  403/404; numbers are from search summaries, and Cloud Owl's all-up mass is inconsistent with its battery.
- **Cells:** VTC6 impedance and continuous-current rating (not in the sheet's extracted text); NCR18650GA
  internal resistance; P45B release-version datasheet (Molicel's own URL returned 404; the sheet read is ver. 0.2);
  P28A/P42A rate curves exist only as figures, so their 15-minute capacity factors (0.97, 0.95) are assumed.
  Capacity at low temperature or in water-cooled operation was not looked for.
- **Filaments:** Prusament PLA data sheet (file over the fetch size limit), ABS/ASA and unfilled PA12 data sheets
  (URLs returned 404). No sheet gives properties after immersion.
- **Skins:** Icarex PC31 areal mass with a consistent unit; tear strength of any ripstop in N; data for TPU film,
  latex sheet or silicone sheet as products (areal mass is from density × thickness only); PET density on the
  fetched Mylar sheet; kitebuilder.com's fabric table (connection reset twice).
- **Adhesives and tapes:** only the E-Flap's statement (cyanoacrylate, ripstop tape) and one remark on glue
  weight; no mass per metre of joint, no peel strength.
- **Wing mass** of Robo Raven, SmartBird, Phoenix, Slow Hawk.
- **Electronics:** receiver mass (the one search returned nothing on it); flight-controller and ESC masses are
  shop listings (Matek's own page redirected); no continuous-current rating for the flight-controller board.
- **Waterproofing:** no measured depth-and-duration result for conformal coating, potting, grease-filled servos,
  O-ring shaft seals or magnetic couplings; how published amphibious robots sealed their actuators was not
  retrieved beyond what A9 and A11 hold. Radio link behaviour under water was not searched.
- **Power on land** for any wing- or flipper-supported walker; swimming power for a flapping-wing (not tail-beat)
  swimmer of 0.2–1.5 kg.
- **MetaFly / MetaBird / Bionic Bird** battery capacity and flapping frequency; Dove's motor and power.
