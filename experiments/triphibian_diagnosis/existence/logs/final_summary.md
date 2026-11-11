| item | B1 hand | B0 hand | B1 still | B0 still |
|---|---|---|---|---|
| air | **8/8** >= 0.012; 8/8 >= 0.15; 8/8 >= 0.5; min 0.891 med 0.911 max 0.927 | **8/8** >= 0.012; 8/8 >= 0.15; 8/8 >= 0.5; min 0.913 med 0.924 max 0.937 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 |
| water | **8/8** >= 0.012; 8/8 >= 0.15; 2/8 >= 0.5; min 0.211 med 0.358 max 0.717 | **8/8** >= 0.012; 8/8 >= 0.15; 5/8 >= 0.5; min 0.401 med 0.788 max 0.853 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 |
| land | **8/8** >= 0.012; 8/8 >= 0.15; 8/8 >= 0.5; min 0.711 med 0.744 max 0.761 | **8/8** >= 0.012; 8/8 >= 0.15; 7/8 >= 0.5; min 0.188 med 0.679 max 0.744 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 | **0/8** >= 0.012; 0/8 >= 0.15; 0/8 >= 0.5; min 0.000 med 0.000 max 0.000 |
| air_to_water | **8/8** crossed; hold 0.87-0.87; approach 0.87-0.87 | **8/8** crossed; hold 0.96-0.96; approach 0.96-0.96 | **0/8** crossed; hold 0.00-0.00; approach 0.00-0.00; (did not hold before the command to cross) | **0/8** crossed; hold 0.00-0.00; approach 0.00-0.00; (did not hold before the command to cross) |
| water_to_air | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) |
| water_to_land | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) |
| land_to_air | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **8/8** crossed; hold 1.00-1.00; approach 1.00-1.00 | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) | **0/8** crossed; hold 1.00-1.00; approach 0.00-0.00; (never crossed the boundary) |
| mission_air | **8/8** both transitions; transitions 2/2; continuous_score 0.704-0.704 | **8/8** both transitions; transitions 2/2; continuous_score 0.700-0.700 | **0/8** both transitions; transitions 1/2; continuous_score 0.484-0.484 | **0/8** both transitions; transitions 1/2; continuous_score 0.575-0.575 |
| mission_water | **8/8** both transitions; transitions 2/2; continuous_score 0.775-0.775 | **8/8** both transitions; transitions 2/2; continuous_score 0.763-0.763 | **0/8** both transitions; transitions 0/2; continuous_score 0.337-0.337 | **0/8** both transitions; transitions 0/2; continuous_score 0.350-0.350 |
| mission_land | **8/8** both transitions; transitions 2/2; continuous_score 0.792-0.792 | **8/8** both transitions; transitions 2/2; continuous_score 0.795-0.795 | **0/8** both transitions; transitions 0/2; continuous_score 0.148-0.148 | **0/8** both transitions; transitions 0/2; continuous_score 0.148-0.148 |

Per-seed scores (hand arm):
- B1 air: 0.912 0.905 0.917 0.891 0.927 0.924 0.895 0.910
- B1 water: 0.223 0.717 0.422 0.516 0.425 0.211 0.293 0.218
- B1 land: 0.713 0.741 0.747 0.761 0.751 0.711 0.753 0.737
- B0 air: 0.916 0.922 0.937 0.925 0.925 0.929 0.921 0.913
- B0 water: 0.401 0.853 0.769 0.806 0.809 0.425 0.813 0.418
- B0 land: 0.723 0.627 0.640 0.717 0.723 0.188 0.744 0.584

Seed-independence of the crossing and mission probes (identical score, booked energy and end state of charge on every seed):
- B1 hand: identical on 7 of 7 items
- B0 hand: identical on 7 of 7 items
- B1 still: identical on 7 of 7 items
- B0 still: identical on 7 of 7 items

Battery and booked energy (per rollout; a segment starts at the scatter's state of charge, 0.4-1.0):
| body arm | item | end state of charge min-max | booked Wh min-max | mean power W (mean) | peak draw W max | max actuator overload | failures | bad_qacc |
|---|---|---|---|---|---|---|---|---|
| B1 hand | air | 0.433-0.955 | 0.931-0.984 | 433 | 2345 | 2.86 | - | 0 |
| B1 hand | water | 0.500-0.949 | 0.181-0.376 | 137 | 700 | 2.86 | - | 0 |
| B1 hand | land | 0.453-0.922 | 0.463-0.497 | 214 | 2046 | 2.86 | - | 0 |
| B1 hand | air_to_water | 0.459-0.459 | 0.539-0.539 | 323 | 2440 | 2.86 | - | 0 |
| B1 hand | water_to_air | 0.498-0.498 | 0.523-0.523 | 314 | 2483 | 2.86 | - | 0 |
| B1 hand | water_to_land | 0.922-0.922 | 0.431-0.431 | 258 | 2050 | 2.86 | - | 0 |
| B1 hand | land_to_air | 0.941-0.941 | 0.603-0.603 | 362 | 2505 | 2.86 | - | 0 |
| B1 hand | mission_air | 0.988-0.988 | 2.000-2.000 | 300 | 2295 | 2.86 | - | 0 |
| B1 hand | mission_water | 0.990-0.990 | 1.786-1.786 | 268 | 2509 | 2.86 | - | 0 |
| B1 hand | mission_land | 0.993-0.993 | 1.216-1.216 | 182 | 2505 | 2.86 | - | 0 |
| B0 hand | air | 0.428-0.949 | 0.633-0.662 | 292 | 2050 | 2.86 | - | 0 |
| B0 hand | water | 0.499-0.948 | 0.088-0.170 | 65 | 754 | 2.86 | - | 0 |
| B0 hand | land | 0.450-0.919 | 0.325-0.350 | 150 | 1871 | 2.86 | - | 0 |
| B0 hand | air_to_water | 0.457-0.457 | 0.391-0.391 | 235 | 2157 | 2.86 | - | 0 |
| B0 hand | water_to_air | 0.494-0.494 | 0.466-0.466 | 280 | 2012 | 2.86 | - | 0 |
| B0 hand | water_to_land | 0.920-0.920 | 0.284-0.284 | 171 | 1853 | 2.86 | - | 0 |
| B0 hand | land_to_air | 0.938-0.938 | 0.413-0.413 | 248 | 2058 | 2.86 | - | 0 |
| B0 hand | mission_air | 0.976-0.976 | 1.422-1.422 | 213 | 2019 | 2.86 | - | 0 |
| B0 hand | mission_water | 0.974-0.974 | 1.585-1.585 | 238 | 2172 | 2.86 | - | 0 |
| B0 hand | mission_land | 0.986-0.986 | 0.845-0.845 | 127 | 2058 | 2.86 | - | 0 |
| B1 still | air | 0.438-0.960 | 0.016-0.017 | 7 | 19 | 0.02 | - | 0 |
| B1 still | water | 0.502-0.950 | 0.016-0.016 | 7 | 7 | 0.00 | - | 0 |
| B1 still | land | 0.456-0.925 | 0.016-0.016 | 7 | 8 | 0.00 | - | 0 |
| B1 still | air_to_water | 0.463-0.463 | 0.012-0.012 | 7 | 12 | 0.01 | did not hold before the command to cross | 0 |
| B1 still | water_to_air | 0.502-0.502 | 0.012-0.012 | 7 | 7 | 0.00 | never crossed the boundary | 0 |
| B1 still | water_to_land | 0.925-0.925 | 0.012-0.012 | 7 | 7 | 0.00 | never crossed the boundary | 0 |
| B1 still | land_to_air | 0.944-0.944 | 0.012-0.012 | 7 | 8 | 0.00 | never crossed the boundary | 0 |
| B1 still | mission_air | 1.000-1.000 | 0.049-0.049 | 7 | 8 | 0.00 | - | 0 |
| B1 still | mission_water | 1.000-1.000 | 0.049-0.049 | 7 | 7 | 0.00 | - | 0 |
| B1 still | mission_land | 1.000-1.000 | 0.049-0.049 | 7 | 7 | 0.00 | - | 0 |
| B0 still | air | 0.438-0.960 | 0.016-0.017 | 7 | 32 | 0.05 | - | 0 |
| B0 still | water | 0.501-0.950 | 0.016-0.016 | 7 | 7 | 0.00 | - | 0 |
| B0 still | land | 0.456-0.925 | 0.016-0.016 | 7 | 8 | 0.00 | - | 0 |
| B0 still | air_to_water | 0.463-0.463 | 0.012-0.012 | 7 | 22 | 0.03 | did not hold before the command to cross | 0 |
| B0 still | water_to_air | 0.502-0.502 | 0.012-0.012 | 7 | 7 | 0.00 | never crossed the boundary | 0 |
| B0 still | water_to_land | 0.925-0.925 | 0.012-0.012 | 7 | 7 | 0.00 | never crossed the boundary | 0 |
| B0 still | land_to_air | 0.944-0.944 | 0.012-0.012 | 7 | 7 | 0.00 | never crossed the boundary | 0 |
| B0 still | mission_air | 0.999-0.999 | 0.049-0.049 | 7 | 7 | 0.00 | - | 0 |
| B0 still | mission_water | 0.999-0.999 | 0.049-0.049 | 7 | 7 | 0.00 | - | 0 |
| B0 still | mission_land | 0.999-0.999 | 0.049-0.049 | 7 | 7 | 0.00 | - | 0 |

Energy against the pack, as `evaluate_tier1` extrapolates it (segment mean power x 300 s x 3 cycles + `transition_energy`; usable = 0.85 x pack):
| body | pack Wh | P air W | P water W | P land W | required Wh | usable Wh | usable/required | minutes of the 45 covered |
|---|---|---|---|---|---|---|---|---|
| B1 | 200 lis | 433 | 137 | 214 | 196.1 | 170.0 | 0.87 | 39 |
| B0 | 60 lipo | 292 | 65 | 150 | 127.1 | 51.0 | 0.40 | 18 |
