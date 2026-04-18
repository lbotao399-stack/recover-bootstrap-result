# Figure 6 sparse hierarchy in instanton units

- Vertical variable: `eta = ln(E / E_inst)`.
- Level 7 reuses the earlier 12-point seed scan.
- Levels 8-9 use a 9-point sparse grid inherited from the previous level.
- Level 10 uses a 6-point subset inherited from level 9.
- Level 7: 12 lower points, 12 upper points.
- Level 8: 9 lower points, 9 upper points.
- Level 9: 9 lower points, 9 upper points.
- Level 10: 6 lower points, 6 upper points.
- Current limitation: levels 7-9 are nearly indistinguishable at this tolerance, and very small-`g` points are still clipped by the `eta in [-10,10]` scan window.
