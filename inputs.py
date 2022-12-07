from pathlib import Path

input_path = Path(__file__)
output_path = Path(__file__).parent/'output'

##############################  Thermal model  ################################

thermal_conf = {
    'output_path': output_path/'thermal_01',
    'default_model': 'TM3'
}

thermal_constants = {
    'Tp':     1350.0,    # [ºC]    Mantle potential temperature
    'G':      0.0004,    # [K/m]   Adiabatic gradient
    'kappa':  1e-06,     # [m2/s]  Thermal diffusivity
    'alpha':  20.0,      # [º]     Angle of subduction
    'V':      6.6e4,     # [m/Ma]  Convergence rate
    'b':      1.0,
}

TM1_input = {
    'delta':  None,      # [km]    Radiogenic decay
    'H0':     2.5e-06,   # [W/m3]  Heat production at surface
    'k':      2.25,      # [W/mK]  Thermal conductivity
    'constants': thermal_constants
}

TM2_input = {
    'Huc':    1.27e-06,  # [W/m3]  Heat production (upper crust)
    'Hlc':    4.20e-07,  # [W/m3]  Heat production (lower crust)
    'k':      2.25,      # [W/mK]  Thermal conductivity
    'constants': thermal_constants
}

TM3_input = {
    'Huc':    1.27e-06,  # [W/m3]  Heat production (upper crust)
    'Hlc':    4.20e-07,  # [W/m3]  Heat production (lower crust)
    'kuc':    3.0,       # [W/mK]  Thermal conductivity (upper crust)
    'klcm':   2.1,       # [W/mK]  Thermal conductivity (lower crust & mantle)
    'constants': thermal_constants
}

thermal_inputs = dict(
    zip(['TM1', 'TM2', 'TM3'], [TM1_input, TM2_input, TM3_input])
)

##############################  Mechanic model  ###############################

mechanic_conf = {
    'output_path': thermal_conf['output_path']/'mechanic_01',
}

mechanic_constants = {
    'Bs_t':     20.e3,   # [MPa]   Stress gradient under tension
    'Bs_c':     -55.e3,  # [MPa]   Stress gradient under compression
    'e':        1e-15,   # [s^-1]  Strain rate (s-1)
    's_max':    200,     #s_max    Maximum available stress
    'R':        8.31,    # [J mol^-1 K^-1] Gas constant
}

MM_input = {
    'uc':       1,       #         Rheologic model id for upper crust
    'lc':       14,      #         Rheologic model id for lower crust
    'lm':       30,      #         Rheologic model id for lithospheric mantle
    'serp':     23,      #         Rheologic model id for forearc serpentinite
    'serp_pct': 0.65,    #         Forearc serpentinization percentage
}
