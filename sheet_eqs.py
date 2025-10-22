# model equations
# ============= Inhibitory Neurons =============
eqs_R = '''
dv/dt = (IsynEI + Isyn_ii + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
IsynEI : 1
Isyn_ii:1   
'''

Ach_eqs_EI = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEI_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

GABA_eqs = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_i_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

GABA_eqs_i = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_ii_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# ============= EPG =============

eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i + Isyn_PxE + Isyn_PyE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PxE_2 : 1
Isyn_PxE_1 : 1
Isyn_PyE_2 : 1
Isyn_PyE_1 : 1
Isyn_PxE = Isyn_PxE_2 + Isyn_PxE_1:1
Isyn_PyE = Isyn_PyE_2 + Isyn_PyE_1:1
'''

# ============= PEN =============
eqs_PENx = '''
dv/dt = (Isyn_ppx + Isyn_EPx + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_ppx : 1
Isyn_EPx : 1
'''

eqs_PENy = '''
dv/dt = (Isyn_ppy + Isyn_EPy + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_ppy : 1
Isyn_EPy : 1
'''

# ============= EPG -> EPG =============

Ach_eqs_EE = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# ============= PEN -> PEN =============
Ach_eqs_PPx = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_ppx_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PPy = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_ppy_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# ============= PEN -> EPG =============


Ach_eqs_EPx = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EPx_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PxE_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PxE_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EPy = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EPy_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PyE_1 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PyE_1_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PyE_2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PyE_2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

#dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
