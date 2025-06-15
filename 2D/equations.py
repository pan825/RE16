# model equations

eqs_R = '''
dv/dt = (IsynEI + Isyn_ii + IsynEIv  + I  + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
IsynEI : 1
Isyn_ii:1   
IsynEIv : 1
'''

# EPG -> R
Ach_eqs_EI = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEI_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# R -> EPG
GABA_eqs = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_i_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# R <-> R
GABA_eqs_i = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_ii_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# EPGv -> R
Ach_eqs_EIv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
IsynEIv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# R -> EPGv
GABAv_eqs = '''
ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
Isyn_iv_post = -s_GABAA*(v-E_GABAA):1 (summed)
wach : 1
'''

# GABAv_eqs_i = '''
# ds_GABAA/dt = -s_GABAA/tau_GABAA : 1 (clock-driven)
# Isyn_iiv_post = -s_GABAA*(v-E_GABAA):1 (summed)
# wach : 1
# '''

eqs_EPG = '''
dv/dt = ( Isyn + Isyn_i +Isyn_PE + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn : 1
Isyn_i : 1
Isyn_PE2R :1
Isyn_PE2L :1
Isyn_PE1R :1
Isyn_PE1L :1
Isyn_PE2R2:1
Isyn_PE2L2 :1
Isyn_PE1R2 :1
Isyn_PE1L2:1
Isyn_PE7:1
Isyn_PE8:1
Isyn_PE = Isyn_PE2R + Isyn_PE2L + Isyn_PE1R + Isyn_PE1L + Isyn_PE2R2 + Isyn_PE2L2 + Isyn_PE1R2 + Isyn_PE1L2 + Isyn_PE7 + Isyn_PE8:1

'''

eqs_EPGv = '''
dv/dt = ( Isyn_v + Isyn_iv +Isyn_PEv + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_v : 1
Isyn_iv : 1
Isyn_PE2Rv :1
Isyn_PE2Lv :1
Isyn_PE1Rv :1
Isyn_PE1Lv :1
Isyn_PE2R2v:1
Isyn_PE2L2v :1
Isyn_PE1R2v :1
Isyn_PE1L2v:1
Isyn_PE7v:1
Isyn_PE8v:1
Isyn_PEv = Isyn_PE2Rv + Isyn_PE2Lv + Isyn_PE1Rv + Isyn_PE1Lv + Isyn_PE2R2v + Isyn_PE2L2v + Isyn_PE1R2v + Isyn_PE1L2v + Isyn_PE7v + Isyn_PE8v:1
'''

eqs_PEN = '''
dv/dt = (Isyn_pp + Isyn_EP + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_pp : 1
Isyn_EP : 1
'''

# EPG <-> EPG
Ach_eqs = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

# PEN <-> PEN
Ach_eqs_PP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_pp_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EP = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EP_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE2R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R2 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R2_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE7 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE7_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE8 = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE8_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''



#########

# model equations

eqs_PENv = '''
dv/dt = (Isyn_ppv + Isyn_EPv + I + E_l - v) / taum + sigma*sqrt(2/taum)*xi: 1 (unless refractory)
I : 1
Isyn_ppv : 1
Isyn_EPv : 1
'''

Ach_eqs_v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PPv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_ppv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_EPv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_EPv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PEv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PEv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2Rv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2Rv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2Lv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2Lv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1Lv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1Lv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1Rv = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1Rv_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE2R2v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2R2v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE2L2v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE2L2v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE1L2v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1L2v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE1R2v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE1R2v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

Ach_eqs_PE7v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE7v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''
Ach_eqs_PE8v = '''
ds_ach/dt = -s_ach/tau_ach : 1 (clock-driven)
Isyn_PE8v_post = -s_ach*(v-E_ach):1 (summed)
wach : 1 
'''

