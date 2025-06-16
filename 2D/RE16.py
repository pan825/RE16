from brian2 import *
import brian2cuda
import numpy as np
import time
from equations import *



def visual_cue(theta, index, stimulus = 0.03, sigma = 2 * np.pi/8):
    """
    param: 
        theta: the angle of the visual input
        index: the index of the neuron
        stimulus: the strength of the visual input
        sigma: the standard deviation of the Gaussian distribution
    """
    A = stimulus
    phi = index * np.pi/8
    d1 = (theta-phi)**2 
    d2 = (theta-phi + 2*np.pi)**2
    d3 = (theta-phi - 2*np.pi)**2
    return A * (np.exp(-d1/(2*sigma**2)) + np.exp(-d2/(2*sigma**2)) + np.exp(-d3/(2*sigma**2)))

def simulator( 
        # parameters
        w_EE = 0.719, # EB <-> EB
        w_EI = 0.143, # R -> EB
        w_IE = 0.740, # EB -> R
        w_II = 0.01, # R <-> R
        w_PP = 0.01, # PEN <-> PEN
        w_EP = 0.012, # EB -> PEN 
        w_PE = 0.709, # PEN -> EB
        sigma = 0.0001, # noise level
        
        stimulus_strength = 0.05, 
        stimulus_location = 0*np.pi, # from 0 to np.pi
        stimulus_location_v = 0*np.pi,
        shifter_strength = 0.015,
        shifter_strength_v = 0.015,
        half_PEN = 'right',
        half_PENv = 'right',
        
        t_epg_open = 200, # stimulus
        t_epg_close = 500,    # no stimulus
        t_pen_open = 5000,   # shift

):
    """
    param:
    w_EE: the weight of the EPG to EPG synapse default 0.772
    w_EI: the weight of the R to EPG synapse default 0.209
    w_IE: the weight of the EPG to R synapse default 0.743
    w_II: the weight of the R to R synapse default 0.01
    w_PP: the weight of the PEN to PEN synapse default 0.01
    w_EP: the weight of the EPG to PEN synapse default 0.008
    w_PE: the weight of the PEN to EPG synapse default 0.811
    sigma: the noise level default 0.001
    stimulus_strength: the strength of the visual input default 0.05
    stimulus_location: the location of the visual input (from 0 to np.pi) default 0*np.pi
    shifter_strength: the strength of the shifter neuron input default 0.015
    ang_vel: the angular velocity of the cue rotation
    activate_duration: give the visual cue
    bump_duration: close the visual cue input
    shift_duration: the duration of the body ratotion
    half_PEN: 'left' or 'right'
    
    """

    start = time.time()
    start_scope()  
    print('start')
    
    taum   = 20*ms   # time constant
    Cm     = 0.1
    g_L    = 10   # leak conductance
    E_l    = -0.07  # leak reversal potential (volt)
    E_e    = 0   # excitatory reversal potential
    tau_e  = 5*ms    # excitatory synaptic time constant
    Vr     = E_l     # reset potential
    Vth    = -0.05  # spike threshold (volt)
    Vs     = 0.02   # spiking potential (volt)
    w_e    = 0.1  	 # excitatory synaptic weight (units of g_L)
    v_e    = 5*Hz    # excitatory Poisson rate
    N_e         = 100     # number of excitatory inputs
    E_ach       = 0
    tau_ach     = 10*ms
    E_achv      = 0        # vertical ACh reversal potential
    tau_achv    = 10*ms    # vertical ACh synaptic time constant
    E_GABAA     = -0.07 # GABAA reversal potential
    tau_GABAA   = 5*ms # GABAA synaptic time constant
    E_GABAAv    = -0.07    # vertical GABAA reversal potential  
    tau_GABAAv  = 5*ms     # vertical GABAA synaptic time constant


    # create neuron
    EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    EPGv = NeuronGroup(48, model=eqs_EPGv, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PENv = NeuronGroup(48,model=eqs_PENv, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron1
    EPG.v = E_l
    PEN.v = E_l
    EPGv.v = E_l
    PENv.v = E_l
    R.v = E_l

    EPG_groups = []
    EPG_groups.append(EPG[0:3]) # EPG1
    EPG_groups.append(EPG[3:6]) # EPG2
    EPG_groups.append(EPG[6:9]) # EPG3
    EPG_groups.append(EPG[9:12]) # EPG4
    EPG_groups.append(EPG[12:15]) # EPG5
    EPG_groups.append(EPG[15:18]) # EPG6
    EPG_groups.append(EPG[18:21]) # EPG7
    EPG_groups.append(EPG[21:24]) # EPG8
    EPG_groups.append(EPG[24:27]) # EPG9
    EPG_groups.append(EPG[27:30]) # EPG10
    EPG_groups.append(EPG[30:33]) # EPG11
    EPG_groups.append(EPG[33:36]) # EPG12
    EPG_groups.append(EPG[36:39]) # EPG13
    EPG_groups.append(EPG[39:42]) # EPG14
    EPG_groups.append(EPG[42:45]) # EPG15
    EPG_groups.append(EPG[45:48]) # EPG16

    EPGv_groups = []
    EPGv_groups.append(EPGv[0:3]) # EPG1
    EPGv_groups.append(EPGv[3:6]) # EPG2
    EPGv_groups.append(EPGv[6:9]) # EPG3
    EPGv_groups.append(EPGv[9:12]) # EPG4
    EPGv_groups.append(EPGv[12:15]) # EPG5
    EPGv_groups.append(EPGv[15:18]) # EPG6 
    EPGv_groups.append(EPGv[18:21]) # EPG7
    EPGv_groups.append(EPGv[21:24]) # EPG8
    EPGv_groups.append(EPGv[24:27]) # EPG9
    EPGv_groups.append(EPGv[27:30]) # EPG10
    EPGv_groups.append(EPGv[30:33]) # EPG11
    EPGv_groups.append(EPGv[33:36]) # EPG12
    EPGv_groups.append(EPGv[36:39]) # EPG13
    EPGv_groups.append(EPGv[39:42]) # EPG14
    EPGv_groups.append(EPGv[42:45]) # EPG15
    EPGv_groups.append(EPGv[45:48]) # EPG16

    PEN_groups = []
    PEN_groups.append(PEN[0:3]) # PEN1
    PEN_groups.append(PEN[3:6]) # PEN2
    PEN_groups.append(PEN[6:9]) # PEN3
    PEN_groups.append(PEN[9:12]) # PEN4
    PEN_groups.append(PEN[12:15]) # PEN5
    PEN_groups.append(PEN[15:18]) # PEN6
    PEN_groups.append(PEN[18:21]) # PEN7
    PEN_groups.append(PEN[21:24]) # PEN8
    PEN_groups.append(PEN[24:27]) # PEN9
    PEN_groups.append(PEN[27:30]) # PEN10
    PEN_groups.append(PEN[30:33]) # PEN11
    PEN_groups.append(PEN[33:36]) # PEN12
    PEN_groups.append(PEN[36:39]) # PEN13
    PEN_groups.append(PEN[39:42]) # PEN14
    PEN_groups.append(PEN[42:45]) # PEN15
    PEN_groups.append(PEN[45:48]) # PEN16

    PENv_groups = []

    PENv_groups.append(PENv[0:3]) # PEN1
    PENv_groups.append(PENv[3:6]) # PEN2
    PENv_groups.append(PENv[6:9]) # PEN3
    PENv_groups.append(PENv[9:12]) # PEN4
    PENv_groups.append(PENv[12:15]) # PEN5
    PENv_groups.append(PENv[15:18]) # PEN6
    PENv_groups.append(PENv[18:21]) # PEN7
    PENv_groups.append(PENv[21:24]) # PEN8
    PENv_groups.append(PENv[24:27]) # PEN9
    PENv_groups.append(PENv[27:30]) # PEN10
    PENv_groups.append(PENv[30:33]) # PEN11
    PENv_groups.append(PENv[33:36]) # PEN12
    PENv_groups.append(PENv[36:39]) # PEN13
    PENv_groups.append(PENv[39:42]) # PEN14
    PENv_groups.append(PENv[42:45]) # PEN15
    PENv_groups.append(PENv[45:48]) # PEN16

    EPG_syn = []
    PEN_syn = []
    PE2R_syn = []
    PE2L_syn = []
    PE1R_syn = []
    PE1L_syn = []
    PE2R_syn2 = []
    PE2L_syn2 = []
    PE1R_syn2 = []
    PE1L_syn2 = []

    EP_syn = []
    EPv_syn = []

    EPGv_syn = []
    PENv_syn = []
    PE2Rv_syn = []
    PE2Lv_syn = []
    PE1Rv_syn = []
    PE1Lv_syn = []
    PE2Rv_syn2 = []
    PE2Lv_syn2 = []
    PE1Rv_syn2 = []
    PE1Lv_syn2 = []
    PE7v_syn = []
    PE8v_syn = []
    EIv_syn = []

    
    # EPG_EPG
    print("Creating EPG-EPG connections...")
    for k in range(0,16):
        # EPG to EPG
        EPG_syn.append(Synapses(EPG_groups[k], EPG_groups[k], Ach_eqs, on_pre='s_ach += w_EE', method='euler'))
        EPG_syn[k].connect(condition='i != j')
    
    # PEN_PEN
    print("Creating PEN-PEN connections...")
    for k2 in range(0,16):
        # PEN to PEN
        PEN_syn.append(Synapses(PEN_groups[k2], PEN_groups[k2], Ach_eqs_PP, on_pre='s_ach += w_PP', method='euler'))
        PEN_syn[k2].connect(condition='i != j')
    
    # EPG_R and R_EPG
    # EPG to R
    print("Creating EPG-R connections...")
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    for a in range(0,48):
        for b in range(0,3):
            S_EI.connect(i=a, j=b)
    
    # R to EPG
    print("Creating R-EPG connections...")
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    for a2 in range(0,48):
        for b2 in range(0,3):
            S_IE.connect(i=b2, j=a2)
    
    # R <-> R
    print("Creating R-R connections...")
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # EPG_PEN synapse
    print("Creating EPG-PEN connections...")
    for k3 in range(0,16):
        # EPG to PEN
        EP_syn.append(Synapses(EPG_groups[k3], PEN_groups[k3], Ach_eqs_EP, on_pre='s_ach += w_EP', method='euler'))
        EP_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    ###
    
    # PEN_EPG synapse #v
    # PEN0-6 -> EPG0-8
    print("Creating PEN0-6 -> EPG0-8 connections (PE2R)...")
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+1], Ach_eqs_PE2R, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating PEN0-6 -> EPG0-8 connections (PE1R)...")
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+2], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating  PE1R connection...")
    PE1R_syn.append(Synapses(PEN_groups[6], EPG_groups[0], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG0-8 
    print("Creating PEN9-15 -> EPG0-8 connections (PE2R2)...")
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+1], Ach_eqs_PE2R2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating PEN9-15 -> EPG0-8 connections (PE1R2)...")
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+2], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating  PE1R2 connection...")
    PE1R_syn2.append(Synapses(PEN_groups[15], EPG_groups[0], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN7-8
    print("Creating PEN7 connections...")
    PE7_syn = []
    # PEN7 -> EPG connections
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[0], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[1], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[15], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[14], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE7_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating PEN8 connections...")
    PE8_syn = []
    # PEN8 -> EPG connections
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[0], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[1], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[15], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[14], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE8_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN0-6 -> EPG8-15
    print("Creating PEN0-6 -> EPG8-15 connections (PE2L)...")
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+8], Ach_eqs_PE2L, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating PEN0-6 -> EPG8-15 connections (PE1L)...")
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn.append(Synapses(PEN_groups[k4+1], EPG_groups[k4+8], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
        
    print("Creating  PE1L connection...")
    PE1L_syn.append(Synapses(PEN_groups[0], EPG_groups[15], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG8-15
    print("Creating PEN9-15 -> EPG8-15 connections (PE2L2)...")
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+8], Ach_eqs_PE2L2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print("Creating PEN9-15 -> EPG8-15 connections (PE1L2)...")
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn2.append(Synapses(PEN_groups[k4+10], EPG_groups[k4+8], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # Additional reciprocal connection if necessary
    print("Creating  PE1L2 connection...")
    PE1L_syn2.append(Synapses(PEN_groups[9], EPG_groups[15], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)

    print('vertical connections')
    #########################
    ## vertical connections
    #########################
    
    # EPGv_EPGv
    print('Creating EPGv-EPGv connections...')
    for k in range(0,16):
        # EPG to EPG
        EPGv_syn.append(Synapses(EPGv_groups[k], EPGv_groups[k], Ach_eqs_v, on_pre='s_ach += w_EE', method='euler'))
        EPGv_syn[k].connect(condition='i != j')
    
    # PENv_PENv
    print('Creating PENv-PENv connections...')
    for k2 in range(0,16):
        # PEN to PEN
        PENv_syn.append(Synapses(PENv_groups[k2], PENv_groups[k2], Ach_eqs_PPv, on_pre='s_ach += w_PP', method='euler'))
        PENv_syn[k2].connect(condition='i != j')
    
    # EPGv to R
    print('Creating EPGv-R connections...')
    Sv_EI = Synapses(EPGv, R, model=Ach_eqs_EIv, on_pre='s_ach += w_EI', method='euler')
    for a in range(0,48):
        for b in range(0,3):
            Sv_EI.connect(i=a, j=b)
    
    # R to EPGv (use vertical GABA synapse model)
    print('Creating R-EPGv connections...')
    Sv_IE = Synapses(R, EPGv, model=GABAv_eqs, on_pre='s_GABAA += w_IE', method='euler')
    for a2 in range(0,48):
        for b2 in range(0,3):
            Sv_IE.connect(i=b2, j=a2)
    
    # EPG_PEN synapse
    print('Creating EPGv-PENv connections...')
    for k3 in range(0,16):
        # EPG to PEN
        EPv_syn.append(Synapses(EPGv_groups[k3], PENv_groups[k3], Ach_eqs_EPv, on_pre='s_ach += w_EP', method='euler'))
        EPv_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    ###
    
    # PEN_EPG synapse #v
    # PEN0-6 -> EPG0-8
    print('Creating PENv0-6 -> EPGv0-8 connections (PE2Rv)...')
    for k4 in range(0,7):
        # PEN to EPG
        PE2Rv_syn.append(Synapses(PENv_groups[k4], EPGv_groups[k4+1], Ach_eqs_PE2Rv, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2Rv_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PENv0-6 -> EPGv0-8 connections (PE1Rv)...')
    for k4 in range(0,6):
        # PEN to EPG
        PE1Rv_syn.append(Synapses(PENv_groups[k4], EPGv_groups[k4+2], Ach_eqs_PE1Rv, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1Rv_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PE1Rv connection...')
    PE1Rv_syn.append(Synapses(PENv_groups[6], EPGv_groups[0], Ach_eqs_PE1Rv, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1Rv_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG0-8 
    print('Creating PENv9-15 -> EPGv0-8 connections (PE2R2v)...')
    for k4 in range(0,7):
        # PEN to EPG
        PE2Rv_syn2.append(Synapses(PENv_groups[k4+9], EPGv_groups[k4+1], Ach_eqs_PE2R2v, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2Rv_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PENv9-15 -> EPGv0-8 connections (PE1R2v)...')
    for k4 in range(0,6):
        # PEN to EPG
        PE1Rv_syn2.append(Synapses(PENv_groups[k4+9], EPGv_groups[k4+2], Ach_eqs_PE1R2v, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1Rv_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PE1R2v connection...')
    PE1Rv_syn2.append(Synapses(PENv_groups[15], EPGv_groups[0], Ach_eqs_PE1R2v, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1Rv_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN7-8
    print('Creating PENv7 connections...')
    PE7v_syn = []
    # PEN7 -> EPG connections
    PE7v_syn.append(Synapses(PENv_groups[7], EPGv_groups[0], Ach_eqs_PE7v, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7v_syn.append(Synapses(PENv_groups[7], EPGv_groups[1], Ach_eqs_PE7v, on_pre='s_ach += 1*w_PE', method='euler'))
    PE7v_syn.append(Synapses(PENv_groups[7], EPGv_groups[15], Ach_eqs_PE7v, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7v_syn.append(Synapses(PENv_groups[7], EPGv_groups[14], Ach_eqs_PE7v, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE7v_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PENv8 connections...')
    PE8v_syn = []
    # PEN8 -> EPG connections
    PE8v_syn.append(Synapses(PENv_groups[8], EPGv_groups[0], Ach_eqs_PE8v, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8v_syn.append(Synapses(PENv_groups[8], EPGv_groups[1], Ach_eqs_PE8v, on_pre='s_ach += 1*w_PE', method='euler'))
    PE8v_syn.append(Synapses(PENv_groups[8], EPGv_groups[15], Ach_eqs_PE8v, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8v_syn.append(Synapses(PENv_groups[8], EPGv_groups[14], Ach_eqs_PE8v, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE8v_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN0-6 -> EPG8-15
    print('Creating PENv0-6 -> EPGv8-15 connections (PE2Lv)...')
    for k4 in range(0,7):
        # PEN to EPG
        PE2Lv_syn.append(Synapses(PENv_groups[k4], EPGv_groups[k4+8], Ach_eqs_PE2Lv, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2Lv_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PENv0-6 -> EPGv8-15 connections (PE1Lv)...')
    for k4 in range(0,6):
        # PEN to EPG
        PE1Lv_syn.append(Synapses(PENv_groups[k4+1], EPGv_groups[k4+8], Ach_eqs_PE1Lv, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1Lv_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
        
    print('Creating PE1Lv connection...')
    PE1Lv_syn.append(Synapses(PENv_groups[0], EPGv_groups[15], Ach_eqs_PE1Lv, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1Lv_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG8-15
    print('Creating PENv9-15 -> EPGv8-15 connections (PE2L2v)...')
    for k4 in range(0,7):
        # PEN to EPG
        PE2Lv_syn2.append(Synapses(PENv_groups[k4+9], EPGv_groups[k4+8], Ach_eqs_PE2L2v, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2Lv_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    print('Creating PENv9-15 -> EPGv8-15 connections (PE1L2v)...')
    for k4 in range(0,6):
        # PEN to EPG
        PE1Lv_syn2.append(Synapses(PENv_groups[k4+10], EPGv_groups[k4+8], Ach_eqs_PE1L2v, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1Lv_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)

    print('Creating PE1L2v connection...')
    # Additional reciprocal connection if necessary
    PE1Lv_syn2.append(Synapses(PENv_groups[9], EPGv_groups[15], Ach_eqs_PE1L2v, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1Lv_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)

    print("All synapse connections completed!")

    # record model state
    PRM0 = PopulationRateMonitor(EPG_groups[0])
    PRM1 = PopulationRateMonitor(EPG_groups[1]) 
    PRM2 = PopulationRateMonitor(EPG_groups[2]) 
    PRM3 = PopulationRateMonitor(EPG_groups[3]) 
    PRM4 = PopulationRateMonitor(EPG_groups[4]) 
    PRM5 = PopulationRateMonitor(EPG_groups[5])
    PRM6 = PopulationRateMonitor(EPG_groups[6]) 
    PRM7 = PopulationRateMonitor(EPG_groups[7]) 
    PRM8 = PopulationRateMonitor(EPG_groups[8])
    PRM9 = PopulationRateMonitor(EPG_groups[9])
    PRM10 = PopulationRateMonitor(EPG_groups[10])
    PRM11 = PopulationRateMonitor(EPG_groups[11])
    PRM12 = PopulationRateMonitor(EPG_groups[12])
    PRM13 = PopulationRateMonitor(EPG_groups[13])
    PRM14 = PopulationRateMonitor(EPG_groups[14])
    PRM15 = PopulationRateMonitor(EPG_groups[15])

    PRM0v = PopulationRateMonitor(EPGv_groups[0])
    PRM1v = PopulationRateMonitor(EPGv_groups[1]) 
    PRM2v = PopulationRateMonitor(EPGv_groups[2]) 
    PRM3v = PopulationRateMonitor(EPGv_groups[3]) 
    PRM4v = PopulationRateMonitor(EPGv_groups[4]) 
    PRM5v = PopulationRateMonitor(EPGv_groups[5])
    PRM6v = PopulationRateMonitor(EPGv_groups[6])
    PRM7v = PopulationRateMonitor(EPGv_groups[7])
    PRM8v = PopulationRateMonitor(EPGv_groups[8])
    PRM9v = PopulationRateMonitor(EPGv_groups[9])
    PRM10v = PopulationRateMonitor(EPGv_groups[10])
    PRM11v = PopulationRateMonitor(EPGv_groups[11])
    PRM12v = PopulationRateMonitor(EPGv_groups[12])
    PRM13v = PopulationRateMonitor(EPGv_groups[13])
    PRM14v = PopulationRateMonitor(EPGv_groups[14])
    PRM15v = PopulationRateMonitor(EPGv_groups[15])

    PRM0p = PopulationRateMonitor(PEN_groups[0])
    PRM1p = PopulationRateMonitor(PEN_groups[1])
    PRM2p = PopulationRateMonitor(PEN_groups[2])
    PRM3p = PopulationRateMonitor(PEN_groups[3])
    PRM4p = PopulationRateMonitor(PEN_groups[4])
    PRM5p = PopulationRateMonitor(PEN_groups[5])
    PRM6p = PopulationRateMonitor(PEN_groups[6])
    PRM7p = PopulationRateMonitor(PEN_groups[7])
    PRM8p = PopulationRateMonitor(PEN_groups[8])
    PRM9p = PopulationRateMonitor(PEN_groups[9])
    PRM10p = PopulationRateMonitor(PEN_groups[10])
    PRM11p = PopulationRateMonitor(PEN_groups[11])
    PRM12p = PopulationRateMonitor(PEN_groups[12])
    PRM13p = PopulationRateMonitor(PEN_groups[13])
    PRM14p = PopulationRateMonitor(PEN_groups[14])
    PRM15p = PopulationRateMonitor(PEN_groups[15])
    
    PRM0pv = PopulationRateMonitor(PENv_groups[0])
    PRM1pv = PopulationRateMonitor(PENv_groups[1])
    PRM2pv = PopulationRateMonitor(PENv_groups[2])
    PRM3pv = PopulationRateMonitor(PENv_groups[3])
    PRM4pv = PopulationRateMonitor(PENv_groups[4])
    PRM5pv = PopulationRateMonitor(PENv_groups[5])
    PRM6pv = PopulationRateMonitor(PENv_groups[6])
    PRM7pv = PopulationRateMonitor(PENv_groups[7])
    PRM8pv = PopulationRateMonitor(PENv_groups[8])
    PRM9pv = PopulationRateMonitor(PENv_groups[9])
    PRM10pv = PopulationRateMonitor(PENv_groups[10])
    PRM11pv = PopulationRateMonitor(PENv_groups[11])
    PRM12pv = PopulationRateMonitor(PENv_groups[12])
    PRM13pv = PopulationRateMonitor(PENv_groups[13])
    PRM14pv = PopulationRateMonitor(PENv_groups[14])
    PRM15pv = PopulationRateMonitor(PENv_groups[15])

    PRMR = PopulationRateMonitor(R)

    # SM = SpikeMonitor(EPG)
    # SMv = SpikeMonitor(EPGv)
    print('collect')
    net=Network(collect())
    net.add(EPG_groups,EPG_syn,PEN_groups,PEN_syn,EP_syn,PE2R_syn,PE2L_syn,PE1R_syn,PE1L_syn,PE2R_syn2,PE2L_syn2,PE1R_syn2,PE1L_syn2,PE7_syn,PE8_syn)
    net.add(EPGv_groups,EPGv_syn,PENv_groups,PENv_syn,EPv_syn,PE2Rv_syn,PE2Lv_syn,PE1Rv_syn,PE1Lv_syn,PE2Rv_syn2,PE2Lv_syn2,PE1Rv_syn2,PE1Lv_syn2,PE7v_syn,PE8v_syn)

    # run simulation

    ## SIMULATION ###
    print('visual cue')
    stimulus_location %= 2*np.pi
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi
    
    theta_r_v = stimulus_location_v/2
    theta_l_v = theta_r_v + np.pi
    
    A = stimulus_strength
    
    for i in range(0,8):
        EPG_groups[i].I = visual_cue(theta_r, i, A)
    for i in range(8,16):
        EPG_groups[i].I = visual_cue(theta_l, i, A)
    for i in range(0,16):
        EPGv_groups[i].I = visual_cue(theta_r_v, i, A)
    for i in range(8,16):
        EPGv_groups[i].I = visual_cue(theta_l_v, i, A)
    
    net.run(t_epg_open*ms)

    for i in range(0,16):
        EPG_groups[i].I = 0
    for i in range(0,16):
        EPGv_groups[i].I = 0
    net.run(t_epg_close * ms)

    print('body rotation')
    if half_PEN == 'right':
        for i in range(8): PEN_groups[i].I = shifter_strength
    elif half_PEN == 'left':
        for i in range(8,16): PEN_groups[i].I = shifter_strength
    if half_PENv == 'right':
        for i in range(8): PENv_groups[i].I = shifter_strength_v
    elif half_PENv == 'left':
        for i in range(8,16): PENv_groups[i].I = shifter_strength_v
            
    net.run(t_pen_open * ms)
    end  = time.time()
    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> eval end', flush=True)
    
    # Build and execute the generated CUDA/C++ standalone code **before** accessing the recorded variables
    # (required because we set `build_on_run=False` in `set_device` at the top of this file)
    device.build(run=True, clean=True)

    firing_rate = [PRM0.smooth_rate(width=5*ms),
                    PRM1.smooth_rate(width=5*ms),
                    PRM2.smooth_rate(width=5*ms),
                    PRM3.smooth_rate(width=5*ms),
                    PRM4.smooth_rate(width=5*ms),
                    PRM5.smooth_rate(width=5*ms),
                    PRM6.smooth_rate(width=5*ms),
                    PRM7.smooth_rate(width=5*ms),
                    PRM8.smooth_rate(width=5*ms),
                    PRM9.smooth_rate(width=5*ms),
                    PRM10.smooth_rate(width=5*ms),
                    PRM11.smooth_rate(width=5*ms),
                    PRM12.smooth_rate(width=5*ms),
                    PRM13.smooth_rate(width=5*ms),
                    PRM14.smooth_rate(width=5*ms),
                    PRM15.smooth_rate(width=5*ms),]
    
    firing_rate_v = [PRM0v.smooth_rate(width=5*ms),
                    PRM1v.smooth_rate(width=5*ms),
                    PRM2v.smooth_rate(width=5*ms),
                    PRM3v.smooth_rate(width=5*ms),
                    PRM4v.smooth_rate(width=5*ms),
                    PRM5v.smooth_rate(width=5*ms),
                    PRM6v.smooth_rate(width=5*ms),
                    PRM7v.smooth_rate(width=5*ms),
                    PRM8v.smooth_rate(width=5*ms),
                    PRM9v.smooth_rate(width=5*ms),
                    PRM10v.smooth_rate(width=5*ms),
                    PRM11v.smooth_rate(width=5*ms),
                    PRM12v.smooth_rate(width=5*ms),
                    PRM13v.smooth_rate(width=5*ms),
                    PRM14v.smooth_rate(width=5*ms),
                    PRM15v.smooth_rate(width=5*ms),]
    
    firing_rate_pen = [PRM0.smooth_rate(width=5*ms),
                       PRM1.smooth_rate(width=5*ms),
                       PRM2.smooth_rate(width=5*ms),
                       PRM3.smooth_rate(width=5*ms),
                       PRM4.smooth_rate(width=5*ms),
                       PRM5.smooth_rate(width=5*ms),
                       PRM6.smooth_rate(width=5*ms),
                       PRM7.smooth_rate(width=5*ms),
                       PRM8.smooth_rate(width=5*ms),
                       PRM9.smooth_rate(width=5*ms),
                       PRM10.smooth_rate(width=5*ms),
                       PRM11.smooth_rate(width=5*ms),
                       PRM12.smooth_rate(width=5*ms),
                       PRM13.smooth_rate(width=5*ms),
                       PRM14.smooth_rate(width=5*ms),
                       PRM15.smooth_rate(width=5*ms),]
    
    firing_rate_pen_v = [PRM0pv.smooth_rate(width=5*ms),
                       PRM1pv.smooth_rate(width=5*ms),
                       PRM2pv.smooth_rate(width=5*ms),
                       PRM3pv.smooth_rate(width=5*ms),
                       PRM4pv.smooth_rate(width=5*ms),
                       PRM5pv.smooth_rate(width=5*ms),
                       PRM6pv.smooth_rate(width=5*ms),
                       PRM7pv.smooth_rate(width=5*ms),  
                       PRM8pv.smooth_rate(width=5*ms),
                       PRM9pv.smooth_rate(width=5*ms),
                       PRM10pv.smooth_rate(width=5*ms),
                       PRM11pv.smooth_rate(width=5*ms),
                       PRM12pv.smooth_rate(width=5*ms),
                       PRM13pv.smooth_rate(width=5*ms),
                       PRM14pv.smooth_rate(width=5*ms),
                       PRM15pv.smooth_rate(width=5*ms),]
    
    firing_rate_r = [PRMR.smooth_rate(width=5*ms),]
    
    firing_rate_pen_array = np.array(firing_rate_pen)
    firing_rate_pen_array_v = np.array(firing_rate_pen_v)
    firing_rate_r_array = np.array(firing_rate_r)

    firing_rate_array = np.array(firing_rate)
    firing_rate_array_v = np.array(firing_rate_v)
    eval_time = np.linspace(0, len(firing_rate[0])/10000, len(firing_rate[0]))
    return eval_time, firing_rate_array, firing_rate_array_v, firing_rate_pen_array, firing_rate_pen_array_v, firing_rate_r_array

if __name__ == '__main__':
    eval_time, firing_rate, firing_rate_v, firing_rate_pen, firing_rate_pen_v, firing_rate_r = simulator()    