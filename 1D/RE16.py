from brian2 import *
from equations import *
import matplotlib.pyplot as plt
import numpy as np
import time

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
        shifter_strength = 0.015,
        half_PEN = 'right',
        
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
    E_GABAA     = -0.07 # GABAA reversal potential
    tau_GABAA   = 5*ms # GABAA synaptic time constant



    # create neuron
    EPG = NeuronGroup(48, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler' )
    PEN = NeuronGroup(48,model=eqs_PEN, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron1
    EPG.v = E_l
    PEN.v = E_l
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

    R_groups = []
    R_groups.append(R[0:3]) # R1

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
    
    # EPG_EPG
    for k in range(0,16):
        # EPG to EPG
        EPG_syn.append(Synapses(EPG_groups[k], EPG_groups[k], Ach_eqs, on_pre='s_ach += w_EE', method='euler'))
        EPG_syn[k].connect(condition='i != j')
    
    # PEN_PEN
    for k2 in range(0,16):
        # PEN to PEN
        PEN_syn.append(Synapses(PEN_groups[k2], PEN_groups[k2], Ach_eqs_PP, on_pre='s_ach += w_PP', method='euler'))
        PEN_syn[k2].connect(condition='i != j')
    
    # EPG to R
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    for a in range(0,48):
        for b in range(0,3):
            S_EI.connect(i=a, j=b)
    
    # R to EPG
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    for a2 in range(0,48):
        for b2 in range(0,3):
            S_IE.connect(i=b2, j=a2)
    
    # R_R with symmetric connections
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # EPG_PEN synapse
    for k3 in range(0,16):
        # EPG to PEN
        EP_syn.append(Synapses(EPG_groups[k3], PEN_groups[k3], Ach_eqs_EP, on_pre='s_ach += w_EP', method='euler'))
        EP_syn[k3].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    ###
    
    # PEN_EPG synapse #v
    # PEN0-6 -> EPG0-8
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+1], Ach_eqs_PE2R, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+2], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_syn.append(Synapses(PEN_groups[6], EPG_groups[0], Ach_eqs_PE1R, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG0-8 
    for k4 in range(0,7):
        # PEN to EPG
        PE2R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+1], Ach_eqs_PE2R2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1R_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+2], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1R_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1R_syn2.append(Synapses(PEN_groups[15], EPG_groups[0], Ach_eqs_PE1R2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1R_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN7-8
    PE7_syn = []
    # PEN7 -> EPG connections
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[0], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[1], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[15], Ach_eqs_PE7, on_pre='s_ach += 2*w_PE', method='euler'))
    PE7_syn.append(Synapses(PEN_groups[7], EPG_groups[14], Ach_eqs_PE7, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE7_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE8_syn = []
    # PEN8 -> EPG connections
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[0], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[1], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[15], Ach_eqs_PE8, on_pre='s_ach += 2*w_PE', method='euler'))
    PE8_syn.append(Synapses(PEN_groups[8], EPG_groups[14], Ach_eqs_PE8, on_pre='s_ach += 1*w_PE', method='euler'))
    for k in range(0,4):
        PE8_syn[k].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN0-6 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn.append(Synapses(PEN_groups[k4], EPG_groups[k4+8], Ach_eqs_PE2L, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn.append(Synapses(PEN_groups[k4+1], EPG_groups[k4+8], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
        
    PE1L_syn.append(Synapses(PEN_groups[0], EPG_groups[15], Ach_eqs_PE1L, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    # PEN9-15 -> EPG8-15
    for k4 in range(0,7):
        # PEN to EPG
        PE2L_syn2.append(Synapses(PEN_groups[k4+9], EPG_groups[k4+8], Ach_eqs_PE2L2, on_pre='s_ach += 2*w_PE', method='euler'))
        PE2L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    for k4 in range(0,6):
        # PEN to EPG
        PE1L_syn2.append(Synapses(PEN_groups[k4+10], EPG_groups[k4+8], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
        PE1L_syn2[k4].connect(j='u for u in range(0,3)', skip_if_invalid=True)
    
    PE1L_syn2.append(Synapses(PEN_groups[9], EPG_groups[15], Ach_eqs_PE1L2, on_pre='s_ach += 1*w_PE', method='euler'))
    PE1L_syn2[6].connect(j='u for u in range(0,3)', skip_if_invalid=True)

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

    PRM_PEN0 = PopulationRateMonitor(PEN_groups[0])
    PRM_PEN1 = PopulationRateMonitor(PEN_groups[1])
    PRM_PEN2 = PopulationRateMonitor(PEN_groups[2])
    PRM_PEN3 = PopulationRateMonitor(PEN_groups[3])
    PRM_PEN4 = PopulationRateMonitor(PEN_groups[4])
    PRM_PEN5 = PopulationRateMonitor(PEN_groups[5])
    PRM_PEN6 = PopulationRateMonitor(PEN_groups[6])
    PRM_PEN7 = PopulationRateMonitor(PEN_groups[7])
    PRM_PEN8 = PopulationRateMonitor(PEN_groups[8])
    PRM_PEN9 = PopulationRateMonitor(PEN_groups[9])
    PRM_PEN10 = PopulationRateMonitor(PEN_groups[10])
    PRM_PEN11 = PopulationRateMonitor(PEN_groups[11])
    PRM_PEN12 = PopulationRateMonitor(PEN_groups[12])
    PRM_PEN13 = PopulationRateMonitor(PEN_groups[13])
    PRM_PEN14 = PopulationRateMonitor(PEN_groups[14])
    PRM_PEN15 = PopulationRateMonitor(PEN_groups[15])
    
    PRM_R0 = PopulationRateMonitor(R_groups[0])


    SM = SpikeMonitor(EPG)
    
    net=Network(collect())
    net.add(EPG_groups,EPG_syn,PEN_groups,PEN_syn,EP_syn,PE2R_syn,PE2L_syn,PE1R_syn,PE1L_syn,PE2R_syn2,PE2L_syn2,PE1R_syn2,PE1L_syn2,PE7_syn,PE8_syn)

    # run simulation

    ## SIMULATION ###

    stimulus_location %= 2*np.pi
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi
    A = stimulus_strength
    
    for i in range(0,8):
        EPG_groups[i].I = visual_cue(theta_r, i, A)
    for i in range(8,16):
        EPG_groups[i].I = visual_cue(theta_l, i, A)
    
    net.run(t_epg_open*ms)

    for i in range(0,16):
        EPG_groups[i].I = 0
    net.run(t_epg_close * ms)

    if half_PEN == 'right':
        for i in range(8):
            PEN_groups[i].I = shifter_strength
    elif half_PEN == 'left':
        for i in range(8,16):
            PEN_groups[i].I = shifter_strength

    net.run(t_pen_open * ms)
    
    last = len(SM.t)
    end  = time.time()
    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> eval end', flush=True)
    
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
    firing_rate_array = np.array(firing_rate)
    
    
    firing_rate_pen = [PRM_PEN0.smooth_rate(width=5*ms),
                       PRM_PEN1.smooth_rate(width=5*ms),
                       PRM_PEN2.smooth_rate(width=5*ms),
                       PRM_PEN3.smooth_rate(width=5*ms),
                       PRM_PEN4.smooth_rate(width=5*ms),
                       PRM_PEN5.smooth_rate(width=5*ms),
                       PRM_PEN6.smooth_rate(width=5*ms),
                       PRM_PEN7.smooth_rate(width=5*ms),
                       PRM_PEN8.smooth_rate(width=5*ms),
                       PRM_PEN9.smooth_rate(width=5*ms),
                       PRM_PEN10.smooth_rate(width=5*ms),
                       PRM_PEN11.smooth_rate(width=5*ms),
                       PRM_PEN12.smooth_rate(width=5*ms),
                       PRM_PEN13.smooth_rate(width=5*ms),
                       PRM_PEN14.smooth_rate(width=5*ms),
                       PRM_PEN15.smooth_rate(width=5*ms),]
    firing_rate_pen_array = np.array(firing_rate_pen)
    
    firing_rate_r = [PRM_R0.smooth_rate(width=5*ms),]
    firing_rate_r_array = np.array(firing_rate_r)
    
    
    
    eval_time = np.linspace(0, len(firing_rate[0])/10000, len(firing_rate[0]))
    return eval_time, firing_rate_array, firing_rate_pen_array, firing_rate_r_array

if __name__ == '__main__':
    t, fr, fr_pen, fr_r = simulator()    