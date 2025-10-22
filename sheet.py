from brian2 import *
from sheet_eqs import *
from sheet_con import build_pen_to_epg_array
import numpy as np
import time
from tqdm import tqdm, trange

def visual_cue(theta, index, stimulus = 0.03, sigma = 0.8 * np.pi/8):
    """
    param: 
    theta: the angle of the visual input
    index: the index of the neuron
    stimulus: the strength of the visual input
    sigma: the standard deviation of the Gaussian distribution
    """
    A = stimulus
    phi = (index * np.pi/8) % (2*np.pi)
    d1 = (theta-phi)**2 
    d2 = (theta-phi + 2*np.pi)**2
    d3 = (theta-phi - 2*np.pi)**2
    return A * (np.exp(-d1/(2*sigma**2)) + np.exp(-d2/(2*sigma**2)) + np.exp(-d3/(2*sigma**2)))

def visual_cue_2D(x, y, i, j, stimulus = 0.03, sigma = 0.8 * np.pi/8):

    A = stimulus
    r1 = np.sqrt((x-i)**2 + (y-j)**2) 

    return A * (np.exp(-r1**2/(2*sigma**2)))



def map_index(i):
    g = i // 3  # subgroup index
    o = i % 3   # offset in subgroup
    return g * 48 + o


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
        
        stimulus_strength = np.array([0.05, 0, 0, 0]), 
        stimulus_location =0 , # from 0 to np.pi
        shifter_strength = 0.015,
        half_PEN = 'right',

        t_epg_open = 200, # stimulus
        t_epg_close = 500,    # no stimulus
        t_pen_open = 5000,   # shift

        # debug
        debug = False,
        target = 0,
        tg = 0
):
    """Simulate the head direction network with visual cues and body rotation."""

    if debug:
        print(f'{time.strftime("%H:%M:%S")} [info] Parameters:')
        print(f'w_EE: {w_EE}')
        print(f'w_EI: {w_EI}')
        print(f'w_IE: {w_IE}')
        print(f'w_II: {w_II}')
        print(f'w_PP: {w_PP}')
        print(f'w_EP: {w_EP}')
        print(f'w_PE: {w_PE}')
        print(f'sigma: {sigma}')
        print(f'stimulus_strength: {stimulus_strength}')
        print(f'stimulus_location: {stimulus_location}')
        print(f'shifter_strength: {shifter_strength}')

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
    target = target
        

    EPG = NeuronGroup(16*16*3, model=eqs_EPG, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENx = NeuronGroup(16*16*3,model=eqs_PENx, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    PENy = NeuronGroup(16*16*3,model=eqs_PENy, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')
    R = NeuronGroup(3,model=eqs_R, threshold='v>Vth', reset='v=Vr', refractory='1*ms', method='euler')

    # initialize neuron
    EPG.v = E_l
    PENx.v = E_l
    PENy.v = E_l
    R.v = E_l

    EPG_groups = [EPG[i:i+3] for i in range(0, 16*16*3, 3)]
    PENx_groups = [PENx[i:i+3] for i in range(0, 16*16*3, 3)]
    PENy_groups = [PENy[i:i+3] for i in range(0, 16*16*3, 3)]
    R_groups = [R[0:3]]
    
    # ========= EPG -> EPG =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> EPG connections')

    S_EE = Synapses(EPG, EPG, Ach_eqs_EE, on_pre='s_ach += w_EE', method='euler')
    S_EE.connect(condition='i//3 == j//3 and i != j')

    # ========= PEN -> PEN =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building PEN -> PEN connections')
    S_PPx = Synapses(PENx, PENx, Ach_eqs_PPx, on_pre='s_ach += w_PP', method='euler')
    S_PPx.connect(condition='i//3 == j//3 and i != j')
    S_PPy = Synapses(PENy, PENy, Ach_eqs_PPy, on_pre='s_ach += w_PP', method='euler')
    S_PPy.connect(condition='i//3 == j//3 and i != j')
    
    # ========= EPG -> R =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> R connections')
    S_EI = Synapses(EPG, R, model=Ach_eqs_EI, on_pre='s_ach += w_EI', method='euler')
    S_EI.connect(condition='True')
    
    # ========= R   -> EPG =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building R -> EPG connections')
    S_IE = Synapses(R, EPG, model=GABA_eqs, on_pre='s_GABAA += w_IE', method='euler')
    S_IE.connect(condition='True')
    
    # ========= R <-> R =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building R <-> R connections')
    S_II = Synapses(R, R, model=GABA_eqs_i, on_pre='s_GABAA += w_II', method='euler')
    S_II.connect(condition='i != j')
    
    # ========= EPG -> PEN =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building EPG -> PEN connections')
    S_EPx = Synapses(EPG, PENx, Ach_eqs_EPx, on_pre='s_ach += w_EP', method='euler')
    S_EPx.connect(condition='i//3 == j//3')
    S_EPy = Synapses(EPG, PENy, Ach_eqs_EPy, on_pre='s_ach += w_EP', method='euler')
    S_EPy.connect(condition='i//3 == j//3')

    # ========= PEN -> EPG (optimized by connectivity matrix) =========
    # print(f'{time.strftime("%H:%M:%S")} [info] Building PEN -> EPG connections')
    S_PxE_2 = Synapses(PENx, EPG, model=Ach_eqs_PxE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PxE_1 = Synapses(PENx, EPG, model=Ach_eqs_PxE_1, on_pre='s_ach += 1*w_PE', method='euler')
    S_PyE_2 = Synapses(PENy, EPG, model=Ach_eqs_PyE_2, on_pre='s_ach += 2*w_PE', method='euler')
    S_PyE_1 = Synapses(PENy, EPG, model=Ach_eqs_PyE_1, on_pre='s_ach += 1*w_PE', method='euler')

    pre2, post2, pre1, post1 = build_pen_to_epg_array()

    # # ---- build all connections at once ----
    for k in range(16):
        S_PxE_2.connect(i=pre2+k*48, j=post2+k*48)
        S_PxE_1.connect(i=pre1+k*48, j=post1+k*48)
        S_PyE_2.connect(i=map_index(pre2)+k*3, j=map_index(post2)+k*3)
        S_PyE_1.connect(i=map_index(pre1)+k*3, j=map_index(post1)+k*3)

    # ========= end PEN -> EPG =========

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
    PRM16 = PopulationRateMonitor(EPG_groups[16])
    PRM17 = PopulationRateMonitor(EPG_groups[17])
    PRM18 = PopulationRateMonitor(EPG_groups[18])
    PRM19 = PopulationRateMonitor(EPG_groups[19])
    PRM20 = PopulationRateMonitor(EPG_groups[20])
    PRM21 = PopulationRateMonitor(EPG_groups[21])
    PRM22 = PopulationRateMonitor(EPG_groups[22])
    PRM23 = PopulationRateMonitor(EPG_groups[23])
    PRM24 = PopulationRateMonitor(EPG_groups[24])
    PRM25 = PopulationRateMonitor(EPG_groups[25])
    PRM26 = PopulationRateMonitor(EPG_groups[26])
    PRM27 = PopulationRateMonitor(EPG_groups[27])
    PRM28 = PopulationRateMonitor(EPG_groups[28])
    PRM29 = PopulationRateMonitor(EPG_groups[29])
    PRM30 = PopulationRateMonitor(EPG_groups[30])
    PRM31 = PopulationRateMonitor(EPG_groups[31])
    PRM32 = PopulationRateMonitor(EPG_groups[32])
    PRM33 = PopulationRateMonitor(EPG_groups[33])
    PRM34 = PopulationRateMonitor(EPG_groups[34])
    PRM35 = PopulationRateMonitor(EPG_groups[35])
    PRM36 = PopulationRateMonitor(EPG_groups[36])
    PRM37 = PopulationRateMonitor(EPG_groups[37])
    PRM38 = PopulationRateMonitor(EPG_groups[38])
    PRM39 = PopulationRateMonitor(EPG_groups[39])
    PRM40 = PopulationRateMonitor(EPG_groups[40])
    PRM41 = PopulationRateMonitor(EPG_groups[41])
    PRM42 = PopulationRateMonitor(EPG_groups[42])
    PRM43 = PopulationRateMonitor(EPG_groups[43])
    PRM44 = PopulationRateMonitor(EPG_groups[44])
    PRM45 = PopulationRateMonitor(EPG_groups[45])
    PRM46 = PopulationRateMonitor(EPG_groups[46])
    PRM47 = PopulationRateMonitor(EPG_groups[47])
    PRM48 = PopulationRateMonitor(EPG_groups[48])
    PRM49 = PopulationRateMonitor(EPG_groups[49])
    PRM50 = PopulationRateMonitor(EPG_groups[50])
    PRM51 = PopulationRateMonitor(EPG_groups[51])
    PRM52 = PopulationRateMonitor(EPG_groups[52])
    PRM53 = PopulationRateMonitor(EPG_groups[53])
    PRM54 = PopulationRateMonitor(EPG_groups[54])
    PRM55 = PopulationRateMonitor(EPG_groups[55])
    PRM56 = PopulationRateMonitor(EPG_groups[56])
    PRM57 = PopulationRateMonitor(EPG_groups[57])
    PRM58 = PopulationRateMonitor(EPG_groups[58])
    PRM59 = PopulationRateMonitor(EPG_groups[59])
    PRM60 = PopulationRateMonitor(EPG_groups[60])
    PRM61 = PopulationRateMonitor(EPG_groups[61])
    PRM62 = PopulationRateMonitor(EPG_groups[62])
    PRM63 = PopulationRateMonitor(EPG_groups[63])
    PRM64 = PopulationRateMonitor(EPG_groups[64])
    PRM65 = PopulationRateMonitor(EPG_groups[65])
    PRM66 = PopulationRateMonitor(EPG_groups[66])
    PRM67 = PopulationRateMonitor(EPG_groups[67])
    PRM68 = PopulationRateMonitor(EPG_groups[68])
    PRM69 = PopulationRateMonitor(EPG_groups[69])
    PRM70 = PopulationRateMonitor(EPG_groups[70])
    PRM71 = PopulationRateMonitor(EPG_groups[71])
    PRM72 = PopulationRateMonitor(EPG_groups[72])
    PRM73 = PopulationRateMonitor(EPG_groups[73])
    PRM74 = PopulationRateMonitor(EPG_groups[74])
    PRM75 = PopulationRateMonitor(EPG_groups[75])
    PRM76 = PopulationRateMonitor(EPG_groups[76])
    PRM77 = PopulationRateMonitor(EPG_groups[77])
    PRM78 = PopulationRateMonitor(EPG_groups[78])
    PRM79 = PopulationRateMonitor(EPG_groups[79])
    PRM80 = PopulationRateMonitor(EPG_groups[80])
    PRM81 = PopulationRateMonitor(EPG_groups[81])
    PRM82 = PopulationRateMonitor(EPG_groups[82])
    PRM83 = PopulationRateMonitor(EPG_groups[83])
    PRM84 = PopulationRateMonitor(EPG_groups[84])
    PRM85 = PopulationRateMonitor(EPG_groups[85])
    PRM86 = PopulationRateMonitor(EPG_groups[86])
    PRM87 = PopulationRateMonitor(EPG_groups[87])
    PRM88 = PopulationRateMonitor(EPG_groups[88])
    PRM89 = PopulationRateMonitor(EPG_groups[89])
    PRM90 = PopulationRateMonitor(EPG_groups[90])
    PRM91 = PopulationRateMonitor(EPG_groups[91])
    PRM92 = PopulationRateMonitor(EPG_groups[92])
    PRM93 = PopulationRateMonitor(EPG_groups[93])
    PRM94 = PopulationRateMonitor(EPG_groups[94])
    PRM95 = PopulationRateMonitor(EPG_groups[95])
    PRM96 = PopulationRateMonitor(EPG_groups[96])
    PRM97 = PopulationRateMonitor(EPG_groups[97])
    PRM98 = PopulationRateMonitor(EPG_groups[98])
    PRM99 = PopulationRateMonitor(EPG_groups[99])
    PRM100 = PopulationRateMonitor(EPG_groups[100])
    PRM101 = PopulationRateMonitor(EPG_groups[101])
    PRM102 = PopulationRateMonitor(EPG_groups[102])
    PRM103 = PopulationRateMonitor(EPG_groups[103])
    PRM104 = PopulationRateMonitor(EPG_groups[104])
    PRM105 = PopulationRateMonitor(EPG_groups[105])
    PRM106 = PopulationRateMonitor(EPG_groups[106])
    PRM107 = PopulationRateMonitor(EPG_groups[107])
    PRM108 = PopulationRateMonitor(EPG_groups[108])
    PRM109 = PopulationRateMonitor(EPG_groups[109])
    PRM110 = PopulationRateMonitor(EPG_groups[110])
    PRM111 = PopulationRateMonitor(EPG_groups[111])
    PRM112 = PopulationRateMonitor(EPG_groups[112])
    PRM113 = PopulationRateMonitor(EPG_groups[113])
    PRM114 = PopulationRateMonitor(EPG_groups[114])
    PRM115 = PopulationRateMonitor(EPG_groups[115])
    PRM116 = PopulationRateMonitor(EPG_groups[116])
    PRM117 = PopulationRateMonitor(EPG_groups[117])
    PRM118 = PopulationRateMonitor(EPG_groups[118])
    PRM119 = PopulationRateMonitor(EPG_groups[119])
    PRM120 = PopulationRateMonitor(EPG_groups[120])
    PRM121 = PopulationRateMonitor(EPG_groups[121])
    PRM122 = PopulationRateMonitor(EPG_groups[122])
    PRM123 = PopulationRateMonitor(EPG_groups[123])
    PRM124 = PopulationRateMonitor(EPG_groups[124])
    PRM125 = PopulationRateMonitor(EPG_groups[125])
    PRM126 = PopulationRateMonitor(EPG_groups[126])
    PRM127 = PopulationRateMonitor(EPG_groups[127])
    PRM128 = PopulationRateMonitor(EPG_groups[128])
    PRM129 = PopulationRateMonitor(EPG_groups[129])
    PRM130 = PopulationRateMonitor(EPG_groups[130])
    PRM131 = PopulationRateMonitor(EPG_groups[131])
    PRM132 = PopulationRateMonitor(EPG_groups[132])
    PRM133 = PopulationRateMonitor(EPG_groups[133])
    PRM134 = PopulationRateMonitor(EPG_groups[134])
    PRM135 = PopulationRateMonitor(EPG_groups[135])
    PRM136 = PopulationRateMonitor(EPG_groups[136])
    PRM137 = PopulationRateMonitor(EPG_groups[137])
    PRM138 = PopulationRateMonitor(EPG_groups[138])
    PRM139 = PopulationRateMonitor(EPG_groups[139])
    PRM140 = PopulationRateMonitor(EPG_groups[140])
    PRM141 = PopulationRateMonitor(EPG_groups[141])
    PRM142 = PopulationRateMonitor(EPG_groups[142])
    PRM143 = PopulationRateMonitor(EPG_groups[143])
    PRM144 = PopulationRateMonitor(EPG_groups[144])
    PRM145 = PopulationRateMonitor(EPG_groups[145])
    PRM146 = PopulationRateMonitor(EPG_groups[146])
    PRM147 = PopulationRateMonitor(EPG_groups[147])
    PRM148 = PopulationRateMonitor(EPG_groups[148])
    PRM149 = PopulationRateMonitor(EPG_groups[149])
    PRM150 = PopulationRateMonitor(EPG_groups[150])
    PRM151 = PopulationRateMonitor(EPG_groups[151])
    PRM152 = PopulationRateMonitor(EPG_groups[152])
    PRM153 = PopulationRateMonitor(EPG_groups[153])
    PRM154 = PopulationRateMonitor(EPG_groups[154])
    PRM155 = PopulationRateMonitor(EPG_groups[155])
    PRM156 = PopulationRateMonitor(EPG_groups[156])
    PRM157 = PopulationRateMonitor(EPG_groups[157])
    PRM158 = PopulationRateMonitor(EPG_groups[158])
    PRM159 = PopulationRateMonitor(EPG_groups[159])
    PRM160 = PopulationRateMonitor(EPG_groups[160])
    PRM161 = PopulationRateMonitor(EPG_groups[161])
    PRM162 = PopulationRateMonitor(EPG_groups[162])
    PRM163 = PopulationRateMonitor(EPG_groups[163])
    PRM164 = PopulationRateMonitor(EPG_groups[164])
    PRM165 = PopulationRateMonitor(EPG_groups[165])
    PRM166 = PopulationRateMonitor(EPG_groups[166])
    PRM167 = PopulationRateMonitor(EPG_groups[167])
    PRM168 = PopulationRateMonitor(EPG_groups[168])
    PRM169 = PopulationRateMonitor(EPG_groups[169])
    PRM170 = PopulationRateMonitor(EPG_groups[170])
    PRM171 = PopulationRateMonitor(EPG_groups[171])
    PRM172 = PopulationRateMonitor(EPG_groups[172])
    PRM173 = PopulationRateMonitor(EPG_groups[173])
    PRM174 = PopulationRateMonitor(EPG_groups[174])
    PRM175 = PopulationRateMonitor(EPG_groups[175])
    PRM176 = PopulationRateMonitor(EPG_groups[176])
    PRM177 = PopulationRateMonitor(EPG_groups[177])
    PRM178 = PopulationRateMonitor(EPG_groups[178])
    PRM179 = PopulationRateMonitor(EPG_groups[179])
    PRM180 = PopulationRateMonitor(EPG_groups[180])
    PRM181 = PopulationRateMonitor(EPG_groups[181])
    PRM182 = PopulationRateMonitor(EPG_groups[182])
    PRM183 = PopulationRateMonitor(EPG_groups[183])
    PRM184 = PopulationRateMonitor(EPG_groups[184])
    PRM185 = PopulationRateMonitor(EPG_groups[185])
    PRM186 = PopulationRateMonitor(EPG_groups[186])
    PRM187 = PopulationRateMonitor(EPG_groups[187])
    PRM188 = PopulationRateMonitor(EPG_groups[188])
    PRM189 = PopulationRateMonitor(EPG_groups[189])
    PRM190 = PopulationRateMonitor(EPG_groups[190])
    PRM191 = PopulationRateMonitor(EPG_groups[191])
    PRM192 = PopulationRateMonitor(EPG_groups[192])
    PRM193 = PopulationRateMonitor(EPG_groups[193])
    PRM194 = PopulationRateMonitor(EPG_groups[194])
    PRM195 = PopulationRateMonitor(EPG_groups[195])
    PRM196 = PopulationRateMonitor(EPG_groups[196])
    PRM197 = PopulationRateMonitor(EPG_groups[197])
    PRM198 = PopulationRateMonitor(EPG_groups[198])
    PRM199 = PopulationRateMonitor(EPG_groups[199])
    PRM200 = PopulationRateMonitor(EPG_groups[200])
    PRM201 = PopulationRateMonitor(EPG_groups[201])
    PRM202 = PopulationRateMonitor(EPG_groups[202])
    PRM203 = PopulationRateMonitor(EPG_groups[203])
    PRM204 = PopulationRateMonitor(EPG_groups[204])
    PRM205 = PopulationRateMonitor(EPG_groups[205])
    PRM206 = PopulationRateMonitor(EPG_groups[206])
    PRM207 = PopulationRateMonitor(EPG_groups[207])
    PRM208 = PopulationRateMonitor(EPG_groups[208])
    PRM209 = PopulationRateMonitor(EPG_groups[209])
    PRM210 = PopulationRateMonitor(EPG_groups[210])
    PRM211 = PopulationRateMonitor(EPG_groups[211])
    PRM212 = PopulationRateMonitor(EPG_groups[212])
    PRM213 = PopulationRateMonitor(EPG_groups[213])
    PRM214 = PopulationRateMonitor(EPG_groups[214])
    PRM215 = PopulationRateMonitor(EPG_groups[215])
    PRM216 = PopulationRateMonitor(EPG_groups[216])
    PRM217 = PopulationRateMonitor(EPG_groups[217])
    PRM218 = PopulationRateMonitor(EPG_groups[218])
    PRM219 = PopulationRateMonitor(EPG_groups[219])
    PRM220 = PopulationRateMonitor(EPG_groups[220])
    PRM221 = PopulationRateMonitor(EPG_groups[221])
    PRM222 = PopulationRateMonitor(EPG_groups[222])
    PRM223 = PopulationRateMonitor(EPG_groups[223])
    PRM224 = PopulationRateMonitor(EPG_groups[224])
    PRM225 = PopulationRateMonitor(EPG_groups[225])
    PRM226 = PopulationRateMonitor(EPG_groups[226])
    PRM227 = PopulationRateMonitor(EPG_groups[227])
    PRM228 = PopulationRateMonitor(EPG_groups[228])
    PRM229 = PopulationRateMonitor(EPG_groups[229])
    PRM230 = PopulationRateMonitor(EPG_groups[230])
    PRM231 = PopulationRateMonitor(EPG_groups[231])
    PRM232 = PopulationRateMonitor(EPG_groups[232])
    PRM233 = PopulationRateMonitor(EPG_groups[233])
    PRM234 = PopulationRateMonitor(EPG_groups[234])
    PRM235 = PopulationRateMonitor(EPG_groups[235])
    PRM236 = PopulationRateMonitor(EPG_groups[236])
    PRM237 = PopulationRateMonitor(EPG_groups[237])
    PRM238 = PopulationRateMonitor(EPG_groups[238])
    PRM239 = PopulationRateMonitor(EPG_groups[239])
    PRM240 = PopulationRateMonitor(EPG_groups[240])
    PRM241 = PopulationRateMonitor(EPG_groups[241])
    PRM242 = PopulationRateMonitor(EPG_groups[242])
    PRM243 = PopulationRateMonitor(EPG_groups[243])
    PRM244 = PopulationRateMonitor(EPG_groups[244])
    PRM245 = PopulationRateMonitor(EPG_groups[245])
    PRM246 = PopulationRateMonitor(EPG_groups[246])
    PRM247 = PopulationRateMonitor(EPG_groups[247])
    PRM248 = PopulationRateMonitor(EPG_groups[248])
    PRM249 = PopulationRateMonitor(EPG_groups[249])
    PRM250 = PopulationRateMonitor(EPG_groups[250])
    PRM251 = PopulationRateMonitor(EPG_groups[251])
    PRM252 = PopulationRateMonitor(EPG_groups[252])
    PRM253 = PopulationRateMonitor(EPG_groups[253])
    PRM254 = PopulationRateMonitor(EPG_groups[254])
    PRM255 = PopulationRateMonitor(EPG_groups[255])




    PRM_PEN0 = PopulationRateMonitor(PENy_groups[0])
    PRM_PEN1 = PopulationRateMonitor(PENy_groups[1])
    PRM_PEN2 = PopulationRateMonitor(PENy_groups[2])
    PRM_PEN3 = PopulationRateMonitor(PENy_groups[3])
    PRM_PEN4 = PopulationRateMonitor(PENy_groups[4])
    PRM_PEN5 = PopulationRateMonitor(PENy_groups[5])
    PRM_PEN6 = PopulationRateMonitor(PENy_groups[6])
    PRM_PEN7 = PopulationRateMonitor(PENy_groups[7])
    PRM_PEN8 = PopulationRateMonitor(PENy_groups[8])
    PRM_PEN9 = PopulationRateMonitor(PENy_groups[9])
    PRM_PEN10 = PopulationRateMonitor(PENy_groups[10])
    PRM_PEN11 = PopulationRateMonitor(PENy_groups[11])
    PRM_PEN12 = PopulationRateMonitor(PENy_groups[12])
    PRM_PEN13 = PopulationRateMonitor(PENy_groups[13])
    PRM_PEN14 = PopulationRateMonitor(PENy_groups[14])
    PRM_PEN15 = PopulationRateMonitor(PENy_groups[15])
    
    PRM_R0 = PopulationRateMonitor(R_groups[0])


    net = Network(collect())

    # run simulation

    ## SIMULATION ###
    print(f'\r{time.strftime("%H:%M:%S")} : {(time.time() - start)//60:.0f} min {(time.time() - start)%60:.1f} sec -> simulation start', flush=True)

    # DTheta / Dt = omega
    # DTheta = omega * Dt
    # Dx range: 0 to 2*pi
    w = 10
    Dx = w * 0.1
    Dy = 1
    Dt = 100 # ms
    A = stimulus_strength
    theta_r = stimulus_location/2
    theta_l = theta_r + np.pi

        

    for i in range(target*16,target*16+8):
        EPG_groups[i%256].I = visual_cue(theta_r, i, 0.05)
        EPG_groups[(i+8)%256].I = visual_cue(theta_l, i+8, 0.05)
        EPG_groups[(i+128)%256].I = visual_cue(theta_r, i+128, 0.05)
        EPG_groups[(i+136)%256].I = visual_cue(theta_l, i+136, 0.05)
    net.run(300 * ms)

    for i in range(256):
        EPG_groups[i].I = 0
    net.run(300 * ms)

    
    def reset():
        for i in range(256):
            PENx_groups[i].I = 0
            PENy_groups[i].I = 0

    def right(strength):
        for i in range(8): 
            for j in range(16):
                PENx_groups[i+j*16].I = strength
    def left(strength):
        for i in range(8,16): 
            for j in range(16):
                PENx_groups[i+j*16].I = strength
    def up(strength):
        for i in range(8):
            for j in range(16):
                PENy_groups[i*16+j].I = strength
    def down(strength):
        for i in range(8,16):
            for j in range(16):
                PENy_groups[i*16+j].I = strength


    # for i in trange(16):
    #     for i in range(8): 
    #         for j in range(16):
    #             PENx_groups[i+j*16].I = shifter_strength
    #     net.run(300 * ms)
    #     for i in range(256):
    #         PENx_groups[i].I = 0
    #         PENy_groups[i].I = 0
    #     for i in range(8):
    #         for j in range(16):
    #             PENy_groups[i*16+j].I = shifter_strength
    #     net.run(300 * ms)
    #     for i in range(256):
    #         PENx_groups[i].I = 0
    #         PENy_groups[i].I = 0


    right(shifter_strength)
    net.run(1000 * ms)
    reset()
    net.run(100*ms)
    up(shifter_strength)
    net.run(1000 * ms)
    reset()
    net.run(100*ms)
    left(shifter_strength)
    net.run(1000 * ms)
    reset()
    net.run(100*ms)
    down(shifter_strength)
    net.run(1000 * ms)
    reset()

    # if half_PEN == 'right': 
    #     for i in range(8): 
    #         for j in range(16):
    #             PENx_groups[i+j*16].I = shifter_strength
    # elif half_PEN == 'left': 
    #     for i in range(8,16): 
    #         for j in range(16):
    #             PENx_groups[i+j*16].I = shifter_strength
    # elif half_PEN == 'up':
    #     for i in range(8):
    #         for j in range(16):
    #             PENy_groups[i*16+j].I = shifter_strength
    # elif half_PEN == 'down':
    #     for i in range(8,16):
    #         for j in range(16):
    #             PENy_groups[i*16+j].I = shifter_strength
    # else: raise ValueError(f"half_PEN {half_PEN} must be 'right' or 'left' or 'up' or 'down'")
    # net.run(1000 * ms)

    end  = time.time()

    print(f'\r{time.strftime("%H:%M:%S")} : {(end - start)//60:.0f} min {(end - start)%60:.1f} sec -> simulation end', flush=True)

    fr_epg = [PRM0.smooth_rate(width=5*ms),
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
                PRM15.smooth_rate(width=5*ms),
                PRM16.smooth_rate(width=5*ms),
                PRM17.smooth_rate(width=5*ms),
                PRM18.smooth_rate(width=5*ms),
                PRM19.smooth_rate(width=5*ms),
                PRM20.smooth_rate(width=5*ms),
                PRM21.smooth_rate(width=5*ms),
                PRM22.smooth_rate(width=5*ms),
                PRM23.smooth_rate(width=5*ms),
                PRM24.smooth_rate(width=5*ms),
                PRM25.smooth_rate(width=5*ms),
                PRM26.smooth_rate(width=5*ms),
                PRM27.smooth_rate(width=5*ms),
                PRM28.smooth_rate(width=5*ms),
                PRM29.smooth_rate(width=5*ms),
                PRM30.smooth_rate(width=5*ms),
                PRM31.smooth_rate(width=5*ms),
                PRM32.smooth_rate(width=5*ms),
                PRM33.smooth_rate(width=5*ms),
                PRM34.smooth_rate(width=5*ms),
                PRM35.smooth_rate(width=5*ms),
                PRM36.smooth_rate(width=5*ms),
                PRM37.smooth_rate(width=5*ms),
                PRM38.smooth_rate(width=5*ms),
                PRM39.smooth_rate(width=5*ms),
                PRM40.smooth_rate(width=5*ms),
                PRM41.smooth_rate(width=5*ms),
                PRM42.smooth_rate(width=5*ms),
                PRM43.smooth_rate(width=5*ms),
                PRM44.smooth_rate(width=5*ms),
                PRM45.smooth_rate(width=5*ms),
                PRM46.smooth_rate(width=5*ms),
                PRM47.smooth_rate(width=5*ms),
                PRM48.smooth_rate(width=5*ms),
                PRM49.smooth_rate(width=5*ms),
                PRM50.smooth_rate(width=5*ms),
                PRM51.smooth_rate(width=5*ms),
                PRM52.smooth_rate(width=5*ms),
                PRM53.smooth_rate(width=5*ms),
                PRM54.smooth_rate(width=5*ms),
                PRM55.smooth_rate(width=5*ms),
                PRM56.smooth_rate(width=5*ms),
                PRM57.smooth_rate(width=5*ms),
                PRM58.smooth_rate(width=5*ms),
                PRM59.smooth_rate(width=5*ms),
                PRM60.smooth_rate(width=5*ms),
                PRM61.smooth_rate(width=5*ms),
                PRM62.smooth_rate(width=5*ms),
                PRM63.smooth_rate(width=5*ms),
                PRM64.smooth_rate(width=5*ms),
                PRM65.smooth_rate(width=5*ms),
                PRM66.smooth_rate(width=5*ms),
                PRM67.smooth_rate(width=5*ms),
                PRM68.smooth_rate(width=5*ms),
                PRM69.smooth_rate(width=5*ms),
                PRM70.smooth_rate(width=5*ms),
                PRM71.smooth_rate(width=5*ms),
                PRM72.smooth_rate(width=5*ms),
                PRM73.smooth_rate(width=5*ms),
                PRM74.smooth_rate(width=5*ms),
                PRM75.smooth_rate(width=5*ms),
                PRM76.smooth_rate(width=5*ms),
                PRM77.smooth_rate(width=5*ms),
                PRM78.smooth_rate(width=5*ms),
                PRM79.smooth_rate(width=5*ms),
                PRM80.smooth_rate(width=5*ms),
                PRM81.smooth_rate(width=5*ms),
                PRM82.smooth_rate(width=5*ms),
                PRM83.smooth_rate(width=5*ms),
                PRM84.smooth_rate(width=5*ms),
                PRM85.smooth_rate(width=5*ms),
                PRM86.smooth_rate(width=5*ms),
                PRM87.smooth_rate(width=5*ms),
                PRM88.smooth_rate(width=5*ms),
                PRM89.smooth_rate(width=5*ms),
                PRM90.smooth_rate(width=5*ms),
                PRM91.smooth_rate(width=5*ms),
                PRM92.smooth_rate(width=5*ms),
                PRM93.smooth_rate(width=5*ms),
                PRM94.smooth_rate(width=5*ms),
                PRM95.smooth_rate(width=5*ms),
                PRM96.smooth_rate(width=5*ms),
                PRM97.smooth_rate(width=5*ms),
                PRM98.smooth_rate(width=5*ms),
                PRM99.smooth_rate(width=5*ms),
                PRM100.smooth_rate(width=5*ms),
                PRM101.smooth_rate(width=5*ms),
                PRM102.smooth_rate(width=5*ms),
                PRM103.smooth_rate(width=5*ms),
                PRM104.smooth_rate(width=5*ms),
                PRM105.smooth_rate(width=5*ms),
                PRM106.smooth_rate(width=5*ms),
                PRM107.smooth_rate(width=5*ms),
                PRM108.smooth_rate(width=5*ms),
                PRM109.smooth_rate(width=5*ms),
                PRM110.smooth_rate(width=5*ms),
                PRM111.smooth_rate(width=5*ms),
                PRM112.smooth_rate(width=5*ms),
                PRM113.smooth_rate(width=5*ms),
                PRM114.smooth_rate(width=5*ms),
                PRM115.smooth_rate(width=5*ms),
                PRM116.smooth_rate(width=5*ms),
                PRM117.smooth_rate(width=5*ms),
                PRM118.smooth_rate(width=5*ms),
                PRM119.smooth_rate(width=5*ms),
                PRM120.smooth_rate(width=5*ms),
                PRM121.smooth_rate(width=5*ms),
                PRM122.smooth_rate(width=5*ms),
                PRM123.smooth_rate(width=5*ms),
                PRM124.smooth_rate(width=5*ms),
                PRM125.smooth_rate(width=5*ms),
                PRM126.smooth_rate(width=5*ms),
                PRM127.smooth_rate(width=5*ms),
                PRM128.smooth_rate(width=5*ms),
                PRM129.smooth_rate(width=5*ms),
                PRM130.smooth_rate(width=5*ms),
                PRM131.smooth_rate(width=5*ms),
                PRM132.smooth_rate(width=5*ms),
                PRM133.smooth_rate(width=5*ms),
                PRM134.smooth_rate(width=5*ms),
                PRM135.smooth_rate(width=5*ms),
                PRM136.smooth_rate(width=5*ms),
                PRM137.smooth_rate(width=5*ms),
                PRM138.smooth_rate(width=5*ms),
                PRM139.smooth_rate(width=5*ms),
                PRM140.smooth_rate(width=5*ms),
                PRM141.smooth_rate(width=5*ms),
                PRM142.smooth_rate(width=5*ms),
                PRM143.smooth_rate(width=5*ms),
                PRM144.smooth_rate(width=5*ms),
                PRM145.smooth_rate(width=5*ms),
                PRM146.smooth_rate(width=5*ms),
                PRM147.smooth_rate(width=5*ms),
                PRM148.smooth_rate(width=5*ms),
                PRM149.smooth_rate(width=5*ms),
                PRM150.smooth_rate(width=5*ms),
                PRM151.smooth_rate(width=5*ms),
                PRM152.smooth_rate(width=5*ms),
                PRM153.smooth_rate(width=5*ms),
                PRM154.smooth_rate(width=5*ms),
                PRM155.smooth_rate(width=5*ms),
                PRM156.smooth_rate(width=5*ms),
                PRM157.smooth_rate(width=5*ms),
                PRM158.smooth_rate(width=5*ms),
                PRM159.smooth_rate(width=5*ms),
                PRM160.smooth_rate(width=5*ms),
                PRM161.smooth_rate(width=5*ms),
                PRM162.smooth_rate(width=5*ms),
                PRM163.smooth_rate(width=5*ms),
                PRM164.smooth_rate(width=5*ms),
                PRM165.smooth_rate(width=5*ms),
                PRM166.smooth_rate(width=5*ms),
                PRM167.smooth_rate(width=5*ms),
                PRM168.smooth_rate(width=5*ms),
                PRM169.smooth_rate(width=5*ms),
                PRM170.smooth_rate(width=5*ms),
                PRM171.smooth_rate(width=5*ms),
                PRM172.smooth_rate(width=5*ms),
                PRM173.smooth_rate(width=5*ms),
                PRM174.smooth_rate(width=5*ms),
                PRM175.smooth_rate(width=5*ms),
                PRM176.smooth_rate(width=5*ms),
                PRM177.smooth_rate(width=5*ms),
                PRM178.smooth_rate(width=5*ms),
                PRM179.smooth_rate(width=5*ms),
                PRM180.smooth_rate(width=5*ms),
                PRM181.smooth_rate(width=5*ms),
                PRM182.smooth_rate(width=5*ms),
                PRM183.smooth_rate(width=5*ms),
                PRM184.smooth_rate(width=5*ms),
                PRM185.smooth_rate(width=5*ms),
                PRM186.smooth_rate(width=5*ms),
                PRM187.smooth_rate(width=5*ms),
                PRM188.smooth_rate(width=5*ms),
                PRM189.smooth_rate(width=5*ms),
                PRM190.smooth_rate(width=5*ms),
                PRM191.smooth_rate(width=5*ms),
                PRM192.smooth_rate(width=5*ms),
                PRM193.smooth_rate(width=5*ms),
                PRM194.smooth_rate(width=5*ms),
                PRM195.smooth_rate(width=5*ms),
                PRM196.smooth_rate(width=5*ms),
                PRM197.smooth_rate(width=5*ms),
                PRM198.smooth_rate(width=5*ms),
                PRM199.smooth_rate(width=5*ms),
                PRM200.smooth_rate(width=5*ms),
                PRM201.smooth_rate(width=5*ms),
                PRM202.smooth_rate(width=5*ms),
                PRM203.smooth_rate(width=5*ms),
                PRM204.smooth_rate(width=5*ms),
                PRM205.smooth_rate(width=5*ms),
                PRM206.smooth_rate(width=5*ms),
                PRM207.smooth_rate(width=5*ms),
                PRM208.smooth_rate(width=5*ms),
                PRM209.smooth_rate(width=5*ms),
                PRM210.smooth_rate(width=5*ms),
                PRM211.smooth_rate(width=5*ms),
                PRM212.smooth_rate(width=5*ms),
                PRM213.smooth_rate(width=5*ms),
                PRM214.smooth_rate(width=5*ms),
                PRM215.smooth_rate(width=5*ms),
                PRM216.smooth_rate(width=5*ms),
                PRM217.smooth_rate(width=5*ms),
                PRM218.smooth_rate(width=5*ms),
                PRM219.smooth_rate(width=5*ms),
                PRM220.smooth_rate(width=5*ms),
                PRM221.smooth_rate(width=5*ms),
                PRM222.smooth_rate(width=5*ms),
                PRM223.smooth_rate(width=5*ms),
                PRM224.smooth_rate(width=5*ms),
                PRM225.smooth_rate(width=5*ms),
                PRM226.smooth_rate(width=5*ms),
                PRM227.smooth_rate(width=5*ms),
                PRM228.smooth_rate(width=5*ms),
                PRM229.smooth_rate(width=5*ms),
                PRM230.smooth_rate(width=5*ms),
                PRM231.smooth_rate(width=5*ms),
                PRM232.smooth_rate(width=5*ms),
                PRM233.smooth_rate(width=5*ms),
                PRM234.smooth_rate(width=5*ms),
                PRM235.smooth_rate(width=5*ms),
                PRM236.smooth_rate(width=5*ms),
                PRM237.smooth_rate(width=5*ms),
                PRM238.smooth_rate(width=5*ms),
                PRM239.smooth_rate(width=5*ms),
                PRM240.smooth_rate(width=5*ms),
                PRM241.smooth_rate(width=5*ms),
                PRM242.smooth_rate(width=5*ms),
                PRM243.smooth_rate(width=5*ms),
                PRM244.smooth_rate(width=5*ms),
                PRM245.smooth_rate(width=5*ms),
                PRM246.smooth_rate(width=5*ms),
                PRM247.smooth_rate(width=5*ms),
                PRM248.smooth_rate(width=5*ms),
                PRM249.smooth_rate(width=5*ms),
                PRM250.smooth_rate(width=5*ms),
                PRM251.smooth_rate(width=5*ms),
                PRM252.smooth_rate(width=5*ms),
                PRM253.smooth_rate(width=5*ms),
                PRM254.smooth_rate(width=5*ms),
                PRM255.smooth_rate(width=5*ms),]

    fr_pen = [PRM_PEN0.smooth_rate(width=5*ms),
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
    
    fr = np.array(fr_epg)
    t = np.linspace(0, len(fr[0])/10000, len(fr[0]))
    print(fr.shape) # (256, time_length)
    # Reshape fr from (256, time_length) to (16, 16, time_length)
    # This organizes the data into 16 rings with 16 neurons each
    time_length = fr.shape[1]
    fr = fr.reshape(16, 16, time_length)
    print(f"Reshaped fr to: {fr.shape}")  # Should print (16, 16, time_length)
    fr_r = [PRM_R0.smooth_rate(width=5*ms),]
    fr_r = np.array(fr_r)
    fr_pen = np.array(fr_pen)
    return t, fr, fr_r, fr_pen

if __name__ == '__main__':
    t, fr, fr_r, fr_pen = simulator()    