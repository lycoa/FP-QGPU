import numpy as np 


def u_gate_numba(number_of_qubits:int, acting_on:int, u:np.ndarray, vec:np.darray) -> np.ndarray:

    num = number_of_qubits
    act_on = acting_on 

    
    vec_enum = enumerate(vec) #index, qubit-state

    lower = vec_enum[0:acting_on]
    target = vec_enum[acting_on]
    upper = vec_enum[(acting_on+1):]
    
    idx_lower = np.sum[qstate*2**index for index, qstate in lower]
    idx_upper = np.sum[qstate*2**index for index, qstate in upper]

    for i in lower: 
        for j in upper: 
            idx0 = idx_lower + idx_upper
            idx1 = idx_lower + idx_upper + [1*2**index for index, qstate in target ][0]

            s0 = vec[idx0]
            s1 = vec[idx1]


    


