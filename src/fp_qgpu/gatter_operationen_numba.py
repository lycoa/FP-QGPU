import numpy as np


def _axis_to_bit_position(number_of_qubits: int, axis_index: int) -> int:
    """
    Convert a tensor axis index to the corresponding bit position in the flat index.

    Tensor state shape is [2] * number_of_qubits in C-order.
    Axis 0 is the most significant bit in the flattened basis index.
    """
    return number_of_qubits - 1 - axis_index


def u_gate_numba_compatible(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a single-qubit gate without using numpy.einsum.

    This implementation is written in explicit loops and scalar operations so that it
    is straightforward to port to/compile with Numba.
    """
    # Step 1: flatten the input tensor state to a 1D statevector.
    # The original tensor shape is restored at the end.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size

    # Step 2: prepare the output statevector with zeros.
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 3: identify the bit position that corresponds to the chosen tensor axis.
    bit_position = _axis_to_bit_position(number_of_qubits, acting_on)
    bit_weight = 2**bit_position

    # Step 4: iterate over all basis indices of the input state.
    # For each index:
    #   - read the input bit value at "acting_on"
    #   - distribute amplitude to output bit 0 and output bit 1 via U[:, input_bit]
    for input_index in range(state_size):
        input_bit = (input_index // bit_weight) % 2
        amplitude = input_state[input_index]

        # Build both output indices for this basis configuration:
        # one where target bit is 0 and one where target bit is 1.
        # Remove the contribution of the current bit value to force the bit to 0.
        index_with_bit_0 = input_index - input_bit * bit_weight
        # Add the bit weight once to force the bit to 1.
        index_with_bit_1 = index_with_bit_0 + bit_weight

        # Accumulate the two contributions from the gate matrix.
        output_state[index_with_bit_0] += u[0, input_bit] * amplitude
        output_state[index_with_bit_1] += u[1, input_bit] * amplitude

    # Step 5: reshape back to tensor form [2] * number_of_qubits.
    return output_state.reshape([2] * number_of_qubits)


def cx_numba_compatible(
    number_of_qubits: int, control: int, target: int, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a CX gate without using numpy.einsum.

    The rule is explicit:
    - if control bit is 0: amplitude stays at same index
    - if control bit is 1: flip target bit
    """
    # Step 1: flatten input tensor state and allocate output.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 2: map tensor axes to bit positions in flattened indexing.
    control_bit_position = _axis_to_bit_position(number_of_qubits, control)
    target_bit_position = _axis_to_bit_position(number_of_qubits, target)
    control_bit_weight = 2**control_bit_position
    target_bit_weight = 2**target_bit_position

    # Step 3: walk through each input basis index and route its amplitude.
    for input_index in range(state_size):
        amplitude = input_state[input_index]
        control_bit = (input_index // control_bit_weight) % 2
        target_bit = (input_index // target_bit_weight) % 2

        # If control is 1, toggle target bit. Otherwise keep index.
        if control_bit == 1:
            # Toggle target bit arithmetically:
            # - if target_bit == 0, add target_bit_weight
            # - if target_bit == 1, subtract target_bit_weight
            output_index = input_index + (1 - 2 * target_bit) * target_bit_weight
        else:
            output_index = input_index

        output_state[output_index] += amplitude

    # Step 4: reshape to tensor form, matching existing API behavior.
    return output_state.reshape([2] * number_of_qubits)
