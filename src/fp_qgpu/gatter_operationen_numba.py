import numpy as np
import numba


@numba.njit(cache=True)
def _axis_to_bit_position(number_of_qubits: int, axis_index: int) -> int:
    """
    Convert a tensor axis index to the corresponding bit position in the flat index.

    Tensor state shape is [2] * number_of_qubits in C-order.
    Axis 0 is the most significant bit in the flattened basis index.
    """
    return number_of_qubits - 1 - axis_index


@numba.njit(cache=True)
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
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)


@numba.njit(cache=True)
def u_gate_numba_compatible_two_loops(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a single-qubit gate without using numpy.einsum.

    This variant uses exactly two loops and explicit index composition to avoid
    per-element bit extraction in the inner update.
    """
    # Step 1: flatten input tensor state and allocate output.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 2: map the target tensor axis to the corresponding bit position q.
    q = _axis_to_bit_position(number_of_qubits, acting_on)
    two_pow_q = 2**q

    # Step 3: precompute loop ranges for the two-loop traversal.
    # upper: bits above q
    # lower: bits below q
    upper_count = 2 ** (number_of_qubits - q - 1)
    lower_count = 2**q
    upper_stride = 2 ** (q + 1)

    # Step 4: iterate over all (upper, lower) pairs and construct:
    #   idx0 = idx_lower + idx_upper + 0 * 2**q
    #   idx1 = idx_lower + idx_upper + 1 * 2**q
    for upper in range(upper_count):
        idx_upper = upper * upper_stride
        for lower in range(lower_count):
            idx_lower = lower

            idx0 = idx_lower + idx_upper + 0 * two_pow_q
            idx1 = idx_lower + idx_upper + 1 * two_pow_q

            amp0 = input_state[idx0]
            amp1 = input_state[idx1]

            output_state[idx0] = u[0, 0] * amp0 + u[0, 1] * amp1
            output_state[idx1] = u[1, 0] * amp0 + u[1, 1] * amp1

    # Step 5: reshape to tensor form, matching existing API behavior.
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)


@numba.njit(cache=True)
def u_gate_numba_compatible_three_loops(
    number_of_qubits: int, acting_on: int, u: np.ndarray, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a single-qubit gate without using numpy.einsum.

    This variant traverses the state with exactly three nested loops to avoid
    per-index bit extraction and process two-state blocks directly.
    """
    # Step 1: flatten input tensor state and allocate output.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 2: map the target tensor axis to the corresponding bit position.
    target_bit_position = _axis_to_bit_position(number_of_qubits, acting_on)
    target_bit_weight = 2**target_bit_position

    # Step 3: precompute the three-loop traversal ranges.
    # upper: bits above the target bit
    # middle: fixed separator level (size 1) to keep a uniform 3-loop layout
    # lower: bits below the target bit
    upper_count = 2 ** (number_of_qubits - target_bit_position - 1)
    middle_count = 1
    lower_count = 2**target_bit_position

    upper_stride = 2 ** (target_bit_position + 1)
    middle_stride = 2**target_bit_position

    # Step 4: iterate over all two-state blocks (i0, i1) where only the target
    # bit differs, and apply the 2x2 gate matrix explicitly.
    for upper in range(upper_count):
        upper_base = upper * upper_stride
        for middle in range(middle_count):
            middle_base = upper_base + middle * middle_stride
            for lower in range(lower_count):
                i0 = middle_base + lower
                i1 = i0 + target_bit_weight

                amp0 = input_state[i0]
                amp1 = input_state[i1]

                output_state[i0] = u[0, 0] * amp0 + u[0, 1] * amp1
                output_state[i1] = u[1, 0] * amp0 + u[1, 1] * amp1

    # Step 5: reshape to tensor form, matching existing API behavior.
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)


@numba.njit(cache=True)
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
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)


@numba.njit(cache=True)
def cx_numba_compatible_three_loops(
    number_of_qubits: int, control: int, target: int, vec: np.ndarray
) -> np.ndarray:
    """
    Apply a CX gate without using numpy.einsum.

    This variant avoids per-index bit extraction by traversing the state in
    structured blocks with exactly three nested loops.
    """
    # Step 1: flatten input tensor state and allocate output.
    input_state = np.ascontiguousarray(vec.reshape(-1))
    state_size = input_state.size
    output_state = np.zeros(state_size, dtype=np.complex128)

    # Step 2: map tensor axes to bit positions in flattened indexing.
    control_bit_position = _axis_to_bit_position(number_of_qubits, control)
    target_bit_position = _axis_to_bit_position(number_of_qubits, target)

    # Step 3: identify the higher and lower bit positions.
    # We iterate over blocks where these two bits are the only varying bits.
    if control_bit_position > target_bit_position:
        higher_bit_position = control_bit_position
        lower_bit_position = target_bit_position
        control_is_higher_bit = True
    else:
        higher_bit_position = target_bit_position
        lower_bit_position = control_bit_position
        control_is_higher_bit = False

    higher_bit_weight = 2**higher_bit_position
    lower_bit_weight = 2**lower_bit_position

    # Step 4: precompute loop ranges for the three-loop traversal.
    # upper: bits above the higher of control/target
    # middle: bits between higher and lower
    # lower: bits below the lower of control/target
    upper_count = 2 ** (number_of_qubits - higher_bit_position - 1)
    middle_count = 2 ** (higher_bit_position - lower_bit_position - 1)
    lower_count = 2**lower_bit_position

    upper_stride = 2 ** (higher_bit_position + 1)
    middle_stride = 2 ** (lower_bit_position + 1)

    # Step 5: walk through the state in three nested loops.
    # For each (upper, middle, lower), we get one 4-state block:
    # i00, i01, i10, i11 for (higher_bit, lower_bit) in {0,1} x {0,1}.
    for upper in range(upper_count):
        upper_base = upper * upper_stride
        for middle in range(middle_count):
            middle_base = upper_base + middle * middle_stride
            for lower in range(lower_count):
                i00 = middle_base + lower
                i01 = i00 + lower_bit_weight
                i10 = i00 + higher_bit_weight
                i11 = i10 + lower_bit_weight

                # Step 6: apply CX routing inside each 4-state block.
                # If control is the higher bit:
                #   i00, i01 stay; i10 <-> i11 swap.
                # If control is the lower bit:
                #   i00, i10 stay; i01 <-> i11 swap.
                if control_is_higher_bit:
                    output_state[i00] = input_state[i00]
                    output_state[i01] = input_state[i01]
                    output_state[i10] = input_state[i11]
                    output_state[i11] = input_state[i10]
                else:
                    output_state[i00] = input_state[i00]
                    output_state[i10] = input_state[i10]
                    output_state[i01] = input_state[i11]
                    output_state[i11] = input_state[i01]

    # Step 7: reshape to tensor form, matching existing API behavior.
    # Using vec.shape keeps the dimensionality explicit and Numba-friendly.
    return output_state.reshape(vec.shape)
