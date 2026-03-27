import math

import numpy as np
from numba import cuda


def _axis_to_bit_position(number_of_qubits: int, axis_index: int) -> int:
    """Convert tensor-axis index to flat-index bit position."""
    return number_of_qubits - 1 - axis_index


@cuda.jit
def app_u_kernel(
    bit_position: int,
    input_state: np.ndarray,
    output_state: np.ndarray,
    u00: complex,
    u01: complex,
    u10: complex,
    u11: complex,
) -> None:
    """Apply a single-qubit U gate on the GPU.

    One CUDA thread updates exactly one basis-state pair (|...0...>, |...1...>)
    that differs only in the target bit.
    """
    pair_index = cuda.grid(1)
    total_pairs = input_state.size // 2

    if pair_index >= total_pairs:
        return

    # Build pair indices without per-state loops.
    lower_mask = (1 << bit_position) - 1
    upper = pair_index >> bit_position
    lower = pair_index & lower_mask

    idx0 = (upper << (bit_position + 1)) | lower
    idx1 = idx0 | (1 << bit_position)

    amp0 = input_state[idx0]
    amp1 = input_state[idx1]

    output_state[idx0] = u00 * amp0 + u01 * amp1
    output_state[idx1] = u10 * amp0 + u11 * amp1


@cuda.jit
def app_cx_kernel(
    control_bit_position: int,
    target_bit_position: int,
    state: np.ndarray,
) -> None:
    """Apply an in-place CX gate on the GPU.

    A thread swaps amplitudes only for indices where control=1 and target=0.
    This guarantees each pair is swapped exactly once.
    """
    i = cuda.grid(1)

    if i >= state.size or control_bit_position == target_bit_position:
        return

    control_mask = 1 << control_bit_position
    target_mask = 1 << target_bit_position

    # Swap only one direction (target=0 -> target=1) to avoid double swaps.
    if (i & control_mask) != 0 and (i & target_mask) == 0:
        j = i | target_mask
        tmp = state[i]
        state[i] = state[j]
        state[j] = tmp


def u_gate_cuda(
    number_of_qubits: int,
    acting_on: int,
    u: np.ndarray,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Host helper to launch ``app_u_kernel`` and return the updated state."""
    bit_position = _axis_to_bit_position(number_of_qubits, acting_on)
    original_shape = vec.shape
    input_state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    u_local = np.ascontiguousarray(u, dtype=np.complex128)

    d_input = cuda.device_array(input_state.shape, dtype=np.complex128)
    d_output = cuda.device_array(input_state.shape, dtype=np.complex128)
    d_input.copy_to_device(input_state)

    total_pairs = input_state.size // 2
    blocks_per_grid = math.ceil(total_pairs / threads_per_block)

    app_u_kernel[blocks_per_grid, threads_per_block](
        bit_position,
        d_input,
        d_output,
        u_local[0, 0],
        u_local[0, 1],
        u_local[1, 0],
        u_local[1, 1],
    )

    return d_output.copy_to_host().reshape(original_shape)


def cx_gate_cuda(
    number_of_qubits: int,
    control: int,
    target: int,
    vec: np.ndarray,
    threads_per_block: int = 256,
) -> np.ndarray:
    """Host helper to launch ``app_cx_kernel`` and return the updated state."""
    control_bit_position = _axis_to_bit_position(number_of_qubits, control)
    target_bit_position = _axis_to_bit_position(number_of_qubits, target)
    original_shape = vec.shape
    state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    d_state = cuda.device_array(state.shape, dtype=np.complex128)
    d_state.copy_to_device(state)

    blocks_per_grid = math.ceil(state.size / threads_per_block)

    app_cx_kernel[blocks_per_grid, threads_per_block](
        control_bit_position,
        target_bit_position,
        d_state,
    )

    return d_state.copy_to_host().reshape(original_shape)


def simulate_circuit_cuda(
    number_of_qubits: int,
    circuit: list[list[object]],
    threads_per_block: int = 256,
) -> np.ndarray:
    """Simulate a transpiled [u, cx] circuit while keeping state on GPU."""
    state_size = 2**number_of_qubits
    total_pairs = state_size // 2

    state_host = np.zeros(state_size, dtype=np.complex128)
    state_host[0] = 1.0 + 0.0j

    d_state_a = cuda.device_array(state_host.shape, dtype=np.complex128)
    d_state_b = cuda.device_array(state_host.shape, dtype=np.complex128)
    d_state_a.copy_to_device(state_host)

    blocks_u = math.ceil(total_pairs / threads_per_block)
    blocks_state = math.ceil(state_size / threads_per_block)

    for gate in circuit:
        name = gate[0]
        acting_on = gate[1]

        if name == "u":
            axis_index = number_of_qubits - 1 - acting_on[0]
            bit_position = _axis_to_bit_position(number_of_qubits, axis_index)
            u = np.ascontiguousarray(gate[2], dtype=np.complex128)
            app_u_kernel[blocks_u, threads_per_block](
                bit_position,
                d_state_a,
                d_state_b,
                u[0, 0],
                u[0, 1],
                u[1, 0],
                u[1, 1],
            )
            d_state_a, d_state_b = d_state_b, d_state_a
            continue

        if name == "cx":
            control = number_of_qubits - 1 - acting_on[0]
            target = number_of_qubits - 1 - acting_on[1]
            control_bit_position = _axis_to_bit_position(number_of_qubits, control)
            target_bit_position = _axis_to_bit_position(number_of_qubits, target)
            app_cx_kernel[blocks_state, threads_per_block](
                control_bit_position,
                target_bit_position,
                d_state_a,
            )
            continue

        if name == "barrier" or name == "measure":
            continue

    return d_state_a.copy_to_host()
