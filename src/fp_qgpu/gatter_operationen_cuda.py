import math

import numpy as np
from numba import cuda


def _axis_to_bit_position(number_of_qubits: int, axis_index: int) -> int:
    """Convert tensor-axis index to flat-index bit position."""
    return number_of_qubits - 1 - axis_index


@cuda.jit
def app_u_kernel(
    number_of_qubits: int,
    bit_position: int,
    u: np.ndarray,
    input_state: np.ndarray,
    output_state: np.ndarray,
) -> None:
    """Apply a single-qubit U gate on the GPU.

    One CUDA thread updates exactly one basis-state pair (|...0...>, |...1...>)
    that differs only in the target bit.
    """
    pair_index = cuda.grid(1)
    total_pairs = input_state.size // 2
    _ = number_of_qubits  # Kept for API consistency with other backends.

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

    u00 = u[0, 0]
    u01 = u[0, 1]
    u10 = u[1, 0]
    u11 = u[1, 1]

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
    input_state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    output_state = np.empty_like(input_state)
    u_local = np.ascontiguousarray(u, dtype=np.complex128)

    d_input = cuda.to_device(input_state)
    d_output = cuda.device_array_like(output_state)
    d_u = cuda.to_device(u_local)

    total_pairs = input_state.size // 2
    blocks_per_grid = math.ceil(total_pairs / threads_per_block)

    app_u_kernel[blocks_per_grid, threads_per_block](
        number_of_qubits,
        bit_position,
        d_u,
        d_input,
        d_output,
    )

    return d_output.copy_to_host().reshape(vec.shape)


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
    state = np.ascontiguousarray(vec.reshape(-1), dtype=np.complex128)
    d_state = cuda.to_device(state)

    blocks_per_grid = math.ceil(state.size / threads_per_block)

    app_cx_kernel[blocks_per_grid, threads_per_block](
        control_bit_position,
        target_bit_position,
        d_state,
    )

    return d_state.copy_to_host().reshape(vec.shape)
