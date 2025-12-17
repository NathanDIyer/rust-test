//! High-performance Newton-Raphson Power Flow Solver in WebAssembly
//! Optimized for power systems analysis with up to 500 buses

#![no_std]

use core::f64::consts::PI;

// Maximum system size (optimized for 118-bus test case)
const MAX_BUSES: usize = 150;
const MAX_BRANCHES: usize = 250;

// Static memory allocation for WASM (no heap)
static mut N_BUSES: usize = 0;
static mut N_BRANCHES: usize = 0;
static mut BASE_MVA: f64 = 100.0;

// Bus data: [vm, va, psch, qsch, type] per bus
// type: 0=PQ, 1=PV, 2=Slack
static mut BUS_VM: [f64; MAX_BUSES] = [1.0; MAX_BUSES];
static mut BUS_VA: [f64; MAX_BUSES] = [0.0; MAX_BUSES];
static mut BUS_PSCH: [f64; MAX_BUSES] = [0.0; MAX_BUSES];
static mut BUS_QSCH: [f64; MAX_BUSES] = [0.0; MAX_BUSES];
static mut BUS_TYPE: [u8; MAX_BUSES] = [0; MAX_BUSES];

// Branch data: [from, to, r, x, b, tap, status]
static mut BR_FROM: [usize; MAX_BRANCHES] = [0; MAX_BRANCHES];
static mut BR_TO: [usize; MAX_BRANCHES] = [0; MAX_BRANCHES];
static mut BR_R: [f64; MAX_BRANCHES] = [0.0; MAX_BRANCHES];
static mut BR_X: [f64; MAX_BRANCHES] = [0.0; MAX_BRANCHES];
static mut BR_B: [f64; MAX_BRANCHES] = [0.0; MAX_BRANCHES];
static mut BR_TAP: [f64; MAX_BRANCHES] = [1.0; MAX_BRANCHES];
static mut BR_STATUS: [u8; MAX_BRANCHES] = [1; MAX_BRANCHES];

// Branch results: [p_from, q_from, p_to, q_to]
static mut BR_PFLOW: [f64; MAX_BRANCHES] = [0.0; MAX_BRANCHES];
static mut BR_QFLOW: [f64; MAX_BRANCHES] = [0.0; MAX_BRANCHES];

// Y-bus matrix (sparse COO format would be better, but dense for simplicity)
// Using separate G and B matrices
static mut YBUS_G: [[f64; MAX_BUSES]; MAX_BUSES] = [[0.0; MAX_BUSES]; MAX_BUSES];
static mut YBUS_B: [[f64; MAX_BUSES]; MAX_BUSES] = [[0.0; MAX_BUSES]; MAX_BUSES];

// Jacobian and work arrays (sized for 2n-1 variables where n=MAX_BUSES)
static mut JACOBIAN: [[f64; 300]; 300] = [[0.0; 300]; 300];
static mut MISMATCH: [f64; 300] = [0.0; 300];
static mut DELTA: [f64; 300] = [0.0; 300];

// Calculated power
static mut PCALC: [f64; MAX_BUSES] = [0.0; MAX_BUSES];
static mut QCALC: [f64; MAX_BUSES] = [0.0; MAX_BUSES];

// Iteration info
static mut ITERATIONS: u32 = 0;
static mut MAX_MISMATCH: f64 = 0.0;
static mut CONVERGED: u8 = 0;

// PQ and PV bus indices
static mut PQ_BUSES: [usize; MAX_BUSES] = [0; MAX_BUSES];
static mut PV_BUSES: [usize; MAX_BUSES] = [0; MAX_BUSES];
static mut N_PQ: usize = 0;
static mut N_PV: usize = 0;
static mut SLACK_BUS: usize = 0;

// ============================================
// WASM Exports
// ============================================

#[no_mangle]
pub extern "C" fn init(n_buses: usize, n_branches: usize, base_mva: f64) {
    unsafe {
        N_BUSES = n_buses.min(MAX_BUSES);
        N_BRANCHES = n_branches.min(MAX_BRANCHES);
        BASE_MVA = base_mva;

        // Reset all arrays
        for i in 0..MAX_BUSES {
            BUS_VM[i] = 1.0;
            BUS_VA[i] = 0.0;
            BUS_PSCH[i] = 0.0;
            BUS_QSCH[i] = 0.0;
            BUS_TYPE[i] = 0;
            PCALC[i] = 0.0;
            QCALC[i] = 0.0;

            for j in 0..MAX_BUSES {
                YBUS_G[i][j] = 0.0;
                YBUS_B[i][j] = 0.0;
            }
        }

        for i in 0..MAX_BRANCHES {
            BR_FROM[i] = 0;
            BR_TO[i] = 0;
            BR_R[i] = 0.01;
            BR_X[i] = 0.1;
            BR_B[i] = 0.0;
            BR_TAP[i] = 1.0;
            BR_STATUS[i] = 1;
            BR_PFLOW[i] = 0.0;
            BR_QFLOW[i] = 0.0;
        }

        ITERATIONS = 0;
        MAX_MISMATCH = 0.0;
        CONVERGED = 0;
    }
}

#[no_mangle]
pub extern "C" fn set_bus(idx: usize, vm: f64, va: f64, psch: f64, qsch: f64, bus_type: u8) {
    unsafe {
        if idx < N_BUSES {
            BUS_VM[idx] = vm;
            BUS_VA[idx] = va;
            BUS_PSCH[idx] = psch;
            BUS_QSCH[idx] = qsch;
            BUS_TYPE[idx] = bus_type;
        }
    }
}

#[no_mangle]
pub extern "C" fn set_branch(idx: usize, from: usize, to: usize, r: f64, x: f64, b: f64, tap: f64, status: u8) {
    unsafe {
        if idx < N_BRANCHES {
            BR_FROM[idx] = from;
            BR_TO[idx] = to;
            BR_R[idx] = r;
            BR_X[idx] = x;
            BR_B[idx] = b;
            BR_TAP[idx] = if tap == 0.0 { 1.0 } else { tap };
            BR_STATUS[idx] = status;
        }
    }
}

#[no_mangle]
pub extern "C" fn get_bus_vm(idx: usize) -> f64 {
    unsafe { if idx < N_BUSES { BUS_VM[idx] } else { 0.0 } }
}

#[no_mangle]
pub extern "C" fn get_bus_va(idx: usize) -> f64 {
    unsafe { if idx < N_BUSES { BUS_VA[idx] } else { 0.0 } }
}

#[no_mangle]
pub extern "C" fn get_branch_pflow(idx: usize) -> f64 {
    unsafe { if idx < N_BRANCHES { BR_PFLOW[idx] } else { 0.0 } }
}

#[no_mangle]
pub extern "C" fn get_branch_qflow(idx: usize) -> f64 {
    unsafe { if idx < N_BRANCHES { BR_QFLOW[idx] } else { 0.0 } }
}

#[no_mangle]
pub extern "C" fn get_iterations() -> u32 {
    unsafe { ITERATIONS }
}

#[no_mangle]
pub extern "C" fn get_max_mismatch() -> f64 {
    unsafe { MAX_MISMATCH }
}

#[no_mangle]
pub extern "C" fn get_converged() -> u8 {
    unsafe { CONVERGED }
}

// ============================================
// Y-Bus Formation
// ============================================

#[no_mangle]
pub extern "C" fn build_ybus() {
    unsafe {
        let n = N_BUSES;

        // Clear Y-bus
        for i in 0..n {
            for j in 0..n {
                YBUS_G[i][j] = 0.0;
                YBUS_B[i][j] = 0.0;
            }
        }

        // Add branch admittances
        for k in 0..N_BRANCHES {
            if BR_STATUS[k] == 0 { continue; }

            let i = BR_FROM[k];
            let j = BR_TO[k];
            if i >= n || j >= n { continue; }

            let r = BR_R[k];
            let x = BR_X[k];
            let b = BR_B[k];
            let tap = BR_TAP[k];

            // Series admittance: y = 1/(r + jx)
            let denom = r * r + x * x;
            if denom < 1e-12 { continue; }

            let g = r / denom;
            let b_series = -x / denom;

            if tap == 1.0 {
                // Simple line (no tap)
                // Off-diagonal
                YBUS_G[i][j] -= g;
                YBUS_B[i][j] -= b_series;
                YBUS_G[j][i] -= g;
                YBUS_B[j][i] -= b_series;

                // Diagonal (with line charging)
                YBUS_G[i][i] += g;
                YBUS_B[i][i] += b_series + b / 2.0;
                YBUS_G[j][j] += g;
                YBUS_B[j][j] += b_series + b / 2.0;
            } else {
                // Transformer with tap
                let tap2 = tap * tap;

                // Y_ii = y / tap^2
                YBUS_G[i][i] += g / tap2;
                YBUS_B[i][i] += b_series / tap2;

                // Y_jj = y
                YBUS_G[j][j] += g;
                YBUS_B[j][j] += b_series;

                // Y_ij = Y_ji = -y / tap
                YBUS_G[i][j] -= g / tap;
                YBUS_B[i][j] -= b_series / tap;
                YBUS_G[j][i] -= g / tap;
                YBUS_B[j][i] -= b_series / tap;
            }
        }
    }
}

// ============================================
// Power Calculation
// ============================================

fn calculate_power() {
    unsafe {
        let n = N_BUSES;

        for i in 0..n {
            let mut p = 0.0;
            let mut q = 0.0;
            let vi = BUS_VM[i];
            let ti = BUS_VA[i];

            for j in 0..n {
                let vj = BUS_VM[j];
                let tj = BUS_VA[j];
                let gij = YBUS_G[i][j];
                let bij = YBUS_B[i][j];
                let tij = ti - tj;

                let cos_tij = cos_approx(tij);
                let sin_tij = sin_approx(tij);

                p += vi * vj * (gij * cos_tij + bij * sin_tij);
                q += vi * vj * (gij * sin_tij - bij * cos_tij);
            }

            PCALC[i] = p;
            QCALC[i] = q;
        }
    }
}

// ============================================
// Newton-Raphson Solver
// ============================================

#[no_mangle]
pub extern "C" fn solve(max_iter: u32, tolerance: f64) -> u8 {
    unsafe {
        let n = N_BUSES;

        // Identify bus types
        N_PQ = 0;
        N_PV = 0;
        SLACK_BUS = 0;

        for i in 0..n {
            match BUS_TYPE[i] {
                0 => { PQ_BUSES[N_PQ] = i; N_PQ += 1; }
                1 => { PV_BUSES[N_PV] = i; N_PV += 1; }
                2 => { SLACK_BUS = i; }
                _ => {}
            }
        }

        let n_pq = N_PQ;
        let n_pv = N_PV;
        let slack = SLACK_BUS;

        // Jacobian size: (n-1) for P equations + n_pq for Q equations
        let n_p = n - 1;  // All buses except slack
        let n_q = n_pq;   // Only PQ buses
        let jsize = n_p + n_q;

        if jsize == 0 {
            CONVERGED = 1;
            ITERATIONS = 0;
            return 1;
        }

        CONVERGED = 0;

        for iter in 0..max_iter {
            ITERATIONS = iter + 1;

            // Calculate power injections
            calculate_power();

            // Build mismatch vector
            let mut max_mis: f64 = 0.0;
            let mut mis_idx = 0;

            // P mismatches for all non-slack buses
            for i in 0..n {
                if i == slack { continue; }
                let dp = BUS_PSCH[i] - PCALC[i];
                MISMATCH[mis_idx] = dp;
                if dp.abs() > max_mis { max_mis = dp.abs(); }
                mis_idx += 1;
            }

            // Q mismatches for PQ buses only
            for k in 0..n_pq {
                let i = PQ_BUSES[k];
                let dq = BUS_QSCH[i] - QCALC[i];
                MISMATCH[mis_idx] = dq;
                if dq.abs() > max_mis { max_mis = dq.abs(); }
                mis_idx += 1;
            }

            MAX_MISMATCH = max_mis;

            // Check convergence
            if max_mis < tolerance {
                CONVERGED = 1;
                calculate_branch_flows();
                return 1;
            }

            // Build Jacobian matrix
            build_jacobian(n, slack, n_pq);

            // Solve linear system: J * delta = mismatch
            if !solve_linear_system(jsize) {
                return 0;  // Singular matrix
            }

            // Update state variables
            let mut delta_idx = 0;

            // Update angles for non-slack buses
            for i in 0..n {
                if i == slack { continue; }
                BUS_VA[i] += DELTA[delta_idx];
                delta_idx += 1;
            }

            // Update voltage magnitudes for PQ buses
            for k in 0..n_pq {
                let i = PQ_BUSES[k];
                BUS_VM[i] *= 1.0 + DELTA[delta_idx];
                delta_idx += 1;
            }
        }

        // Did not converge
        calculate_branch_flows();
        0
    }
}

fn build_jacobian(n: usize, slack: usize, n_pq: usize) {
    unsafe {
        // Build non-slack bus index mapping
        let mut nonslack_to_idx: [usize; MAX_BUSES] = [0; MAX_BUSES];
        let mut idx_to_nonslack: [usize; MAX_BUSES] = [0; MAX_BUSES];
        let mut ns_count = 0;

        for i in 0..n {
            if i != slack {
                nonslack_to_idx[i] = ns_count;
                idx_to_nonslack[ns_count] = i;
                ns_count += 1;
            }
        }

        // Build PQ bus index mapping for Q/V submatrices
        let mut pq_to_idx: [usize; MAX_BUSES] = [0; MAX_BUSES];
        for k in 0..n_pq {
            pq_to_idx[PQ_BUSES[k]] = k;
        }

        let n_p = ns_count;

        // J1: dP/dtheta (n_p x n_p)
        for ii in 0..n_p {
            let i = idx_to_nonslack[ii];
            for jj in 0..n_p {
                let j = idx_to_nonslack[jj];
                let gij = YBUS_G[i][j];
                let bij = YBUS_B[i][j];
                let vi = BUS_VM[i];
                let vj = BUS_VM[j];
                let tij = BUS_VA[i] - BUS_VA[j];

                if i == j {
                    // Diagonal: dPi/dtheta_i = -Qi - Bii*Vi^2
                    JACOBIAN[ii][jj] = -QCALC[i] - bij * vi * vi;
                } else {
                    // Off-diagonal: dPi/dtheta_j = Vi*Vj*(Gij*sin - Bij*cos)
                    let sin_tij = sin_approx(tij);
                    let cos_tij = cos_approx(tij);
                    JACOBIAN[ii][jj] = vi * vj * (gij * sin_tij - bij * cos_tij);
                }
            }
        }

        // J2: dP/dV (n_p x n_pq) - only for PQ buses
        for ii in 0..n_p {
            let i = idx_to_nonslack[ii];
            for jj in 0..n_pq {
                let j = PQ_BUSES[jj];
                let col = n_p + jj;
                let gij = YBUS_G[i][j];
                let bij = YBUS_B[i][j];
                let vi = BUS_VM[i];
                let vj = BUS_VM[j];
                let tij = BUS_VA[i] - BUS_VA[j];

                if i == j {
                    // Diagonal: dPi/dVi = Pi/Vi + Gii*Vi
                    JACOBIAN[ii][col] = PCALC[i] / vi + gij * vi;
                } else {
                    // Off-diagonal: dPi/dVj = Vi*(Gij*cos + Bij*sin)
                    let sin_tij = sin_approx(tij);
                    let cos_tij = cos_approx(tij);
                    JACOBIAN[ii][col] = vi * (gij * cos_tij + bij * sin_tij);
                }
            }
        }

        // J3: dQ/dtheta (n_pq x n_p)
        for ii in 0..n_pq {
            let i = PQ_BUSES[ii];
            let row = n_p + ii;
            for jj in 0..n_p {
                let j = idx_to_nonslack[jj];
                let gij = YBUS_G[i][j];
                let bij = YBUS_B[i][j];
                let vi = BUS_VM[i];
                let vj = BUS_VM[j];
                let tij = BUS_VA[i] - BUS_VA[j];

                if i == j {
                    // Diagonal: dQi/dtheta_i = Pi - Gii*Vi^2
                    JACOBIAN[row][jj] = PCALC[i] - gij * vi * vi;
                } else {
                    // Off-diagonal: dQi/dtheta_j = -Vi*Vj*(Gij*cos + Bij*sin)
                    let sin_tij = sin_approx(tij);
                    let cos_tij = cos_approx(tij);
                    JACOBIAN[row][jj] = -vi * vj * (gij * cos_tij + bij * sin_tij);
                }
            }
        }

        // J4: dQ/dV (n_pq x n_pq)
        for ii in 0..n_pq {
            let i = PQ_BUSES[ii];
            let row = n_p + ii;
            for jj in 0..n_pq {
                let j = PQ_BUSES[jj];
                let col = n_p + jj;
                let gij = YBUS_G[i][j];
                let bij = YBUS_B[i][j];
                let vi = BUS_VM[i];
                let vj = BUS_VM[j];
                let tij = BUS_VA[i] - BUS_VA[j];

                if i == j {
                    // Diagonal: dQi/dVi = Qi/Vi - Bii*Vi
                    JACOBIAN[row][col] = QCALC[i] / vi - bij * vi;
                } else {
                    // Off-diagonal: dQi/dVj = Vi*(Gij*sin - Bij*cos)
                    let sin_tij = sin_approx(tij);
                    let cos_tij = cos_approx(tij);
                    JACOBIAN[row][col] = vi * (gij * sin_tij - bij * cos_tij);
                }
            }
        }
    }
}

fn solve_linear_system(n: usize) -> bool {
    unsafe {
        // LU decomposition with partial pivoting (in-place)
        for k in 0..n {
            // Find pivot
            let mut max_val = JACOBIAN[k][k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = JACOBIAN[i][k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            if max_val < 1e-12 {
                return false;  // Singular
            }

            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    let tmp = JACOBIAN[k][j];
                    JACOBIAN[k][j] = JACOBIAN[max_row][j];
                    JACOBIAN[max_row][j] = tmp;
                }
                let tmp = MISMATCH[k];
                MISMATCH[k] = MISMATCH[max_row];
                MISMATCH[max_row] = tmp;
            }

            // Elimination
            for i in (k + 1)..n {
                let factor = JACOBIAN[i][k] / JACOBIAN[k][k];
                JACOBIAN[i][k] = factor;  // Store L factor
                for j in (k + 1)..n {
                    JACOBIAN[i][j] -= factor * JACOBIAN[k][j];
                }
                MISMATCH[i] -= factor * MISMATCH[k];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            let mut sum = MISMATCH[i];
            for j in (i + 1)..n {
                sum -= JACOBIAN[i][j] * DELTA[j];
            }
            DELTA[i] = sum / JACOBIAN[i][i];
        }

        true
    }
}

// ============================================
// Branch Flow Calculation
// ============================================

fn calculate_branch_flows() {
    unsafe {
        for k in 0..N_BRANCHES {
            if BR_STATUS[k] == 0 {
                BR_PFLOW[k] = 0.0;
                BR_QFLOW[k] = 0.0;
                continue;
            }

            let i = BR_FROM[k];
            let j = BR_TO[k];

            let vi = BUS_VM[i];
            let vj = BUS_VM[j];
            let ti = BUS_VA[i];
            let tj = BUS_VA[j];

            let r = BR_R[k];
            let x = BR_X[k];
            let b = BR_B[k];
            let tap = BR_TAP[k];

            // Series admittance
            let denom = r * r + x * x;
            if denom < 1e-12 { continue; }
            let g = r / denom;
            let b_s = -x / denom;

            // Complex voltages
            let vi_r = vi * cos_approx(ti);
            let vi_i = vi * sin_approx(ti);
            let vj_r = vj * cos_approx(tj);
            let vj_i = vj * sin_approx(tj);

            // Current from i to j
            let dv_r = vi_r / tap - vj_r;
            let dv_i = vi_i / tap - vj_i;

            let i_r = g * dv_r - b_s * dv_i;
            let i_i = g * dv_i + b_s * dv_r;

            // Add shunt current at from end
            let ish_r = -(b / 2.0) * vi_i / tap;
            let ish_i = (b / 2.0) * vi_r / tap;

            let itot_r = i_r + ish_r;
            let itot_i = i_i + ish_i;

            // Power: S = V * I*
            let p = (vi_r / tap) * itot_r + (vi_i / tap) * itot_i;
            let q = (vi_i / tap) * itot_r - (vi_r / tap) * itot_i;

            BR_PFLOW[k] = p * BASE_MVA;
            BR_QFLOW[k] = q * BASE_MVA;
        }
    }
}

// ============================================
// Math Approximations (no std lib)
// ============================================

fn sin_approx(x: f64) -> f64 {
    // Normalize to [-pi, pi]
    let mut x = x % (2.0 * PI);
    if x > PI { x -= 2.0 * PI; }
    if x < -PI { x += 2.0 * PI; }

    // Taylor series approximation
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    let x9 = x7 * x2;

    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0 + x9 / 362880.0
}

fn cos_approx(x: f64) -> f64 {
    // Normalize to [-pi, pi]
    let mut x = x % (2.0 * PI);
    if x > PI { x -= 2.0 * PI; }
    if x < -PI { x += 2.0 * PI; }

    // Taylor series approximation
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x6 * x2;

    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0
}

// Panic handler for no_std
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}
