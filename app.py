# app.py — Multi-Qubit Quantum Circuit Visualizer (complete, beginner-friendly)
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit.visualization import plot_bloch_vector

# -------------------------
# Helpful constants & funcs
# -------------------------
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def basis_labels(n):
    return [format(i, f"0{n}b") for i in range(2**n)]

def nice_complex(z: complex, digits=4):
    return complex(np.round(z.real, digits), np.round(z.imag, digits))

def bloch_from_rho(rho: np.ndarray):
    """Return Bloch vector (rx, ry, rz) from 2x2 density matrix rho."""
    rx = float(np.real(np.trace(rho @ SX)))
    ry = float(np.real(np.trace(rho @ SY)))
    rz = float(np.real(np.trace(rho @ SZ)))
    return rx, ry, rz

def purity_of_rho(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))

def von_neumann_entropy(rho: np.ndarray, base=2.0):
    # eigenvalues (real, >=0)
    vals = np.linalg.eigvalsh(rho)
    # numerical stability: clip tiny negatives to zero
    vals = np.clip(vals.real, 0.0, 1.0)
    eps = 1e-12
    vals = vals[vals > eps]
    if vals.size == 0:
        return 0.0
    return float(-np.sum(vals * np.log(vals) / np.log(base)))

# -------------------------
# Circuit builder utilities
# -------------------------
def build_circuit(num_qubits: int, gates: list) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for g in gates:
        gate = g["gate"]
        if gate in ("H", "X", "Y", "Z"):
            qc.__getattribute__(gate.lower())(g["target"])
        elif gate in ("RX", "RY", "RZ"):
            theta = float(g.get("theta", 0.0))
            if gate == "RX": qc.rx(theta, g["target"])
            if gate == "RY": qc.ry(theta, g["target"])
            if gate == "RZ": qc.rz(theta, g["target"])
        elif gate == "CX":
            qc.cx(g["control"], g["target"])
        elif gate == "CZ":
            qc.cz(g["control"], g["target"])
        elif gate == "SWAP":
            qc.swap(g["control"], g["target"])
        elif gate == "CCX":  # Toffoli (control1, control2 -> target)
            qc.ccx(g["control1"], g["control2"], g["target"])
    return qc

def preset_fill(name: str, n: int):
    if name == "Bell (Φ+)":
        if n < 2: return []
        return [{"gate":"H","target":0}, {"gate":"CX","control":0,"target":1}]
    if name == "GHZ (3)":
        if n < 3: return []
        return [{"gate":"H","target":0}, {"gate":"CX","control":0,"target":1}, {"gate":"CX","control":1,"target":2}]
    if name == "Product: H on all":
        return [{"gate":"H","target":q} for q in range(n)]
    if name == "W (3)":
        # simple W-ish prep (approx) for demo (requires 3 qubits)
        if n < 3: return []
        return [
            {"gate":"RY","target":0,"theta":2*math.acos(1/math.sqrt(3))},
            {"gate":"CX","control":0,"target":1},
            {"gate":"CX","control":0,"target":2},
        ]
    return []

# -------------------------
# Streamlit UI + Flow
# -------------------------
st.set_page_config(page_title="MQQC Visualizer", layout="wide")
st.title("🔬 Multi-Qubit Quantum Circuit Visualizer (MQQC)")

st.markdown(
    "Build a multi-qubit circuit (1–4 qubits), run a simulation, "
    "inspect the global state, compute reduced single-qubit density matrices "
    "via partial trace, and visualize each qubit on a Bloch sphere."
)

# session state for gates
if "gates" not in st.session_state:
    st.session_state.gates = []

# ----- Step 1: choose qubits -----
st.header("Step 1 — Choose number of qubits")
num_qubits = st.slider("Number of qubits", min_value=1, max_value=4, value=2)
st.info(f"System starts in |{'0'*num_qubits}⟩ (all qubits = |0⟩).")

# presets row
st.write("**Presets (one-click)**")
pcols = st.columns(4)
with pcols[0]:
    if st.button("Bell (Φ+)", use_container_width=True) and num_qubits >= 2:
        st.session_state.gates = preset_fill("Bell (Φ+)", num_qubits)
with pcols[1]:
    if st.button("GHZ (3)", use_container_width=True) and num_qubits >= 3:
        st.session_state.gates = preset_fill("GHZ (3)", num_qubits)
with pcols[2]:
    if st.button("H on all", use_container_width=True):
        st.session_state.gates = preset_fill("Product: H on all", num_qubits)
with pcols[3]:
    if st.button("W (3)", use_container_width=True) and num_qubits >= 3:
        st.session_state.gates = preset_fill("W (3)", num_qubits)

# ----- Step 2: Add gates -----
st.header("Step 2 — Build your circuit (add gates)")
with st.expander("What this builder supports", expanded=False):
    st.write(
        "- Single-qubit: H, X, Y, Z, RX, RY, RZ\n"
        "- Multi-qubit: CX (CNOT), CZ, SWAP\n"
        "- Toffoli (CCX) supported when you choose two controls + a target.\n"
        "Use the 'Add gate' button to append steps. Use 'Remove last' or 'Clear all' to edit."
    )

# gate form
col_a, col_b, col_c = st.columns([2,2,4])
with col_a:
    gate = st.selectbox("Gate", ["H","X","Y","Z","RX","RY","RZ","CX","CZ","SWAP","CCX"])
with col_b:
    # target selector depends on gate
    if gate == "CCX":
        # need target and two controls
        target = st.selectbox("Target qubit", [f"q{i}" for i in range(num_qubits)], index=0)
        control1 = st.selectbox("Control 1", [f"q{i}" for i in range(num_qubits)], index=1 if num_qubits>1 else 0)
        control2 = st.selectbox("Control 2", [f"q{i}" for i in range(num_qubits)], index=2 if num_qubits>2 else 0)
    elif gate in ("CX","CZ","SWAP"):
        control = st.selectbox("Control / qubit A", [f"q{i}" for i in range(num_qubits)], index=0)
        target = st.selectbox("Target / qubit B", [f"q{i}" for i in range(num_qubits)], index=1 if num_qubits>1 else 0)
    else:
        target = st.selectbox("Target qubit", [f"q{i}" for i in range(num_qubits)], index=0)
with col_c:
    theta = None
    if gate in ("RX","RY","RZ"):
        theta = st.slider("Angle θ (rad)", 0.0, float(2*math.pi), float(math.pi/2), step=0.01)
    add_btn, remove_btn, clear_btn = st.columns(3)
    with add_btn:
        if st.button("➕ Add gate"):
            # validation & append
            if gate == "CCX":
                c1 = int(control1[1]); c2 = int(control2[1]); t = int(target[1])
                if t in (c1, c2) or c1 == c2:
                    st.error("Controls and target must be distinct for CCX.")
                else:
                    st.session_state.gates.append({"gate":"CCX","control1":c1,"control2":c2,"target":t})
            elif gate in ("CX","CZ","SWAP"):
                c = int(control[1]); t = int(target[1])
                if c == t:
                    st.error("Control and target must be different.")
                else:
                    st.session_state.gates.append({"gate":gate,"control":c,"target":t})
            elif gate in ("RX","RY","RZ"):
                t = int(target[1])
                st.session_state.gates.append({"gate":gate,"target":t,"theta":float(theta)})
            else:
                t = int(target[1])
                st.session_state.gates.append({"gate":gate,"target":t})
    with remove_btn:
        if st.button("↩️ Remove last"):
            if st.session_state.gates:
                st.session_state.gates.pop()
    with clear_btn:
        if st.button("🗑 Clear all"):
            st.session_state.gates = []

# show current sequence
st.markdown("**Gate sequence**")
if not st.session_state.gates:
    st.info("No gates yet. Add gates above or choose a preset.")
else:
    seq_lines = []
    for i, g in enumerate(st.session_state.gates, start=1):
        if g["gate"] == "CCX":
            seq_lines.append(f"{i}. CCX  controls=q{g['control1']},q{g['control2']}  target=q{g['target']}")
        elif g["gate"] in ("CX","CZ","SWAP"):
            seq_lines.append(f"{i}. {g['gate']}  control=q{g['control']}  target=q{g['target']}")
        elif g["gate"] in ("RX","RY","RZ"):
            seq_lines.append(f"{i}. {g['gate']}  q{g['target']}  θ={g['theta']:.3f}")
        else:
            seq_lines.append(f"{i}. {g['gate']}  q{g['target']}")
    st.code("\n".join(seq_lines), language="text")

# ----- Step 3: visualize circuit -----
st.header("Step 3 — Circuit preview")
qc = build_circuit(num_qubits, st.session_state.gates)
try:
    fig = qc.draw(output="mpl", fold=40)
    st.pyplot(fig, use_container_width=True)
except Exception:
    st.info("Circuit drawing (matplotlib) failed — showing text diagram instead.")
    st.text(qc.draw())

# ----- Step 4: simulate & inspect -----
st.header("Step 4 — Simulate and inspect")
if st.button("▶️ Run Simulation"):
    # simulate
    try:
        sv = Statevector.from_instruction(qc)
    except Exception as e:
        st.error(f"Simulation failed: {e}")
        sv = None

    if sv is not None:
        dm_full = DensityMatrix(sv)  # full density matrix
        st.subheader("Global state (statevector & probabilities)")
        amps = [nice_complex(a) for a in sv.data]
        probs = [float(np.real(abs(a)**2)) for a in sv.data]
        table = {
            "basis |q...q0>": basis_labels(num_qubits),
            "amplitude": amps,
            "probability": [round(p, 6) for p in probs],
        }
        st.table(table)

        st.subheader("Full density matrix (rounded)")
        st.write(np.round(dm_full.data, 6))

        st.divider()
        st.header("Per-qubit reduced states + Bloch spheres")

        cols = st.columns(min(4, num_qubits))
        # for each qubit compute reduced density matrix by tracing out others
        for q in range(num_qubits):
            trace_out = [i for i in range(num_qubits) if i != q]
            reduced = partial_trace(dm_full, trace_out)   # qiskit DensityMatrix
            rho_q = np.array(reduced.data, dtype=complex)

            rx, ry, rz = bloch_from_rho(rho_q)
            pur = purity_of_rho(rho_q)
            entropy = von_neumann_entropy(rho_q)  # in bits

            with cols[q % 4]:
                st.markdown(f"**Qubit q{q}**")
                st.caption("Reduced density matrix (after tracing out other qubits)")
                st.write(np.round(rho_q, 6))
                st.text(f"Bloch vector: (rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f})")
                st.text(f"Purity: {pur:.4f}   |   von Neumann entropy: {entropy:.4f} bits")

                # interpret purity / entropy lightly
                if num_qubits == 1:
                    st.success("Single qubit — pure/mixed is actual state.")
                else:
                    if entropy > 1e-3:
                        st.warning("This qubit is mixed (entropy > 0) — likely entangled/correlated with other qubits.")
                    else:
                        st.success("Nearly pure reduced state (entropy ≈ 0) — not entangled with the rest.")

                # Bloch sphere plot
                try:
                    bloch_fig = plot_bloch_vector([rx, ry, rz])
                    st.pyplot(bloch_fig)
                except Exception as e:
                    # fallback: draw a simple arrow plot on a 2D circle projection
                    st.info("Could not draw Qiskit Bloch sphere; showing 2D projection.")
                    fig2, ax2 = plt.subplots(figsize=(3,3))
                    ax2.set_aspect('equal')
                    circle = plt.Circle((0,0), 1.0, fill=False)
                    ax2.add_artist(circle)
                    ax2.arrow(0, 0, rx, rz, head_width=0.06, length_includes_head=True)
                    ax2.set_xlim(-1.1,1.1); ax2.set_ylim(-1.1,1.1)
                    ax2.set_xlabel("X"); ax2.set_ylabel("Z")
                    ax2.set_title(f"2D X–Z projection (q{q})")
                    st.pyplot(fig2)

        st.divider()
        st.header("Entanglement hints & notes")
        st.write(
            "- If a single-qubit reduced state is **mixed** (non-zero entropy / purity < 1), "
            "that indicates correlations or entanglement with the rest of the system.\n"
            "- For fully mixed single-qubit (Bloch vector ≈ 0), look at measurement probabilities to understand global correlations."
        )
