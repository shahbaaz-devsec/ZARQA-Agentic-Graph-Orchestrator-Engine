# ZARQA-Agentic-Graph-Orchestrator-Engine

[![DOI - Software (Latest)](https://img.shields.io/badge/Zenodo%20Software-10.5281%2Fzenodo.23908307-blue)](https://doi.org/10.5281/zenodo.23908307)
[![DOI - Phase I Paper](https://img.shields.io/badge/Zenodo%20Phase%20I%20Paper-10.5281%2Fzenodo.23908912-00557f)](https://doi.org/10.5281/zenodo.23908912)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> **A Defensible Single-Node Implementation of State-Streaming, Semantic Pruning, and Defence-in-Depth: Formal Mathematical Verification and Substrate Portability.**

---

## 📌 Overview

The **ZARQA-Agentic-Graph-Orchestrator-Engine (Phase 1)** implements a highly resilient, single-node Python daemon (`zarqa_semantic_state_pruning_core.py` v2.6.0) that stores per-session state slices in a durable SHA3-512 hash chain, prunes them under a fractional-knapsack capacity bound, and mediates access through Ed25519-signed capabilities. 

This repository enforces an immutable operational invariant: **No state is ingested, and no memory is allocated unless it passes a strict pre-accept TCP semaphore gate, a single monotonic body deadline, and a zero-trust capability verification layer.** 

---

## 🏛️ Core Mathematical & Defensive Guarantees

### Phase 1: Semantic State Pruning & Cyber-Physical Orchestration
1. **TCP Saturation & Load Shedding (FIX-Q/R/S/T2):** Implements a strict pre-accept semaphore gate, bounding user-space file descriptors to exactly $1 + W$ (where $W=64$ workers). This mathematically eliminates `EMFILE` spin and kernel-level memory exhaustion. A unified monotonic deadline strictly severs complete-body drip attacks. 
2. **Hypervisor Race Resolution (Theorem 7):** The TCP ingestion boundary elevates the worker-timeout budget to 15 seconds, mathematically annihilating the asymptotic accept latency anomaly observed under hypervisor scheduling jitter (WSL2).
3. **Hardware Acceleration (FIX-P):** Ensures cross-process propagation of Flush-To-Zero (FTZ) and Denormals-Are-Zero (DAZ) via `.init_array` `LD_PRELOAD` shims. This guarantees that subnormal floating-point arithmetic is flushed in 1 cycle, bypassing CPU microcode-assist latency traps.
4. **Fractional-Knapsack Pruning (Theorem 2):** A density-greedy selector bounds state resident memory to strictly defined limits ($C_i$). It ensures optimal semantic state retention without destructive delta overrides.
5. **PCA Dimensionality Canonicalization (Theorem 3):** Achieves Eckart-Young minimal Frobenius distortion via Truncated SVD, exactly isolating discarded spectral energy.
6. **Ed25519 Capability Soundness (Theorem 4):** Defends the loopback HTTP surface using a capability manager (FIPS 186-5 EUF-CMA). Enforces absolute wall-clock TTLs and strict spatial scoping for zero-trust state modifications.
7. **Topological DoS Protection:** Rejects deep-array payload attacks before AST construction via pre-lexical $O(N)$ JSON depth scanning (bounded at depth 64), executed at memory bandwidth via a GIL-released Rust FFI.

---

## 📊 Phase 1 Verification Evidence & Execution Logs

The following terminal logs capture the live production deployment and deterministic self-test execution of the ZARQA Semantic State Pruning Core (v2.6.0).

#### 1. Automated Deployment, Substrate Detection & Rust FFI Compilation
*Execution of `--auto-deploy` detecting hypervisor (WSL2) constraints, provisioning isolated environments, and successfully building the Rust `cdylib` cache[cite: 5].*  
![Auto Deploy Initialization](assets/zagoe_1.PNG)

#### 2. State Integrity Assurance (T1-T10)
*Execution of the deterministic self-test suite, proving Fractional-Knapsack bounds, Ed25519 capability signatures, Merkle-inclusion approximations, and Banach fixed-point convergence[cite: 14].*  
![State Integrity T1-T10](assets/zagoe_2.PNG)

#### 3. Hypervisor TCP Ingestion & I/O Isolation (T11-T22)
*Proof of the FIX-Q/R/S/T2 TCP saturation bounds, proving $p99 \le 15\text{ms}$ latency, $O(1)$ coordinated cache hits during Thundering Herd, and successful mitigation of Slowloris deadline attacks[cite: 13].*  
![State Integrity T11-T22](assets/zagoe_3.PNG)

#### 4. Hardware Acceleration & Cross-Process FTZ/DAZ Propagation (T23-T28)
*Validation of the FTZ/DAZ microcode-assist bypass, successful `LD_PRELOAD` shim injections into child processes, and deterministic recovery from the Write-Ahead Log (WAL)[cite: 12].*  
![State Integrity T23-T28](assets/zagoe_4.PNG)

#### 5. Engagement Governance & Defence-in-Depth (B1-B10, D1-D3)
*Testing Truthful Reporting, Authorized Engagement Governance bounds, Peace Preference matrices, and Layered Defense Analytics[cite: 11].*  
![Engagement & Defense B1-D3](assets/zagoe_5.PNG)

#### 6. Information-Theoretic Bounds (D4-D10, H1-H9)
*Validation of Shannon Entropy measurements, Bayesian Learning posteriors, and Schnorr ZK Soundness[cite: 10].*  
![Info-Theoretic Bounds D4-H9](assets/zagoe_6.PNG)

#### 7. Substrate Portability & Static Compiler Verification (H10, X1-X10, SCV1-SCV2)
*Roofline model limits, PALM scoring, and Polyhedral Legality matrix verification across simulated architectures[cite: 9].*  
![Portability & SCV](assets/zagoe_7.PNG)

#### 8. Derived Composite Metrics Finalization
*Computation of the final design-time composite guarantee ($G = 0.9000$, bounded by Semantic Retention) passing the strict `--min-quality 0.85` threshold[cite: 8].*  
![Composite Metrics](assets/zagoe_8.PNG)

#### 9. Immutable Release Deployment & Systemd Initialization
*Clean deployment of `zarqa-state` into `/opt/zarqa/current`, writing the `systemd` service unit, and successfully verifying the HTTP 200 OK `/healthz` endpoint[cite: 7].*  
![Systemd Deployment](assets/zagoe_9.PNG)

#### 10. Service Daemon Telemetry & Journalctl Logs
*Live `journalctl -u zarqa-state` output proving flawless startup, active worker pool concurrency, and pristine background memory thresholds without segmentation faults[cite: 6].*  
![Daemon Telemetry](assets/zagoe_10.PNG)

---

## 📂 Repository Structure

```text
ZARQA-Agentic-Graph-Orchestrator-Engine/
├── LICENSE
├── README.md
├── .gitignore
├── .zenodo.json                       # Automated Zenodo metadata citation schema
│
├── assets/
│   └── .gitkeep                       # Image references and execution logs
│
└── zarqa_semantic_state_pruning_core.py # Phase I runtime mathematical & pruning engine

```

---

## 🚀 Getting Started & Usage

### 1. Requirements & Prerequisites

* Linux OS (Ubuntu 22.04 / 24.04 LTS / WSL2)
* Python 3.10+
* `rustc` (for topological JSON depth-scan FFI)
* `numpy` & `cryptography` (auto-provisioned in venv)

### 2. Standard Pre-Flight Self-Tests (Single-Run Verification)

Execute the deterministic mathematical and algorithmic verification across all phases (T+B+D+H+X+SCV+NOVEL) without deploying background services:

```bash
sudo /opt/zarqa/zagoe/zarqa_semantic_state_pruning_core.py --test

```

### 3. One-Click Production Deployment (Root Required)

Provisions isolated virtual environments under `/opt/`, builds the Rust `cdylib`, writes the systemd daemon services, and enforces the FIX-P LD_PRELOAD drop-in for sibling daemons:

```bash
sudo /opt/zarqa/zagoe/zarqa_semantic_state_pruning_core.py --auto-deploy

```

*(Note: To deploy without touching `zarqa-core` configurations, use `--skip-core-shim`)*

### 4. Monitor System Health & Telemetry

```bash
# Check daemon status
sudo /opt/zarqa/zagoe/zarqa_semantic_state_pruning_core.py --status

# Probe FTZ/DAZ hardware effectiveness
sudo /opt/zarqa/zagoe/zarqa_semantic_state_pruning_core.py --enable-ftz

```

---

## 📖 Citation

If you use this codebase or mathematical architecture in your research, please cite our official Zenodo whitepaper and software repository:

```bibtex
@software{ahmed_zagoe_software_phase1_2026,
  author       = {Ahmed, Mohammad Shahbaaz},
  title        = {ZARQA-Agentic-Graph-Orchestrator-Engine: Phase 1 Core Software (v2.6.0)},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23908307},
  url          = {[https://doi.org/10.5281/zenodo.23908307](https://doi.org/10.5281/zenodo.23908307)}
}

@techreport{ahmed_zagoe_phase1_2026,
  author       = {Ahmed, Mohammad Shahbaaz},
  title        = {ZARQA v2.6.0: Mathematical Foundations of Semantic State Pruning and Symplectic Cyber-Physical Orchestration},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23908912},
  url          = {[https://doi.org/10.5281/zenodo.23908912](https://doi.org/10.5281/zenodo.23908912)}
}

```

---

## ⚖️ License & Disclaimer

This project is licensed under the **MIT License** - see the `LICENSE` file for details.

*Disclaimer: This codebase is a sovereign cyber-physical reference implementation designed for academic peer review, state-streaming optimization, and secure infrastructure verification. I explicitly disclaim implementation of Merkle-DAG inclusion proofs and post-quantum cryptography inside this specific single-node codebase.*
