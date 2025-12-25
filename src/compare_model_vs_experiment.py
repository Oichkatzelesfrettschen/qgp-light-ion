#!/usr/bin/env python3
"""
Compare model predictions against experimental data.

Analyzes discrepancies between:
1. R_AA predictions vs CMS O-O data
2. Flow predictions vs CMS/ALICE data
3. Ne-Ne/O-O ratios
"""

from pathlib import Path

import numpy as np

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
EXP_DIR = DATA_DIR / "experimental"
MODEL_JET_DIR = DATA_DIR / "jet_quenching"
MODEL_FLOW_DIR = DATA_DIR / "flow"


def load_data(filepath):
    """Load space-separated data, skipping comments."""
    data = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                data.append([float(x) for x in line.split()])
    return np.array(data)


def analyze_raa_oo():
    """Compare O-O R_AA model vs CMS data."""
    print("=" * 80)
    print("1. O-O R_AA ANALYSIS")
    print("=" * 80)

    # Load CMS experimental data
    cms_data = load_data(EXP_DIR / "CMS_OO_RAA_HIN25008.dat")
    pT_cms = cms_data[:, 2]  # pT_center
    raa_cms = cms_data[:, 3]  # R_AA
    stat_err_cms = cms_data[:, 4]
    sys_err_cms = cms_data[:, 5]

    # Load model prediction
    model_data = load_data(MODEL_JET_DIR / "RAA_OO.dat")
    pT_model = model_data[:, 0]
    raa_model = model_data[:, 1]

    # Find minimum values
    idx_cms_min = np.argmin(raa_cms)
    idx_model_min = np.argmin(raa_model)

    print("\nMINIMUM R_AA VALUES:")
    print(f"  CMS Data:  R_AA = {raa_cms[idx_cms_min]:.4f} at pT = {pT_cms[idx_cms_min]:.1f} GeV")
    print(
        f"  Model:     R_AA = {raa_model[idx_model_min]:.4f} at pT = {pT_model[idx_model_min]:.1f} GeV"
    )
    print(f"  Discrepancy: {abs(raa_cms[idx_cms_min] - raa_model[idx_model_min]):.4f}")

    # Interpolate model to CMS pT points for direct comparison
    raa_model_interp = np.interp(pT_cms, pT_model, raa_model)

    # Calculate chi-squared
    total_err_cms = np.sqrt(stat_err_cms**2 + sys_err_cms**2)
    chi2_per_point = ((raa_cms - raa_model_interp) / total_err_cms) ** 2
    chi2 = np.sum(chi2_per_point)
    ndof = len(pT_cms)

    print("\nFIT QUALITY:")
    print(f"  χ² = {chi2:.2f}")
    print(f"  ndof = {ndof}")
    print(f"  χ²/ndof = {chi2 / ndof:.2f}")

    # Key pT points comparison
    print("\nKEY pT POINTS COMPARISON:")
    key_pts = [4.0, 6.0, 8.0, 10.0, 20.0]
    for pt in key_pts:
        if pt >= pT_cms.min() and pt <= pT_cms.max():
            raa_exp = np.interp(pt, pT_cms, raa_cms)
            raa_mod = np.interp(pt, pT_model, raa_model)
            err = np.interp(pt, pT_cms, total_err_cms)
            sigma_diff = abs(raa_exp - raa_mod) / err
            print(
                f"  pT = {pt:4.1f} GeV:  CMS = {raa_exp:.3f}  Model = {raa_mod:.3f}  Δ/σ = {sigma_diff:.2f}"
            )

    # High-pT recovery check
    high_pt_cms = raa_cms[pT_cms > 20]
    high_pt_model_interp = raa_model_interp[pT_cms > 20]
    print("\nHIGH-pT RECOVERY (pT > 20 GeV):")
    print(f"  CMS mean R_AA:   {np.mean(high_pt_cms):.3f}")
    print(f"  Model mean R_AA: {np.mean(high_pt_model_interp):.3f}")

    return {
        "cms_min": raa_cms[idx_cms_min],
        "model_min": raa_model[idx_model_min],
        "chi2_per_ndof": chi2 / ndof,
    }


def analyze_raa_nene():
    """Compare Ne-Ne R_AA model predictions."""
    print("\n" + "=" * 80)
    print("2. Ne-Ne R_AA ANALYSIS")
    print("=" * 80)

    # Load model predictions
    oo_data = load_data(MODEL_JET_DIR / "RAA_OO.dat")
    nene_data = load_data(MODEL_JET_DIR / "RAA_NeNe.dat")

    pT = oo_data[:, 0]
    raa_oo = oo_data[:, 1]
    raa_nene = nene_data[:, 1]

    # Find minimum values
    idx_oo_min = np.argmin(raa_oo)
    idx_nene_min = np.argmin(raa_nene)

    print("\nMINIMUM R_AA VALUES:")
    print(f"  O-O:    R_AA = {raa_oo[idx_oo_min]:.4f} at pT = {pT[idx_oo_min]:.1f} GeV")
    print(f"  Ne-Ne:  R_AA = {raa_nene[idx_nene_min]:.4f} at pT = {pT[idx_nene_min]:.1f} GeV")
    print("\nNote: CMS reports Ne-Ne R_AA min ≈ 0.65 at pT ≈ 6 GeV")
    print(f"      Model prediction: {raa_nene[idx_nene_min]:.4f}")
    print(f"      Discrepancy: {abs(0.65 - raa_nene[idx_nene_min]):.4f}")

    return {
        "nene_min_model": raa_nene[idx_nene_min],
        "nene_min_expected": 0.65,
    }


def analyze_flow_oo():
    """Compare O-O flow predictions vs CMS data."""
    print("\n" + "=" * 80)
    print("3. O-O FLOW ANALYSIS")
    print("=" * 80)

    # Load CMS experimental data
    cms_data = load_data(EXP_DIR / "CMS_OO_flow_HIN25009.dat")
    cent_cms = cms_data[:, 2]  # cent_mean
    v2_cms = cms_data[:, 3]
    v2_stat_cms = cms_data[:, 4]
    v2_sys_cms = cms_data[:, 5]
    v3_cms = cms_data[:, 6]
    v3_stat_cms = cms_data[:, 7]
    v3_sys_cms = cms_data[:, 8]

    # Load model prediction
    model_data = load_data(MODEL_FLOW_DIR / "vn_vs_cent_O.dat")
    cent_model = model_data[:, 0]
    v2_model = model_data[:, 1]
    v3_model = model_data[:, 2]

    # Interpolate model to CMS centrality points
    v2_model_interp = np.interp(cent_cms, cent_model, v2_model)
    v3_model_interp = np.interp(cent_cms, cent_model, v3_model)

    # Calculate chi-squared for v2
    v2_total_err = np.sqrt(v2_stat_cms**2 + v2_sys_cms**2)
    chi2_v2 = np.sum(((v2_cms - v2_model_interp) / v2_total_err) ** 2)

    # Calculate chi-squared for v3
    v3_total_err = np.sqrt(v3_stat_cms**2 + v3_sys_cms**2)
    chi2_v3 = np.sum(((v3_cms - v3_model_interp) / v3_total_err) ** 2)

    ndof_v2 = len(cent_cms)
    ndof_v3 = len(cent_cms)

    print("\nv2 FIT QUALITY:")
    print(f"  χ²/ndof = {chi2_v2 / ndof_v2:.2f}")

    print("\nv3 FIT QUALITY:")
    print(f"  χ²/ndof = {chi2_v3 / ndof_v3:.2f}")

    # Key centrality points
    print("\nKEY CENTRALITY POINTS (v2):")
    key_cents = [0.5, 7.5, 22.5, 37.5]
    for cent in key_cents:
        if cent >= cent_cms.min() and cent <= cent_cms.max():
            v2_exp = np.interp(cent, cent_cms, v2_cms)
            v2_mod = np.interp(cent, cent_model, v2_model)
            err = np.interp(cent, cent_cms, v2_total_err)
            sigma_diff = abs(v2_exp - v2_mod) / err
            print(
                f"  Centrality {cent:5.1f}%:  CMS = {v2_exp:.5f}  Model = {v2_mod:.5f}  Δ/σ = {sigma_diff:.2f}"
            )

    print("\nKEY CENTRALITY POINTS (v3):")
    for cent in key_cents:
        if cent >= cent_cms.min() and cent <= cent_cms.max():
            v3_exp = np.interp(cent, cent_cms, v3_cms)
            v3_mod = np.interp(cent, cent_model, v3_model)
            err = np.interp(cent, cent_cms, v3_total_err)
            sigma_diff = abs(v3_exp - v3_mod) / err
            print(
                f"  Centrality {cent:5.1f}%:  CMS = {v3_exp:.5f}  Model = {v3_mod:.5f}  Δ/σ = {sigma_diff:.2f}"
            )

    # Peak v2 analysis
    idx_cms_peak = np.argmax(v2_cms)
    idx_model_peak = np.argmax(v2_model)

    print("\nPEAK v2:")
    print(f"  CMS:   v2 = {v2_cms[idx_cms_peak]:.5f} at centrality = {cent_cms[idx_cms_peak]:.1f}%")
    print(
        f"  Model: v2 = {v2_model[idx_model_peak]:.5f} at centrality = {cent_model[idx_model_peak]:.1f}%"
    )

    return {
        "v2_chi2_per_ndof": chi2_v2 / ndof_v2,
        "v3_chi2_per_ndof": chi2_v3 / ndof_v3,
        "v2_peak_cms": v2_cms[idx_cms_peak],
        "v2_peak_model": v2_model[idx_model_peak],
    }


def analyze_flow_nene():
    """Compare Ne-Ne flow predictions vs CMS data."""
    print("\n" + "=" * 80)
    print("4. Ne-Ne FLOW ANALYSIS")
    print("=" * 80)

    # Load CMS experimental data
    cms_data = load_data(EXP_DIR / "CMS_NeNe_flow_HIN25009.dat")
    cent_cms = cms_data[:, 2]  # cent_mean
    v2_cms = cms_data[:, 3]
    v2_stat_cms = cms_data[:, 4]
    v2_sys_cms = cms_data[:, 5]
    _v3_cms = cms_data[:, 6]  # Available for future v3 analysis

    # Load model prediction
    model_data = load_data(MODEL_FLOW_DIR / "vn_vs_cent_Ne.dat")
    cent_model = model_data[:, 0]
    v2_model = model_data[:, 1]
    # v3_model available in column 2 if needed

    # Interpolate model to CMS centrality points
    v2_model_interp = np.interp(cent_cms, cent_model, v2_model)

    # Calculate chi-squared for v2
    v2_total_err = np.sqrt(v2_stat_cms**2 + v2_sys_cms**2)
    chi2_v2 = np.sum(((v2_cms - v2_model_interp) / v2_total_err) ** 2)
    ndof_v2 = len(cent_cms)

    print("\nv2 FIT QUALITY:")
    print(f"  χ²/ndof = {chi2_v2 / ndof_v2:.2f}")

    # Key centrality points
    print("\nKEY CENTRALITY POINTS (v2):")
    key_cents = [0.5, 7.5, 22.5, 37.5]
    for cent in key_cents:
        if cent >= cent_cms.min() and cent <= cent_cms.max():
            v2_exp = np.interp(cent, cent_cms, v2_cms)
            v2_mod = np.interp(cent, cent_model, v2_model)
            err = np.interp(cent, cent_cms, v2_total_err)
            sigma_diff = abs(v2_exp - v2_mod) / err
            print(
                f"  Centrality {cent:5.1f}%:  CMS = {v2_exp:.5f}  Model = {v2_mod:.5f}  Δ/σ = {sigma_diff:.2f}"
            )

    return {
        "v2_chi2_per_ndof": chi2_v2 / ndof_v2,
    }


def analyze_nene_oo_ratios():
    """Analyze Ne-Ne/O-O ratios vs ALICE data."""
    print("\n" + "=" * 80)
    print("5. Ne-Ne/O-O RATIO ANALYSIS")
    print("=" * 80)

    # Load ALICE qualitative data
    alice_data = load_data(EXP_DIR / "ALICE_OO_NeNe_flow_2509.06428.dat")
    cent_alice = alice_data[:, 0]
    v2_ratio_alice = alice_data[:, 1]
    v2_ratio_err_alice = alice_data[:, 2]
    v3_ratio_alice = alice_data[:, 3]

    # Load model predictions
    oo_data = load_data(MODEL_FLOW_DIR / "vn_vs_cent_O.dat")
    nene_data = load_data(MODEL_FLOW_DIR / "vn_vs_cent_Ne.dat")

    cent_model = oo_data[:, 0]
    v2_oo = oo_data[:, 1]
    v3_oo = oo_data[:, 2]
    v2_nene = nene_data[:, 1]
    v3_nene = nene_data[:, 2]

    # Calculate model ratios (avoid division by zero)
    v2_ratio_model = np.where(v2_oo > 1e-6, v2_nene / v2_oo, 1.0)
    v3_ratio_model = np.where(v3_oo > 1e-6, v3_nene / v3_oo, 1.0)

    # Interpolate to ALICE centrality points
    v2_ratio_model_interp = np.interp(cent_alice, cent_model, v2_ratio_model)
    v3_ratio_model_interp = np.interp(cent_alice, cent_model, v3_ratio_model)

    # Calculate chi-squared
    chi2_v2_ratio = np.sum(((v2_ratio_alice - v2_ratio_model_interp) / v2_ratio_err_alice) ** 2)
    ndof = len(cent_alice)

    print("\nv2 RATIO FIT QUALITY:")
    print(f"  χ²/ndof = {chi2_v2_ratio / ndof:.2f}")

    print("\nKEY CENTRALITY POINTS (v2 ratio):")
    for i, cent in enumerate(cent_alice):
        alice_val = v2_ratio_alice[i]
        model_val = v2_ratio_model_interp[i]
        err = v2_ratio_err_alice[i]
        sigma_diff = abs(alice_val - model_val) / err
        print(
            f"  Centrality {cent:5.1f}%:  ALICE = {alice_val:.3f}  Model = {model_val:.3f}  Δ/σ = {sigma_diff:.2f}"
        )

    print("\nKEY CENTRALITY POINTS (v3 ratio):")
    for i, cent in enumerate(cent_alice):
        alice_val = v3_ratio_alice[i]
        model_val = v3_ratio_model_interp[i]
        print(f"  Centrality {cent:5.1f}%:  ALICE = {alice_val:.3f}  Model = {model_val:.3f}")

    # Ultracentral ratio
    print("\nULTRACENTRAL (0% centrality):")
    print(f"  ALICE v2 ratio: {v2_ratio_alice[0]:.3f} ± {v2_ratio_err_alice[0]:.3f}")
    print(f"  Model v2 ratio: {v2_ratio_model_interp[0]:.3f}")

    return {
        "v2_ratio_chi2_per_ndof": chi2_v2_ratio / ndof,
    }


def generate_recommendations():
    """Generate parameter adjustment recommendations."""
    print("\n" + "=" * 80)
    print("PARAMETER ADJUSTMENT RECOMMENDATIONS")
    print("=" * 80)

    print("\n1. R_AA SUPPRESSION DEPTH (qgp_physics.py lines 540-546):")
    print("   Current O-O suppression_max: 0.31 (gives R_AA_min = 0.69)")
    print("   Current Ne-Ne suppression_max: 0.38 (gives R_AA_min ≈ 0.64)")
    print("   ")
    print("   ADJUSTMENTS:")
    print("   - O-O: Keep suppression_max = 0.31 (good match)")
    print("   - Ne-Ne: Change suppression_max from 0.38 to 0.35 to match R_AA_min ≈ 0.65")

    print("\n2. FLOW RESPONSE COEFFICIENTS (qgp_physics.py lines 404-411):")
    print("   Current kappa values: {2: 0.25, 3: 0.15, 4: 0.10, 5: 0.05}")
    print("   ")
    print("   ISSUE: Model predicts too low v2 in central collisions")
    print("   - CMS O-O ultracentral v2 ≈ 0.061, model predicts ≈ 0.00")
    print("   - CMS Ne-Ne ultracentral v2 ≈ 0.067, model predicts ≈ 0.022")
    print("   ")
    print("   ADJUSTMENTS:")
    print("   - Increase base v2 response: kappa[2] from 0.25 to 0.30")
    print("   - Add constant term for central collisions (nuclear deformation)")

    print("\n3. ECCENTRICITY MODEL (qgp_physics.py lines 427-443):")
    print("   Current epsilon_2 formula: 0.5 * sin(π*cent/100) * (1 - 0.3*cent/100)")
    print("   ")
    print("   ISSUE: epsilon_2 goes to 0 at cent=0, but CMS sees significant v2")
    print("   ")
    print("   ADJUSTMENTS:")
    print("   - Add baseline eccentricity for central collisions:")
    print("     epsilon_2_base = 0.15 for O-O (alpha clustering)")
    print("     epsilon_2_base = 0.25 for Ne-Ne (prolate deformation)")
    print("   - Modified formula:")
    print("     epsilon_2 = epsilon_2_base + 0.5 * sin(π*cent/100) * (1 - 0.3*cent/100)")

    print("\n4. NUCLEAR DEFORMATION (qgp_physics.py lines 53, 441-442):")
    print("   Current Ne beta2: 0.45")
    print("   Current deformation boost factor: 0.3")
    print("   ")
    print("   ISSUE: Ne-Ne/O-O v2 ratio in model vs ALICE")
    print("   - ALICE ultracentral: ~1.08")
    print("   - Model needs stronger deformation effect in central collisions")
    print("   ")
    print("   ADJUSTMENTS:")
    print("   - Keep beta2 = 0.45 (consistent with literature)")
    print("   - Increase deformation boost from 0.3 to 0.5 in central collisions")

    print("\n5. VISCOSITY DAMPING (qgp_physics.py lines 406-409):")
    print("   Current: damping = exp(-n * knudsen * 5)")
    print("   ")
    print("   ISSUE: May be over-damping v2 in small systems")
    print("   ")
    print("   ADJUSTMENTS:")
    print("   - Reduce damping factor from 5 to 3 for light ions")
    print("   - Better match small-system hydrodynamic response")


if __name__ == "__main__":
    print("QGP Light-Ion: Model vs Experiment Comparison")
    print("=" * 80)

    results = {}

    # Run all analyses
    results["raa_oo"] = analyze_raa_oo()
    results["raa_nene"] = analyze_raa_nene()
    results["flow_oo"] = analyze_flow_oo()
    results["flow_nene"] = analyze_flow_nene()
    results["ratios"] = analyze_nene_oo_ratios()

    # Generate recommendations
    generate_recommendations()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
