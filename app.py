
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Provider Fraud Detection", layout="wide")

st.title("Healthcare Provider Fraud Detection App")
st.write("Upload all 4 unseen/raw CSV files. The app will create features and predict fraud.")

model = joblib.load("provider_fraud_optimized_artifacts/best_provider_fraud_model.pkl")
feature_columns = joblib.load("provider_fraud_optimized_artifacts/final_feature_columns.pkl")
threshold = joblib.load("provider_fraud_optimized_artifacts/best_threshold.pkl")

def create_unseen_provider_features(inpatient, outpatient, beneficiary, provider_file):
    inpatient = inpatient.copy()
    outpatient = outpatient.copy()
    beneficiary = beneficiary.copy()
    provider_file = provider_file.copy()

    for df in [inpatient, outpatient]:
        df["ClaimStartDt"] = pd.to_datetime(df["ClaimStartDt"], errors="coerce")
        df["ClaimEndDt"] = pd.to_datetime(df["ClaimEndDt"], errors="coerce")

    inpatient["AdmissionDt"] = pd.to_datetime(inpatient["AdmissionDt"], errors="coerce")
    inpatient["DischargeDt"] = pd.to_datetime(inpatient["DischargeDt"], errors="coerce")

    beneficiary["DOB"] = pd.to_datetime(beneficiary["DOB"], errors="coerce")
    beneficiary["DOD"] = pd.to_datetime(beneficiary["DOD"], errors="coerce")

    inpatient["PatientCategory"] = "Inpatient"
    outpatient["PatientCategory"] = "Outpatient"

    inpatient["NoOfDaysAdmitted"] = (
        inpatient["DischargeDt"] - inpatient["AdmissionDt"]
    ).dt.days + 1
    inpatient["NoOfDaysAdmitted"] = inpatient["NoOfDaysAdmitted"].clip(lower=0)

    outpatient["NoOfDaysAdmitted"] = 0

    inpatient["ClaimDurationDays"] = (
        inpatient["ClaimEndDt"] - inpatient["ClaimStartDt"]
    ).dt.days + 1

    outpatient["ClaimDurationDays"] = (
        outpatient["ClaimEndDt"] - outpatient["ClaimStartDt"]
    ).dt.days + 1

    beneficiary["IsDead"] = beneficiary["DOD"].notna().astype(int)

    chronic_cols = [c for c in beneficiary.columns if c.startswith("ChronicCond_")]

    for c in chronic_cols:
        beneficiary[c] = beneficiary[c].replace({2: 0}).fillna(0).astype(int)

    beneficiary["ChronicConditionCount"] = beneficiary[chronic_cols].sum(axis=1)

    beneficiary["RenalDiseaseIndicator"] = (
        beneficiary["RenalDiseaseIndicator"]
        .replace({"Y": 1, "0": 0, 0: 0, 1: 1})
        .fillna(0)
        .astype(int)
    )

    beneficiary_selected_cols = [
        "BeneID",
        "DOB",
        "DOD",
        "Gender",
        "Race",
        "RenalDiseaseIndicator",
        "State",
        "County",
        "NoOfMonths_PartACov",
        "NoOfMonths_PartBCov",
        "ChronicConditionCount",
        "IsDead",
    ]

    beneficiary_selected = beneficiary[beneficiary_selected_cols].copy()

    all_claim_cols = sorted(set(inpatient.columns).union(set(outpatient.columns)))

    inp_aligned = inpatient.reindex(columns=all_claim_cols)
    out_aligned = outpatient.reindex(columns=all_claim_cols)

    claims = pd.concat([inp_aligned, out_aligned], ignore_index=True)

    claims = claims.merge(beneficiary_selected, on="BeneID", how="left")

    claims["AgeAtClaim"] = (
        (claims["ClaimStartDt"] - claims["DOB"]).dt.days / 365.25
    ).clip(lower=0)

    for col in [
        "InscClaimAmtReimbursed",
        "DeductibleAmtPaid",
        "NoOfDaysAdmitted",
        "ClaimDurationDays",
    ]:
        if col in claims.columns:
            claims[col] = claims[col].fillna(0)

    diag_cols = [
        c for c in claims.columns
        if "ClmDiagnosisCode" in c or c == "ClmAdmitDiagnosisCode"
    ]

    proc_cols = [
        c for c in claims.columns
        if "ClmProcedureCode" in c
    ]

    phys_cols = [
        "AttendingPhysician",
        "OperatingPhysician",
        "OtherPhysician",
    ]

    claims["IsInpatient"] = (claims["PatientCategory"] == "Inpatient").astype(int)
    claims["IsOutpatient"] = (claims["PatientCategory"] == "Outpatient").astype(int)
    claims["HasDeductible"] = (claims["DeductibleAmtPaid"] > 0).astype(int)

    claims["HighReimbursementClaim"] = (
        claims["InscClaimAmtReimbursed"]
        > claims["InscClaimAmtReimbursed"].quantile(0.95)
    ).astype(int)

    claims["LongAdmissionClaim"] = (
        claims["NoOfDaysAdmitted"]
        > claims["NoOfDaysAdmitted"].quantile(0.95)
    ).astype(int)

    claims["DiagnosisCodeCountPerClaim"] = (
        claims[diag_cols].notna().sum(axis=1) if len(diag_cols) > 0 else 0
    )

    claims["ProcedureCodeCountPerClaim"] = (
        claims[proc_cols].notna().sum(axis=1) if len(proc_cols) > 0 else 0
    )

    available_phys_cols = [c for c in phys_cols if c in claims.columns]

    claims["PhysicianCountPerClaim"] = (
        claims[available_phys_cols].notna().sum(axis=1)
        if len(available_phys_cols) > 0 else 0
    )

    provider_features = claims.groupby("Provider").agg(
        TotalClaims=("ClaimID", "count"),
        UniquePatients=("BeneID", "nunique"),
        InpatientClaimCount=("IsInpatient", "sum"),
        OutpatientClaimCount=("IsOutpatient", "sum"),

        TotalReimbursement=("InscClaimAmtReimbursed", "sum"),
        AvgReimbursement=("InscClaimAmtReimbursed", "mean"),
        MaxReimbursement=("InscClaimAmtReimbursed", "max"),
        MedianReimbursement=("InscClaimAmtReimbursed", "median"),

        TotalDeductible=("DeductibleAmtPaid", "sum"),
        AvgDeductible=("DeductibleAmtPaid", "mean"),
        DeductibleClaimRatio=("HasDeductible", "mean"),

        AvgClaimDuration=("ClaimDurationDays", "mean"),
        MaxClaimDuration=("ClaimDurationDays", "max"),
        AvgAdmissionDays=("NoOfDaysAdmitted", "mean"),
        MaxAdmissionDays=("NoOfDaysAdmitted", "max"),
        LongAdmissionRatio=("LongAdmissionClaim", "mean"),

        AvgDiagnosisCodesPerClaim=("DiagnosisCodeCountPerClaim", "mean"),
        AvgProcedureCodesPerClaim=("ProcedureCodeCountPerClaim", "mean"),
        AvgPhysicianCountPerClaim=("PhysicianCountPerClaim", "mean"),

        AvgPatientAge=("AgeAtClaim", "mean"),
        AvgChronicConditionCount=("ChronicConditionCount", "mean"),
        RenalDiseasePatientRatio=("RenalDiseaseIndicator", "mean"),
        DeadPatientRatio=("IsDead", "mean"),
        MaleRatio=("Gender", lambda x: (x == 1).mean()),
        UniqueStates=("State", "nunique"),
        UniqueCounties=("County", "nunique"),

        HighReimbursementClaimRatio=("HighReimbursementClaim", "mean"),
    ).reset_index()

    provider_features["ClaimsPerPatient"] = (
        provider_features["TotalClaims"]
        / provider_features["UniquePatients"].replace(0, np.nan)
    )

    provider_features["InpatientRatio"] = (
        provider_features["InpatientClaimCount"]
        / provider_features["TotalClaims"].replace(0, np.nan)
    )

    provider_features["OutpatientRatio"] = (
        provider_features["OutpatientClaimCount"]
        / provider_features["TotalClaims"].replace(0, np.nan)
    )

    provider_features["ReimbursementPerPatient"] = (
        provider_features["TotalReimbursement"]
        / provider_features["UniquePatients"].replace(0, np.nan)
    )

    provider_features["DeductiblePerPatient"] = (
        provider_features["TotalDeductible"]
        / provider_features["UniquePatients"].replace(0, np.nan)
    )

    provider_features["RepeatedPatientRatio"] = (
        1 - (
            provider_features["UniquePatients"]
            / provider_features["TotalClaims"].replace(0, np.nan)
        )
    )

    provider_features["ReimbursementToDeductibleRatio"] = (
        provider_features["TotalReimbursement"]
        / (provider_features["TotalDeductible"] + 1)
    )

    provider_features["AdmissionDaysPerClaim"] = (
        provider_features["AvgAdmissionDays"]
        / (provider_features["AvgClaimDuration"] + 1)
    )

    def unique_values_across_columns(group, columns):
        if not columns:
            return 0
        vals = pd.unique(group[columns].values.ravel())
        vals = [v for v in vals if pd.notna(v)]
        return len(vals)

    diversity_rows = []

    for provider, grp in claims.groupby("Provider"):
        diversity_rows.append({
            "Provider": provider,
            "UniqueDiagnosisCodes": unique_values_across_columns(grp, diag_cols),
            "UniqueProcedureCodes": unique_values_across_columns(grp, proc_cols),
            "UniquePhysicians": unique_values_across_columns(grp, available_phys_cols),
        })

    diversity_df = pd.DataFrame(diversity_rows)

    provider_features = provider_features.merge(
        diversity_df,
        on="Provider",
        how="left"
    )

    provider_features["DiagnosisDiversityPerClaim"] = (
        provider_features["UniqueDiagnosisCodes"]
        / provider_features["TotalClaims"].replace(0, np.nan)
    )

    provider_features["ProcedureDiversityPerClaim"] = (
        provider_features["UniqueProcedureCodes"]
        / provider_features["TotalClaims"].replace(0, np.nan)
    )

    provider_features["PhysicianDiversityPerClaim"] = (
        provider_features["UniquePhysicians"]
        / provider_features["TotalClaims"].replace(0, np.nan)
    )

    provider_features = provider_file[["Provider"]].merge(
        provider_features,
        on="Provider",
        how="left"
    )

    provider_features = provider_features.replace([np.inf, -np.inf], np.nan)
    provider_features = provider_features.fillna(0)

    return provider_features


provider_file = st.file_uploader("Upload Provider File", type=["csv"])
inpatient_file = st.file_uploader("Upload Inpatient Claims File", type=["csv"])
outpatient_file = st.file_uploader("Upload Outpatient Claims File", type=["csv"])
beneficiary_file = st.file_uploader("Upload Beneficiary File", type=["csv"])

if provider_file and inpatient_file and outpatient_file and beneficiary_file:

    provider_df = pd.read_csv(provider_file)
    inpatient_df = pd.read_csv(inpatient_file)
    outpatient_df = pd.read_csv(outpatient_file)
    beneficiary_df = pd.read_csv(beneficiary_file)

    st.success("All 4 files uploaded successfully.")

    with st.spinner("Creating provider-level features and predicting fraud..."):

        provider_features = create_unseen_provider_features(
            inpatient_df,
            outpatient_df,
            beneficiary_df,
            provider_df
        )

        provider_ids = provider_features["Provider"].copy()

        X_unseen = provider_features.drop(columns=["Provider"])

        X_unseen = X_unseen.reindex(
            columns=feature_columns,
            fill_value=0
        )

        probabilities = model.predict_proba(X_unseen)[:, 1]

        predictions = (probabilities >= threshold).astype(int)

        labels = np.where(predictions == 1, "Yes", "No")

        result_df = pd.DataFrame({
            "Provider": provider_ids,
            "Probability": probabilities,
            "PotentialFraud": labels
        })

    st.subheader("Prediction Summary")
    st.write(result_df["PotentialFraud"].value_counts())

    st.subheader("Prediction Results")
    st.dataframe(result_df)

    csv = result_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction CSV",
        data=csv,
        file_name="Provider_Fraud_Predictions.csv",
        mime="text/csv"
    )
