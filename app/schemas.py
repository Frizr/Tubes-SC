from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


NormalAbnormal = Literal["normal", "abnormal"]
PresentNotPresent = Literal["present", "notpresent"]
YesNo = Literal["yes", "no"]
GoodPoor = Literal["good", "poor"]
PredictionLabel = Literal["ckd", "notckd"]


SAMPLE_PAYLOAD = {
    "age": 55,
    "blood_pressure": 80,
    "specific_gravity": 1.02,
    "albumin": 0,
    "sugar": 0,
    "red_blood_cells": "normal",
    "pus_cell": "normal",
    "pus_cell_clumps": "notpresent",
    "bacteria": "notpresent",
    "blood_glucose_random": 120,
    "blood_urea": 40,
    "serum_creatinine": 1.2,
    "sodium": 137,
    "potassium": 4.5,
    "hemoglobin": 14.5,
    "packed_cell_volume": 45,
    "white_blood_cell_count": 8000,
    "red_blood_cell_count": 5.2,
    "hypertension": "no",
    "diabetes_mellitus": "no",
    "coronary_artery_disease": "no",
    "appetite": "good",
    "pedal_edema": "no",
    "anemia": "no",
}


class ScreeningRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": SAMPLE_PAYLOAD})

    age: float = Field(..., ge=0, le=120)
    blood_pressure: float = Field(..., ge=30, le=220)
    specific_gravity: float = Field(..., ge=1.0, le=1.04)
    albumin: float = Field(..., ge=0, le=5)
    sugar: float = Field(..., ge=0, le=5)
    red_blood_cells: NormalAbnormal
    pus_cell: NormalAbnormal
    pus_cell_clumps: PresentNotPresent
    bacteria: PresentNotPresent
    blood_glucose_random: float = Field(..., ge=20, le=600)
    blood_urea: float = Field(..., ge=1, le=300)
    serum_creatinine: float = Field(..., ge=0.1, le=30)
    sodium: float = Field(..., ge=90, le=180)
    potassium: float = Field(..., ge=1, le=10)
    hemoglobin: float = Field(..., ge=3, le=25)
    packed_cell_volume: float = Field(..., ge=10, le=70)
    white_blood_cell_count: float = Field(..., ge=1000, le=50000)
    red_blood_cell_count: float = Field(..., ge=1, le=9)
    hypertension: YesNo
    diabetes_mellitus: YesNo
    coronary_artery_disease: YesNo
    appetite: GoodPoor
    pedal_edema: YesNo
    anemia: YesNo


class TopFeature(BaseModel):
    feature: str
    importance: float


class ScreeningResponse(BaseModel):
    prediction: PredictionLabel
    probability: float = Field(..., ge=0, le=1)
    top_features: list[TopFeature]
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ModelInfoResponse(BaseModel):
    model_name: str
    training_timestamp: str
    feature_count: int
    dataset_citation: str
    safety_note: str
