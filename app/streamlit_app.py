import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "models/best_tuned_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)

model = joblib.load(...)
scaler = joblib.load(...)