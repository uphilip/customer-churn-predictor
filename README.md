# Customer Churn Predictor

## 3MTT AI/ML Capstone Project

**Developed by Chukwuka Uchenna**

---

## 1. Project Overview

Customer churn is a major challenge for organisations that depend on recurring customers and subscription-based services.

This project develops a machine learning-based Customer Churn Predictor that estimates the probability of a customer leaving a service based on customer demographic information, service usage, contract information and billing characteristics.

The project uses the Telco Customer Churn dataset and applies data preprocessing, exploratory data analysis, machine learning classification and model evaluation.

A Streamlit web application was developed to allow users to enter customer information and receive an estimated churn probability.

---

## 2. Problem Statement

Customer churn can negatively affect revenue and customer retention.

Traditional approaches may identify customers only after they have already discontinued a service.

This project addresses the problem by developing a predictive machine learning system capable of identifying customers who may have a higher probability of churn.

The system can therefore serve as a decision-support tool for customer retention analysis.

---

## 3. Project Objectives

The objectives of this project are:

1. To explore and preprocess customer churn data.
2. To develop machine learning classification models for predicting customer churn.
3. To evaluate and compare the performance of Logistic Regression and Random Forest models.
4. To develop an interactive application for generating customer churn predictions.

---

## 4. Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains:

- 7,043 customer records
- 21 variables
- Customer demographic information
- Service information
- Contract information
- Billing information
- Customer churn status

The target variable is:

`Churn`

where:

- `Yes` represents customer churn
- `No` represents customer retention

---

## 5. Machine Learning Workflow

The project follows the following machine learning workflow:

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Selection
     ↓
Data Preprocessing
     ↓
Train/Test Split
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Selection
     ↓
Model Saving
     ↓
Streamlit Deployment