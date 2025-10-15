# 📈 ML_Project - Predicting S&P 500 Stock Prices

## 🧠 Description

This project investigates the application of three machine learning models—**XGBoost**, **LSTM**, and **Lasso Regression**—to predict future stock prices of the S&P 500 index. Leveraging historical market data and technical indicators (such as RSI and MACD), the models aim to identify both linear and non-linear patterns for time-series forecasting. The study evaluates predictive accuracy, interpretability, and computational performance across various market conditions.

This project was part of the final coursework for DS3000 at Western University and includes model implementation in Python, data preprocessing pipelines, and performance visualizations.

## 🔬 Findings

- **Lasso Regression**: 
  - High interpretability through sparse feature selection.
  - Struggled with sharp market changes.
  - Accuracy: 78.44% (within 0.5% margin), Directional Accuracy: 81.91%

- **LSTM Neural Network**: 
  - Strong at modeling trends, but less precise on exact value prediction.
  - Accuracy: 14.18%, Directional Accuracy: 50.98%
  - Required significant hyperparameter tuning.

- **XGBoost**: 
  - Best performance overall in capturing dynamic market patterns.
  - Accuracy: 57.79%, Directional Accuracy: 99.92%
  - Improved significantly with relative percentage features and tuning.

## 📄 Final Report

You can view the full project paper here:

[📘 Download DS3000_Final_Report.pdf](./DS3000_Final_Report.pdf)

## 🎥 Proposal Video

Watch our proposal video here:  
📽️ [https://youtu.be/iLkC5ClVUzk](https://youtu.be/iLkC5ClVUzk)

## 🎓 Final Research Presentation

Watch our final presentation video here:  
📽️ [https://youtu.be/11B1ts7Qdx8](https://youtu.be/11B1ts7Qdx8)

## 🔮 Future Work

- Integrate advanced **feature selection** using Shapley values or mutual information to reduce model noise.
- Expand the dataset by incorporating **external sources** such as news sentiment and macroeconomic indicators.
- Build a **user-facing visualization dashboard** to interpret model outputs for retail investors.
- Continue hyperparameter optimization for deep learning architectures.
- Explore real-time inference pipelines with streaming financial data.

## 👨‍💻 Contributors

- **Daniel Harapiak** (Computer Science)  
  - Preprocessing, brain storming, and report writing.

- **Leyang Xing** (Software Engineering)  
  - Feature engineering, Lasso and LSTM development, XGBoost implementation structure, proposal writing.

- **Kai On Ng** (Software Engineering)  
  - XGBoost tuning, model validation strategy, data transformation research.

- **Paul Gherghel** (Software Engineering)  
  - Visualization pipeline, final presentation video, model debugging and polishing.

---

