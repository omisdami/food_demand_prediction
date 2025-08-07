
# 📘 Inventory Forecasting Dataset Schema

This dataset is structured to help build a machine learning model that forecasts inventory needs before each delivery (twice a week: Monday and Saturday). Each row in the dataset represents the **total inventory needed for a delivery period**, based on aggregated sales orders.

### Columns Explained:

- **delivery_date (Date)**: The actual delivery day (either Monday or Saturday). This is the date the inventory should arrive and be stocked.
- **wings (Integer)**: Total number of wings consumed in the delivery window (target to forecast).
- **tenders (Integer)**: Total chicken tenders used.
- **fries_reg / fries_large (Integer)**: Total portions of fries required.
- **veggies (Integer)**: Total servings of veggie sticks.
- **dips (Integer)**: Number of dip cups needed.
- **drinks (Integer)**: Number of fountain drinks served.
- **flavours (Integer)**: Number of sauce flavor servings used (e.g., Lemon Pepper, BBQ).

### 🔍 How to Use for Modeling:

1. **Goal**: Predict how much of each inventory type is needed for the next delivery date.
2. **Input Features**: You can engineer:
   - Lag features (e.g., wings_used_last_delivery)
   - Rolling averages (e.g., 2-week moving average of tenders)
   - Calendar features (day of week, season, event flags)
3. **Target Variables**: Each of the inventory columns (wings, tenders, dips, etc.)
4. **Model Type**: Multi-output regression (e.g., XGBoost, LightGBM, LSTM)

This dataset is ready to train models that can optimize what to order before each delivery — reducing waste and stockouts.
