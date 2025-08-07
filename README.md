# 📦 Inventory Forecasting for Restaurant Supply Optimization

This project aims to develop a machine learning system that forecasts inventory needs for a restaurant, optimizing purchase orders and reducing waste. The dataset is based on aggregated sales orders prior to each delivery window, which occurs **twice a week: Monday and Saturday**.

---

## 🧠 Project Objective

The main goal is to **predict how much of each inventory item will be needed before the next scheduled delivery**, using historical data and advanced machine learning techniques. This enables restaurants to:

- Prevent overstocking or understocking
- Automate ordering decisions
- Improve operational efficiency

---

## 🗃️ Dataset Overview

Each row in the dataset corresponds to a **delivery window**, capturing total demand for that period.  
Below are the key columns:

| Column          | Description                                       |
| --------------- | ------------------------------------------------- |
| `delivery_date` | Date of inventory delivery (Monday/Saturday only) |
| `wings`         | Number of wings consumed (target variable)        |
| `tenders`       | Chicken tenders consumed                          |
| `fries_reg`     | Regular fries servings                            |
| `fries_large`   | Large fries servings                              |
| `veggies`       | Veggie stick servings                             |
| `dips`          | Dip cups used                                     |
| `drinks`        | Fountain drinks served                            |
| `flavours`      | Sauce flavor servings                             |
