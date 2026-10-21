# Tarea 1 — Regresión lineal · Energy Efficiency (UCI 242)

Regresión de **Y1 = Heating Load** con X1..X8. Estilo del profe (`material de la clase`: EDA que decide + Normal manual con `for`-MSE).

## Entregable

- `src/Tareas/Tarea_1/Problema_Regresion_DataEnergy_EJECUTADO.ipynb` — notebook corrido (61 celdas, 0 errores). Es lo único versionado además de este README.
- Para regenerarlo en Colab súbelo + `gd_convergencia.gif` al lado (el notebook lo regenera solo al correr).

## Reproducir local con uv

```bash
uv pip install --system numpy pandas matplotlib seaborn scipy scikit-learn ucimlrepo nbformat nbconvert ipykernel pillow imageio
python -m jupyter nbconvert --to notebook --execute "src/Tareas/Tarea_1/Problema_Regresión_DataEnergy.ipynb" --output-dir "src/Tareas/Tarea_1" --output "Problema_Regresion_DataEnergy_EJECUTADO.ipynb" --ExecutePreprocessor.timeout=600 --allow-errors
```

Dataset: `ucimlrepo id=242` (fallback CSV UCI). Split 80/20 `random_state=42`, `StandardScaler` fit solo en train.

## Hallazgo EDA

0 nulos, 0% outliers IQR en X (simulado discreto), escalas 0.7 vs 671, VIF inf en X2/X3/X4 (`X2≈X3+X4`, `cond(X'X)=9.9e16`), corr Y1: X5 0.89, X4 −0.86, X1 0.62, X2 −0.66, X3 0.45, X7 0.27, X8 0.09, X6 ~0.

## Resultados test

| Modelo | R2 | RMSE |
|---|---|---|
| Normal manual (`pinv`) | 0.9122 | 3.03 |
| sklearn Linear | 0.9122 | 3.03 |
| Ridge a=1 | 0.9116 | 3.04 |
| GD propio (lr=0.3, 2000it) | 0.9121 | 3.03 |
| Lasso a=1 | 0.8808 | 3.53 |

Sin overfitting (gap train–test <0.02). Lasso apaga X6/X8 (confirma EDA). Lineal < RandomForest del paper (MAE HL 0.51) por no-linealidad.

## Respuestas (resumen, detalle al final del notebook)

1. **Overfitting:** posible en lineal (p grande/colinealidad); aquí no.
2. **Outliers:** no hay reales (boxplot + IQR + z>3, 0%).
3. **Eliminarlos:** no; winsorizar/Robust+Ridge si los hubiera.
4. **Escalado:** `(x−μ)/σ` etc.; obligatorio para GD/Lasso/Ridge/KNN, innecesario para Normal cerrada y árboles.
5. **GD vs Normal:** Normal exacta O(p³) para p pequeña; GD O(mpT) para p/m grandes, streaming y pérdidas no-cuadráticas.

## Referencias

- Tsanas & Xifara (2012) Energy & Buildings vol.49 (`6453-English-IranArze.pdf`, local no versionado).
- UCI: https://archive.ics.uci.edu/dataset/242/energy+efficiency
- `papers/` local (no versionado): arxiv-2108.11872 (GD vs Ridge), arxiv-2006.00371 (Ridge esencial), CMU linear_regression.pdf.
