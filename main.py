import click
import mlflow
import mlflow.sklearn
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt

# Metrics we will plot to evaluate
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def plot_real_vs_prediction(actual, pred):
  # Create a new figure
  fig = plt.figure()

  # Plot the real values and the predictions as two different lines
  plt.plot(actual, label='real')
  plt.plot(pred, label='predictions')

  # Set names and activate the legend
  plt.ylabel('Price')
  plt.xlabel('Sample')
  plt.title('Real vs Prediction')
  plt.legend(loc='lower right')

  # Save the figure to mlflow
  mlflow.log_figure(fig, 'my_figure.png')

  # Close the figure so it is not displayed in the output cell
  plt.close(fig)

@click.command()
@click.option("--n_estimators", default=100)
@click.option("--max_depth", default=3)
def main(n_estimators, max_depth):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
        path="boston_housing.npz", test_split=0.2, seed=113
    )

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model
    # Create the model using the parameters
    # Create the model using the parameters
    model = RandomForestRegressor(n_estimators=n_estimators,
                                      max_depth=max_depth)

    # Fit the model to the data
    model.fit(x_train, y_train)

    # Predict the test data
    y_pred = model.predict(x_test)

    # Check the metrics (real vs predicted)
    rmse_test, mae_test, r2_test = eval_metrics(y_test, y_pred)

    # Log the metrics to MLFlow
    mlflow.log_metric("rmse", rmse_test)
    mlflow.log_metric("mae", mae_test)
    mlflow.log_metric("r2", r2_test)

    # Create a figure with the pred vs actual and log it to mlflow
    plot_real_vs_prediction(y_test, y_pred)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    main()