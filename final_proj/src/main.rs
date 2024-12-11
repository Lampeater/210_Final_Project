mod data_processing;
mod model;
mod visualization;

use std::error::Error;
use data_processing::{load_and_normalize_data};
use model::{train_linear_model, evaluate_model};
use visualization::plot_predictions;

fn main() -> Result<(), Box<dyn Error>> {
    // Load and preprocess the dataset
    let (train_features, train_targets, test_features, test_targets) = load_and_normalize_data("NY-House-Dataset.csv")?;

    // Train the linear regression model
    let weights = train_linear_model(&train_features, &train_targets, 0.01, 1000);

    // Evaluate the model
    let (rmse, predictions) = evaluate_model(&test_features, &test_targets, &weights);
    println!("Root Mean Squared Error (Linear Regression): {:.4}", rmse);

    // Plot results
    plot_predictions(&test_targets, &predictions)?;

    Ok(())
}
