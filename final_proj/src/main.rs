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

    // Calculate additional metrics
    let mae = test_targets.iter().zip(&predictions)
        .map(|(actual, pred)| (actual - pred).abs())
        .sum::<f64>() / test_targets.len() as f64;

    let mse = test_targets.iter().zip(&predictions)
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>() / test_targets.len() as f64;

    let mean_actual = test_targets.iter().copied().sum::<f64>() / test_targets.len() as f64;
    let ss_total = test_targets.iter()
        .map(|actual| (actual - mean_actual).powi(2))
        .sum::<f64>();

    let ss_residual = test_targets.iter().zip(&predictions)
        .map(|(actual, pred)| (actual - pred).powi(2))
        .sum::<f64>();

    let r_squared = 1.0 - (ss_residual / ss_total);

    let mape = test_targets.iter().zip(&predictions)
        .map(|(actual, pred)| ((actual - pred).abs() / actual.abs()).min(1.0)) // Avoid division by zero
        .sum::<f64>() / test_targets.len() as f64 * 100.0;

    // Print metrics
    println!("Root Mean Squared Error (Linear Regression): {:.4}", rmse);
    println!("Mean Absolute Error (MAE): {:.4}", mae);
    println!("Mean Squared Error (MSE): {:.4}", mse);
    println!("R-squared (R²): {:.4}", r_squared);
    println!("Mean Absolute Percentage Error (MAPE): {:.2}%", mape);

    // Plot results
    plot_predictions(&test_targets, &predictions)?;

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::data_processing::load_and_normalize_data;
    use super::model::{train_linear_model, evaluate_model};

    #[test]
    fn test_linear_model_training_no_ln() {
        let train_features = vec![
            vec![1.0, 2.0], 
            vec![2.0, 3.0], 
            vec![3.0, 4.0]
        ];
        let train_targets = vec![100.0, 200.0, 300.0]; // Raw price values
        let test_features = vec![
            vec![1.5, 2.5], 
            vec![2.5, 3.5]
        ];
        let test_targets = vec![150.0, 250.0]; // Raw price values

        let weights = train_linear_model(&train_features, &train_targets, 0.01, 100);
        let (rmse, predictions) = evaluate_model(&test_features, &test_targets, &weights);

        assert!(rmse < 50.0, "Prediction RMSE too high: {:.4}", rmse); // Adjust threshold as needed
        for (predicted, actual) in predictions.iter().zip(test_targets.iter()) {
            assert!(
                (predicted - actual).abs() < 50.0, 
                "Prediction error too high. Predicted: {:.2}, Actual: {:.2}",
                predicted,
                actual
            );
        }
    }
}
