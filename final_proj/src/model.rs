pub fn train_linear_model(features: &[Vec<f64>], targets: &[f64], learning_rate: f64, epochs: usize) -> Vec<f64> {
    let mut weights = vec![0.0; features[0].len()];

    for _ in 0..epochs {
        let mut gradients = vec![0.0; weights.len()];

        for (x, &y) in features.iter().zip(targets.iter()) {
            let prediction: f64 = x.iter().zip(weights.iter()).map(|(&xi, &wi)| xi * wi).sum();
            let error = prediction - y;

            for j in 0..weights.len() {
                gradients[j] += error * x[j];
            }
        }

        for j in 0..weights.len() {
            weights[j] -= learning_rate * gradients[j] / features.len() as f64;
        }
    }

    weights
}

pub fn evaluate_model(features: &[Vec<f64>], targets: &[f64], weights: &[f64]) -> (f64, Vec<f64>) {
    let predictions: Vec<f64> = features.iter().map(|x| x.iter().zip(weights.iter()).map(|(&xi, &wi)| xi * wi).sum()).collect();
    let rmse = (targets.iter().zip(predictions.iter())
        .map(|(&y, &y_pred)| (y - y_pred).powi(2))
        .sum::<f64>() / targets.len() as f64)
        .sqrt();

    (rmse, predictions)
}
