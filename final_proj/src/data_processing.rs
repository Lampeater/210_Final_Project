use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn load_and_normalize_data(file_path: &str) -> Result<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>), Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 { continue; }
        let columns: Vec<&str> = line.split(',').collect();

        let beds: f64 = columns.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let bath: f64 = columns.get(4).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let sqft: f64 = columns.get(5).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let latitude: f64 = columns.get(15).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let longitude: f64 = columns.get(16).and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let price: f64 = columns.get(2).and_then(|v| v.parse().ok()).unwrap_or(1.0);

        if beds + bath + sqft + latitude + longitude > 0.0 && price > 0.0 {
            features.push(vec![beds, bath, sqft, latitude, longitude]);
            targets.push(price.ln());
        }
    }

    let mut feature_means = vec![0.0; features[0].len()];
    let mut feature_stds = vec![0.0; features[0].len()];

    for j in 0..features[0].len() {
        feature_means[j] = features.iter().map(|x| x[j]).sum::<f64>() / features.len() as f64;
        feature_stds[j] = (features.iter().map(|x| (x[j] - feature_means[j]).powi(2)).sum::<f64>() / features.len() as f64).sqrt();

        for i in 0..features.len() {
            features[i][j] = (features[i][j] - feature_means[j]) / feature_stds[j].max(1e-8);
        }
    }

    let split_index = (features.len() as f64 * 0.8) as usize;
    let (train_features, test_features) = features.split_at(split_index);
    let (train_targets, test_targets) = targets.split_at(split_index);

    Ok((train_features.to_vec(), train_targets.to_vec(), test_features.to_vec(), test_targets.to_vec()))
}
