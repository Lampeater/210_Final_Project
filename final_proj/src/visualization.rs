use plotters::prelude::*;
use std::error::Error;

pub fn plot_predictions(actual: &[f64], predicted: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("prediction_plot.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Actual vs Predicted Prices", ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0..actual.len(), 0f64..actual.iter().map(|&v| v.exp()).fold(0./0., f64::max))?;

    chart.configure_mesh().draw()?;

    chart.draw_series(
        predicted.iter().enumerate().map(|(i, &y_pred)| {
            Circle::new((i, y_pred.exp()), 5, BLUE.filled())
        }),
    )?;

    chart.draw_series(
        actual.iter().enumerate().map(|(i, &y)| {
            Circle::new((i, y.exp()), 5, RED.filled())
        }),
    )?;

    Ok(())
}
