use anyhow::Result;
use neural_network::dataset::spiral::PointWithClass::{
    ParamsForNewSeriesOfPointWithClass, SeriesOfPointWithClass,
};
use plotters::prelude::*;

const OUTPUT_FILE_NAME: &str = "view_spiral_data.png";

fn main() -> Result<()> {
    let props = ParamsForNewSeriesOfPointWithClass {
        point_per_class: 100,
        number_of_class: 3,
        max_angle: 2.0 * std::f64::consts::PI,
    };
    let points = SeriesOfPointWithClass::new(props);
    let series = points.get_points();
    let series_1 = series.get(0).unwrap();
    let series_2 = series.get(1).unwrap();
    let series_3 = series.get(2).unwrap();

    let root = BitMapBackend::new(OUTPUT_FILE_NAME, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(70)
        .y_label_area_size(80)
        .margin(10)
        .caption("Spiral", ("sans-serif", 40.0).into_font())
        .build_cartesian_2d(-1_f64..1_f64, -1_f64..1_f64)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_desc("x")
        .y_desc("y")
        .x_label_formatter(&|x| format!("{:1.1}", x))
        .y_label_formatter(&|x| format!("{:1.1}", x))
        .axis_desc_style(FontDesc::new(FontFamily::SansSerif, 32., FontStyle::Normal))
        .label_style(FontDesc::new(FontFamily::SansSerif, 24., FontStyle::Normal))
        .draw()?;

    chart
        .draw_series(series_1.iter().map(|point| Circle::new(*point, 5, &BLUE)))?
        .label("class 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
    chart
        .draw_series(series_2.iter().map(|point| Circle::new(*point, 5, &RED)))?
        .label("class 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));
    chart
        .draw_series(series_3.iter().map(|point| Circle::new(*point, 5, &GREEN)))?
        .label("class 3")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .label_font(FontDesc::new(FontFamily::SansSerif, 32., FontStyle::Normal))
        .draw()?;

    // To avoid the IO failure being ignored silently, we manually call the present function
    root.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
    println!("Result has been saved to {}", OUTPUT_FILE_NAME);
    Ok(())
}
