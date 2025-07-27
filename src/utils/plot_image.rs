use burn::tensor::{backend::Backend, Tensor};
use ndarray::{Array2, ArrayView2};
use plotters::{coord::Shift, prelude::*};
use colorous;

/// Function to convert a 2D tensor to a 2D array for plotting
fn tensor_to_ndarray_2d<B: Backend>(tensor: &Tensor<B, 2>) -> Array2<f32> {
    let shape = tensor.dims(); // Use dims() to get the shape as an array
    let array = tensor.clone().into_data().to_vec().expect("Failed to convert tensor to vec"); // Unwrap and expect Result<Vec<_>, DataError>
    Array2::from_shape_vec((shape[0], shape[1]), array).unwrap() // Create a 2D ndarray
}

/// Function to plot a 2D array on a specific drawing area
fn plot_array_on_area(
    array: ArrayView2<f32>,
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
) -> Result<(), Box<dyn std::error::Error>> {
    drawing_area.fill(&WHITE)?;

    // Set the explicit normalization range
    let min = 0.0; // Lower bound of the color ramp (0 meters)
    let max = 25.0; // Upper bound of the color ramp (50 meters)

    let color_map = colorous::VIRIDIS; // Choose a color map for visualization

    // Draw the image pixel by pixel
    for (y, row) in array.axis_iter(ndarray::Axis(0)).enumerate() {
        for (x, &val) in row.iter().enumerate() {
            // Clamp the value between min and max
            let clamped_val = val.max(min).min(max);

            // Normalize the value within the range 0-50 meters
            let norm_val = (clamped_val - min) / (max - min);

            let color = color_map.eval_continuous(norm_val.into()); // Convert to f64
            drawing_area.draw(&Pixel::new((x as i32, y as i32), RGBColor(color.r, color.g, color.b)))?;
        }
    }

    Ok(())
}

/// Helper to plot a 2D array with a custom colormap and normalization
fn plot_array_with_colormap(
    array: ArrayView2<f32>,
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
    min: f32,
    max: f32,
    color_map: &dyn Fn(f64) -> RGBColor,
) -> Result<(), Box<dyn std::error::Error>> {
    drawing_area.fill(&WHITE)?;
    for (y, row) in array.axis_iter(ndarray::Axis(0)).enumerate() {
        for (x, &val) in row.iter().enumerate() {
            let clamped_val = val.max(min).min(max);
            let norm_val = if (max - min).abs() > 1e-6 {
                (clamped_val - min) / (max - min)
            } else {
                0.0
            };
            let color = color_map(norm_val as f64);
            drawing_area.draw(&Pixel::new((x as i32, y as i32), color))?;
        }
    }
    Ok(())
}

fn draw_colorbar(
    drawing_area: &DrawingArea<BitMapBackend, Shift>,
    min: f32,
    max: f32,
    color_map: &dyn Fn(f64) -> RGBColor,
    is_diverging: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (w, h) = drawing_area.dim_in_pixel();
    for y in 0..h {
        let norm_val = 1.0 - (y as f64) / (h as f64 - 1.0); // top=1.0, bottom=0.0
        let color = color_map(norm_val);
        for x in 0..w {
            drawing_area.draw(&Pixel::new((x as i32, y as i32), color))?;
        }
    }
    // Draw min/max labels
    let text_style = ("sans-serif", 12).into_font().into_text_style(drawing_area);
    drawing_area.draw_text(&format!("{:.2}", max), &text_style, (w as i32 + 2, 0))?;
    drawing_area.draw_text(&format!("{:.2}", min), &text_style, (w as i32 + 2, h as i32 - 16))?;
    if is_diverging {
        // Draw zero label in the middle
        drawing_area.draw_text("0", &text_style, (w as i32 + 2, h as i32 / 2 - 8))?;
    }
    Ok(())
}

/// Function to plot the input and target images side by side in a single plot
/// with a gap between them and labels above each image.
pub fn plot_input_vs_target<B: Backend>(
    input: Tensor<B, 4>,
    target: Tensor<B, 4>,
    filename: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = input.dims();
    let image_height = shape[2] as u32;
    let image_width = shape[3] as u32;
    let gap = 30;  // Gap between the two images

    // Extract the first example from the batch and reduce the dimensions to 2D
    let input_image = input.slice([
        0..1, // Select the first batch
        0..1, // Select the first channel
        0..shape[2], // Select all height
        0..shape[3], // Select all width
    ]).reshape([shape[2], shape[3]]); // Reshape to 2D

    let target_image = target.slice([
        0..1, // Select the first batch
        0..1, // Select the first channel
        0..shape[2], // Select all height
        0..shape[3], // Select all width
    ]).reshape([shape[2], shape[3]]); // Reshape to 2D

    // Convert Tensor to ndarray for plotting
    let input_ndarray = tensor_to_ndarray_2d(&input_image);
    let target_ndarray = tensor_to_ndarray_2d(&target_image);

    // Create a drawing area that is wide enough for both images plus the gap and some extra margin for labels
    let root = BitMapBackend::new(filename, (2 * image_width + gap, image_height + 40)).into_drawing_area();
    root.fill(&WHITE)?;

    // Split the canvas into two parts, with a margin for labels at the top
    let (upper, lower) = root.split_vertically(30);
    let (left_area, right_area) = lower.split_horizontally(image_width + gap);

    // Draw labels above the images
    upper.draw_text(
        "Input",
        &("sans-serif", 12).into_font().into_text_style(&upper),
        (image_width as i32 / 2 - 15, 5)
    )?;
    upper.draw_text(
        "Target",
        &("sans-serif", 12).into_font().into_text_style(&upper),
        (image_width as i32 + gap as i32 + image_width as i32 / 2 - 15, 5)
    )?;

    // Plot input image on the left side
    plot_array_on_area(input_ndarray.view(), &left_area,)?;

    // Plot target image on the right side
    plot_array_on_area(target_ndarray.view(), &right_area)?;

    //println!("Saved input and target images with labels and a gap to a single plot.");

    Ok(())
}

/// Plot inferenced, target, and signed difference images side by side
pub fn plot_inferenced_vs_target_and_diff<B: Backend>(
    inferenced: Tensor<B, 4>,
    target: Tensor<B, 4>,
    filename: &str
) -> Result<(), Box<dyn std::error::Error>> {
    let shape = inferenced.dims();
    let image_height = shape[2] as u32;
    let image_width = shape[3] as u32;
    let gap = 30;
    let colorbar_width = 20;
    let colorbar_gap = 10;

    // Extract first batch/channel, reshape to 2D
    let inferenced_image = inferenced.slice([
        0..1, 0..1, 0..shape[2], 0..shape[3]
    ]).reshape([shape[2], shape[3]]);
    let target_image = target.slice([
        0..1, 0..1, 0..shape[2], 0..shape[3]
    ]).reshape([shape[2], shape[3]]);

    let inferenced_nd = tensor_to_ndarray_2d(&inferenced_image);
    let target_nd = tensor_to_ndarray_2d(&target_image);
    let diff_nd = &inferenced_nd - &target_nd;

    // Shared min/max for grayscale
    let min_val = inferenced_nd.iter().chain(target_nd.iter()).fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = inferenced_nd.iter().chain(target_nd.iter()).fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // For difference: symmetric range around zero
    let max_abs_diff = diff_nd.iter().map(|v| v.abs()).fold(0.0_f32, |a, b| a.max(b));

    // Prepare canvas
    let total_width = 3 * image_width + 2 * gap + 3 * (colorbar_width + colorbar_gap);
    let root = BitMapBackend::new(filename, (total_width, image_height + 40)).into_drawing_area();
    root.fill(&WHITE)?;
    let (upper, lower) = root.split_vertically(30);
    // Split for 3 images + 3 colorbars
    let (left, rest) = lower.split_horizontally(image_width + colorbar_width + colorbar_gap + gap);
    let (center, rest) = rest.split_horizontally(image_width + colorbar_width + colorbar_gap + gap);
    let (right, after_right) = rest.split_horizontally(image_width + colorbar_width + colorbar_gap);

    // Now split each image area for image/colorbar
    let (left_img, left_cb) = left.split_horizontally(image_width);
    let (center_img, center_cb) = center.split_horizontally(image_width);
    let (right_img, right_cb) = right.split_horizontally(image_width);

    // Labels
    upper.draw_text(
        "Inferenced",
        &("sans-serif", 12).into_font().into_text_style(&upper),
        (image_width as i32 / 2 - 30, 5)
    )?;
    upper.draw_text(
        "Target",
        &("sans-serif", 12).into_font().into_text_style(&upper),
        (image_width as i32 + gap as i32 + colorbar_width as i32 + colorbar_gap as i32 + image_width as i32 / 2 - 25, 5)
    )?;
    upper.draw_text(
        "Difference",
        &("sans-serif", 12).into_font().into_text_style(&upper),
        (2 * (image_width as i32 + gap as i32 + colorbar_width as i32 + colorbar_gap as i32) + image_width as i32 / 2 - 35, 5)
    )?;

    // Grayscale colormap
    let gray_map = |v: f64| {
        let g = (v * 255.0).round().clamp(0.0, 255.0) as u8;
        RGBColor(g, g, g)
    };
    // Diverging colormap: blue (min) - white (zero) - red (max)
    let diverge_map = |v: f64| {
        // v in [0,1], 0.5 is zero
        if v < 0.5 {
            // interpolate blue to white
            let t = v / 0.5;
            let r = (255.0 * t).round() as u8;
            let g = (255.0 * t).round() as u8;
            let b = 255u8;
            RGBColor(r, g, b)
        } else {
            // interpolate white to red
            let t = (v - 0.5) / 0.5;
            let r = 255u8;
            let g = (255.0 * (1.0 - t)).round() as u8;
            let b = (255.0 * (1.0 - t)).round() as u8;
            RGBColor(r, g, b)
        }
    };

    plot_array_with_colormap(inferenced_nd.view(), &left_img, min_val, max_val, &gray_map)?;
    draw_colorbar(&left_cb, min_val, max_val, &gray_map, false)?;
    plot_array_with_colormap(target_nd.view(), &center_img, min_val, max_val, &gray_map)?;
    draw_colorbar(&center_cb, min_val, max_val, &gray_map, false)?;
    plot_array_with_colormap(
        diff_nd.view(),
        &right_img,
        -max_abs_diff,
        max_abs_diff,
        &diverge_map
    )?;
    draw_colorbar(&right_cb, -max_abs_diff, max_abs_diff, &diverge_map, true)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Shape;
    use burn::backend::ndarray::NdArray;

    #[test]
    fn test_plot_input_vs_target() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Create a mock 4D tensor for input and target using the NdArrayBackend
        // The shape is (batch_size, channels, height, width)

        // Create 4D Tensors: (1, 1, 32, 32)
        let input: Tensor<NdArray, 4> = Tensor::random(Shape::new([1, 1, 256, 256]), burn::tensor::Distribution::Default, &device);
        let target: Tensor<NdArray, 4> = Tensor::random(Shape::new([1, 1, 256, 256]), burn::tensor::Distribution::Default, &device);

        // Call the function to plot input vs target in a single plot with labels
        let result = plot_input_vs_target(input, target, "input_vs_target_with_gap.png");

        // Ensure the function returns OK (no error)
        assert!(result.is_ok());

        // Verify that the single image file was created
        assert!(std::path::Path::new("input_vs_target_with_gap.png").exists());

        // Optionally, clean up the generated file after the test
        //std::fs::remove_file("input_vs_target_with_gap.png").expect("Failed to delete image");
    }

    #[test]
    fn test_plot_inferenced_vs_target_and_diff() {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let inferenced: Tensor<NdArray, 4> = Tensor::random(Shape::new([1, 1, 128, 128]), burn::tensor::Distribution::Default, &device);
        let target: Tensor<NdArray, 4> = Tensor::random(Shape::new([1, 1, 128, 128]), burn::tensor::Distribution::Default, &device);
        let result = plot_inferenced_vs_target_and_diff(inferenced, target, "inferenced_vs_target_and_diff.png");
        assert!(result.is_ok());
        assert!(std::path::Path::new("inferenced_vs_target_and_diff.png").exists());
    }
}
