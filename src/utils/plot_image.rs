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
        Some((0, 1)), // Select the first batch
        Some((0, 1)), // Select the first channel
        Some((0, shape[2] as i64)), // Select all height
        Some((0, shape[3] as i64)), // Select all width
    ]).reshape([shape[2], shape[3]]); // Reshape to 2D

    let target_image = target.slice([
        Some((0, 1)), // Select the first batch
        Some((0, 1)), // Select the first channel
        Some((0, shape[2] as i64)), // Select all height
        Some((0, shape[3] as i64)), // Select all width
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
}
