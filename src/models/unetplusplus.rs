use burn::{
    module::Module, nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, pool::{MaxPool2d, MaxPool2dConfig},
    }, prelude::*, tensor::backend::Backend,
};

use super::model_parts::*;

#[derive(Module, Debug)]
pub struct UNetPlusPlus<B: Backend> {
    down1: DoubleConv<B>,
    down2: DoubleConv<B>,
    down3: DoubleConv<B>,
    down4: DoubleConv<B>,
    bottleneck: DoubleConv<B>,
    up1: ConvTranspose2d<B>,
    upconv1: NestedConv<B>,
    up2: ConvTranspose2d<B>,
    upconv2: NestedConv<B>,
    up3: ConvTranspose2d<B>,
    upconv3: NestedConv<B>,
    up4: ConvTranspose2d<B>,
    upconv4: NestedConv<B>,
    final_conv: Conv2d<B>,
    max_pool: MaxPool2d,
}


impl<B: Backend> UNetPlusPlus<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Encoder
        let conv1 = self.down1.forward(x);
        let x = self.max_pool.forward(conv1.clone());
        let conv2 = self.down2.forward(x);
        let x = self.max_pool.forward(conv2.clone());
        let conv3 = self.down3.forward(x);
        let x = self.max_pool.forward(conv3.clone());
        let conv4 = self.down4.forward(x);
        let x = self.max_pool.forward(conv4.clone());

        // Bottleneck
        let x_bottleneck = self.bottleneck.forward(x);

        // Decoder with Nested Skip Connections, explicitly passing borrowed skip tensors
        // Stage 1: upconv1 gets conv4 as the skip connection
        let x = self.up1.forward(x_bottleneck);
        let x = self.upconv1.forward(x, &[&conv4]);

        // Stage 2: upconv2 gets conv4 and conv3 as skip connections
        let x = self.up2.forward(x);
        let x = self.upconv2.forward(x, &[&conv4, &conv3]);

        // Stage 3: upconv3 gets conv4, conv3, and conv2 as skip connections
        let x = self.up3.forward(x);
        let x = self.upconv3.forward(x, &[&conv4, &conv3, &conv2]);

        // Stage 4: upconv4 gets conv4, conv3, conv2, and conv1 as skip connections
        let x = self.up4.forward(x);
        let x = self.upconv4.forward(x, &[&conv4, &conv3, &conv2, &conv1]);

        // Final Convolution
        self.final_conv.forward(x)
    }
}





impl<B: Backend> UNetPlusPlus<B> {
    pub fn init(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let n1 = 64;
        let filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16];

        // Downsampling layers
        let down1 = DoubleConvConfig::new(in_channels, filters[0]).init(device);
        let down2 = DoubleConvConfig::new(filters[0], filters[1]).init(device);
        let down3 = DoubleConvConfig::new(filters[1], filters[2]).init(device);
        let down4 = DoubleConvConfig::new(filters[2], filters[3]).init(device);
        let bottleneck = DoubleConvConfig::new(filters[3], filters[4]).init(device);

        // Upsampling layers and nested convs, now using mid_channels
        let up1 = ConvTranspose2dConfig::new([filters[4], filters[3]], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv1 = NestedConvConfig::new(filters[4], filters[3] / 2, filters[3]).init(device);  // mid_channels = filters[3] / 2

        let up2 = ConvTranspose2dConfig::new([filters[3], filters[2]], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv2 = NestedConvConfig::new(filters[3], filters[2] / 2, filters[2]).init(device);  // mid_channels = filters[2] / 2

        let up3 = ConvTranspose2dConfig::new([filters[2], filters[1]], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv3 = NestedConvConfig::new(filters[2], filters[1] / 2, filters[1]).init(device);  // mid_channels = filters[1] / 2

        let up4 = ConvTranspose2dConfig::new([filters[1], filters[0]], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv4 = NestedConvConfig::new(filters[1], filters[0] / 2, filters[0]).init(device);  // mid_channels = filters[0] / 2

        // Final convolution layer to match the output channels
        let final_conv = Conv2dConfig::new([filters[0], out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);

        // Max pooling
        let max_pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        UNetPlusPlus {
            down1,
            down2,
            down3,
            down4,
            bottleneck,
            up1,
            upconv1,
            up2,
            upconv2,
            up3,
            upconv3,
            up4,
            upconv4,
            final_conv,
            max_pool,
        }
    }
}



#[derive(Config, Debug)]
pub struct UnetPlusPlusConfig {
    in_channels: usize,
    out_channels: usize,

}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use burn::backend::{Autodiff, Wgpu};
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_unetplusplus_forward_pass_wgpu() {
        // Initialize device using Wgpu backend
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a UNet++ model with 8 input channels and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(1, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 32;
        let width = 32;
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));
    }

    #[test]
    #[serial]
    fn test_unetplusplus_backward_pass_wgpu() {
        // Use Wgpu backend that supports autodiff
        type MyBackend = Autodiff<Wgpu<f32, i32>>;
        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a UNet++ model with 1 input channel and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(1, 1, &device);

        let batch_size = 1;
        let height = 32;
        let width = 32;

        // Create a random input tensor and mark it as requiring gradients
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        // Perform the backward pass to compute gradients
        loss.backward(); // This computes the gradients
    }

    #[test]
    #[serial]
    fn test_unetplusplus_backward_pass_ndarray() {
        // Use NdArray backend that supports autodiff
        type MyBackend = Autodiff<burn::backend::ndarray::NdArray>;
        let device = burn::backend::ndarray::NdArrayDevice::default();

        // Create a UNet++ model with 1 input channel and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(1, 1, &device);

        let batch_size = 1;
        let height = 32;
        let width = 32;

        // Create a random input tensor and mark it as requiring gradients
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        // Perform the backward pass to compute gradients
        loss.backward(); // This computes the gradients
    }

    #[test]
    #[serial]
    fn test_unetplusplus_forward_pass_ndarray() {
        // Initialize device using NdArray backend
        type MyBackend = burn::backend::ndarray::NdArray;

        let device = burn::backend::ndarray::NdArrayDevice::default();

        // Create a UNet++ model with 1 input channel and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(1, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 64;
        let width = 64;
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));
    }
}
