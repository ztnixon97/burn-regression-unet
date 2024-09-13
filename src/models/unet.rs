use burn::{
    module::Module, nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, pool::{MaxPool2d, MaxPool2dConfig},
    }, prelude::*, tensor::backend::Backend,
};

use super::model_parts::*;

#[derive(Module, Debug)]
pub struct UNet<B: Backend> {
    down1: DoubleConv<B>,
    down2: DoubleConv<B>,
    down3: DoubleConv<B>,
    down4: DoubleConv<B>,
    bottleneck: DoubleConv<B>,
    up1: ConvTranspose2d<B>,
    upconv1: DoubleConv<B>,
    up2: ConvTranspose2d<B>,
    upconv2: DoubleConv<B>,
    up3: ConvTranspose2d<B>,
    upconv3: DoubleConv<B>,
    up4: ConvTranspose2d<B>,
    upconv4: DoubleConv<B>,
    final_conv: Conv2d<B>,
    max_pool: MaxPool2d,
}

impl<B: Backend> UNet<B> {
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

        // Decoder
        let x = self.up1.forward(x_bottleneck);
        let x = Tensor::cat((&[x, conv4]).to_vec(), 1);
        let x = self.upconv1.forward(x);

        let x = self.up2.forward(x);
        let x = Tensor::cat((&[x, conv3]).to_vec(), 1);
        let x = self.upconv2.forward(x);

        let x = self.up3.forward(x);
        let x = Tensor::cat((&[x, conv2]).to_vec(), 1);
        let x = self.upconv3.forward(x);

        let x = self.up4.forward(x);
        let x = Tensor::cat((&[x, conv1]).to_vec(), 1);
        let x = self.upconv4.forward(x);

        // Final Convolution
        self.final_conv.forward(x)


    }
}

impl<B: Backend> UNet<B> {
    pub fn init(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let down1 = DoubleConvConfig::new(in_channels, 64).init(device);
        let down2 = DoubleConvConfig::new(64, 128).init(device);
        let down3 = DoubleConvConfig::new(128, 256).init(device);
        let down4 = DoubleConvConfig::new(256, 512).init(device);
        let bottleneck = DoubleConvConfig::new(512, 1024).init(device);

        let up1 = ConvTranspose2dConfig::new([1024, 512], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv1 = DoubleConvConfig::new(1024, 512).init(device);

        let up2 = ConvTranspose2dConfig::new([512, 256], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv2 = DoubleConvConfig::new(512, 256).init(device);

        let up3 = ConvTranspose2dConfig::new([256, 128], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv3 = DoubleConvConfig::new(256, 128).init(device);

        let up4 = ConvTranspose2dConfig::new([128, 64], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv4 = DoubleConvConfig::new(128, 64).init(device);

        let final_conv = Conv2dConfig::new([64, out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);

        let max_pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        UNet {
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




#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use burn::backend::{Wgpu,Autodiff};
    use serial_test::serial;

    #[test]
    fn test_unet_forward_pass() {
        // Initialize device (using CPU for testing)
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a UNet model with 3 input channels (e.g., RGB image) and 1 output channel (e.g., segmentation mask)
        let unet:UNet<MyBackend> = UNet::init(3, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 128;
        let width = 128;
        let input = Tensor::random(Shape::new([batch_size, 3, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));
    }

    #[test]
    #[serial]
    fn test_unet_backward_pass_autodiff() {
        type MyBackend = Wgpu<f32, i32>;
        let device = burn::backend::wgpu::WgpuDevice::default();

        let unet_pp: UNet<Autodiff<MyBackend>> = UNet::init(1, 1, &device);

        let batch_size = 1;
        let height = 32;
        let width = 32;

        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        println!("Loss: {:?}", loss);

        // Perform the backward pass
        loss.backward(); // This computes the gradients
    }

    #[test]
    #[serial]
    fn test_unet_backward_pass_autodiff_ndarray() {
        type MyBackend = burn::backend::ndarray::NdArray;
        let device = burn::backend::ndarray::NdArrayDevice::default();

        let unet_pp: UNet<Autodiff<MyBackend>> = UNet::init(1, 1, &device);

        let batch_size = 1;
        let height = 32;
        let width = 32;

        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        println!("Loss: {:?}", loss);

        // Perform the backward pass
        loss.backward(); // This computes the gradients
    }

    #[test]
    #[serial]
    fn test_unet_forward_pass_torch() {
        // Initialize device using Torch backend
        type MyBackend = burn::backend::libtorch::LibTorch;

        let device = burn::backend::libtorch::LibTorchDevice::Cpu;

        // Create a UNet++ model with 1 input channel and 1 output channel
        let unet_pp: UNet<MyBackend> = UNet::init(1, 1, &device);

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

    #[test]
    #[serial]
    fn test_unet_backward_pass_torch() {
        // Use NdArray backend that supports autodiff
        type MyBackend = Autodiff<burn::backend::libtorch::LibTorch>;

        let device = burn::backend::libtorch::LibTorchDevice::Cpu;

        // Create a UNet++ model with 1 input channel and 1 output channel
        let unet_pp: UNet<MyBackend> = UNet::init(1, 1, &device);

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
}



