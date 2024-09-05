use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor:: backend::Backend,
};


#[derive(Module, Debug)]
pub struct SimpleConv<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    relu: Relu,
}

impl<B: Backend> SimpleConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

#[derive(Debug)]
pub struct SimpleConvConfig {
    in_channels: usize,
    out_channels: usize,
}

impl SimpleConvConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self { in_channels, out_channels }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> SimpleConv<B> {
        let conv = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn = BatchNormConfig::new(self.out_channels).init(device);
        let relu = Relu::new();

        SimpleConv { conv, bn, relu }
    }
}

#[derive(Module, Debug)]
pub struct SimpleUNet<B: Backend> {
    down1: SimpleConv<B>,
    down2: SimpleConv<B>,
    bottleneck: SimpleConv<B>,
    up1: ConvTranspose2d<B>,
    upconv1: SimpleConv<B>,
    up2: ConvTranspose2d<B>,
    upconv2: SimpleConv<B>,
    final_conv: Conv2d<B>,
    max_pool: MaxPool2d,
}

impl<B: Backend> SimpleUNet<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Encoder
        let conv1 = self.down1.forward(x);
        let x = self.max_pool.forward(conv1.clone());

        let conv2 = self.down2.forward(x);
        let x = self.max_pool.forward(conv2.clone());

        // Bottleneck
        let x_bottleneck = self.bottleneck.forward(x);

        // Decoder
        let x = self.up1.forward(x_bottleneck);
        let x = Tensor::cat((&[x, conv2]).to_vec(), 1);
        let x = self.upconv1.forward(x);

        let x = self.up2.forward(x);
        let x = Tensor::cat((&[x, conv1]).to_vec(), 1);
        let x = self.upconv2.forward(x);

        // Final Convolution
        self.final_conv.forward(x)

    }

    pub fn init(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let down1 = SimpleConvConfig::new(in_channels, 16).init(device);
        let down2 = SimpleConvConfig::new(16, 32).init(device);
        let bottleneck = SimpleConvConfig::new(32, 64).init(device);

        let up1 = ConvTranspose2dConfig::new([64, 32], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv1 = SimpleConvConfig::new(64, 32).init(device);

        let up2 = ConvTranspose2dConfig::new([32, 16], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv2 = SimpleConvConfig::new(32, 16).init(device);

        let final_conv = Conv2dConfig::new([16, out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);

        let max_pool = MaxPool2dConfig::new([2, 2])
            .with_strides([2, 2])
            .init();

        SimpleUNet {
            down1,
            down2,
            bottleneck,
            up1,
            upconv1,
            up2,
            upconv2,
            final_conv,
            max_pool,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor};
    use burn::backend::Wgpu;
    use burn::backend::Autodiff;
    use serial_test::serial;


    #[test]
    #[serial]
    fn test_simple_unet_forward_pass() {
        // Initialize device
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a SimpleUNet model with 3 input channels and 1 output channel
        let simple_unet: SimpleUNet<MyBackend> = SimpleUNet::init(3, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 32;
        let width = 32;
        let input = Tensor::random(Shape::new([batch_size, 3, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = simple_unet.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));
    }

    #[test]
    #[serial]
    fn test_simple_unet_backward_pass() {
        // Initialize device
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a SimpleUNet model with 3 input channels and 1 output channel
        let simple_unet: SimpleUNet<MyBackend> = SimpleUNet::init(3, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 32;
        let width = 32;
        let input = Tensor::random(Shape::new([batch_size, 3, height, width]), burn::tensor::Distribution::Default, &device);

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = simple_unet.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);
        println!("Loss: {:?}", loss);
        // Perform the backward pass

    }
    #[test]
    #[serial]
    fn test_simple_unet_backward_pass_autodiff() {
        // Initialize device
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a SimpleUNet model with 3 input channels and 1 output channel
        let simple_unet: SimpleUNet<Autodiff<MyBackend>> = SimpleUNet::init(1, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 32;
        let width = 32;
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = simple_unet.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        println!("Loss: {:?}", loss);

        // Perform the backward pass
        loss.backward(); // This computes the gradients
    }
    #[test]
    #[serial]
    fn test_simple_unet_backward_pass_autodiff_ndarray() {
        // Initialize device
        type MyBackend = burn::backend::ndarray::NdArray;


        let device = burn::backend::ndarray::NdArrayDevice::default();


        // Create a SimpleUNet model with 3 input channels and 1 output channel
        let simple_unet: SimpleUNet<Autodiff<MyBackend>> = SimpleUNet::init(1, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 32;
        let width = 32;
        let input = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device).require_grad();

        // Create a random target tensor with the same shape as the output
        let target = Tensor::random(Shape::new([batch_size, 1, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = simple_unet.forward(input.clone());

        // Compute the Mean Squared Error (MSE) loss
        let loss = burn::nn::loss::MseLoss::new()
            .forward(output, target, burn::nn::loss::Reduction::Mean);

        println!("Loss: {:?}", loss);

        // Perform the backward pass
        loss.backward(); // This computes the gradients
    }
}
