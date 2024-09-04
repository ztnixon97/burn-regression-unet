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


        // Decoder with Nested Skip Connections
        let mut skips = vec![];
        skips.push(conv4.clone());

        let x = self.up1.forward(x_bottleneck);

        let x = Tensor::cat((&[x, conv4]).to_vec(), 1);

        let x = self.upconv1.forward(x, &skips);


        skips.push(conv3.clone());
        let x = self.up2.forward(x);

        let x = Tensor::cat((&[x, conv3]).to_vec(), 1);

        let x = self.upconv2.forward(x, &skips);


        skips.push(conv2.clone());
        let x = self.up3.forward(x);

        let x = Tensor::cat((&[x, conv2]).to_vec(), 1);

        let x = self.upconv3.forward(x, &skips);
  

        skips.push(conv1.clone());
        let x = self.up4.forward(x);

        let x = Tensor::cat((&[x, conv1]).to_vec(), 1);

        let x = self.upconv4.forward(x, &skips);
 

        // Final Convolution
        let x = self.final_conv.forward(x);


        // Optional: Apply sigmoid
        //let x = sigmoid(x);
        //println!("Shape after sigmoid: {:?}", x.shape());
        
        x
    }
}

impl<B: Backend> UNetPlusPlus<B> {
    pub fn init(in_channels: usize, out_channels: usize, device: &B::Device) -> Self {
        let down1 = DoubleConvConfig::new(in_channels, 64).init(device);
        let down2 = DoubleConvConfig::new(64, 128).init(device);
        let down3 = DoubleConvConfig::new(128, 256).init(device);
        let down4 = DoubleConvConfig::new(256, 512).init(device);
        let bottleneck = DoubleConvConfig::new(512, 1024).init(device);

        let up1 = ConvTranspose2dConfig::new([1024, 512], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv1 = NestedConvConfig::new(1024, 512, 1).init(device);

        let up2 = ConvTranspose2dConfig::new([512, 256], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv2 = NestedConvConfig::new(512, 256, 2).init(device);

        let up3 = ConvTranspose2dConfig::new([256, 128], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv3 = NestedConvConfig::new(256, 128, 3).init(device);

        let up4 = ConvTranspose2dConfig::new([128, 64], [2, 2])
            .with_stride([2, 2])
            .init(device);
        let upconv4 = NestedConvConfig::new(128, 64, 4).init(device);

        let final_conv = Conv2dConfig::new([64, out_channels], [1, 1])
            .with_stride([1, 1])
            .init(device);

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
    use burn::backend::Wgpu;

    #[test]
    fn test_unetplusplus_forward_pass() {
        // Initialize device (using CPU for testing)
        type MyBackend = Wgpu<f32, i32>;

        let device = burn::backend::wgpu::WgpuDevice::default();

        // Create a UNet++ model with 3 input channels and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(8, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1;
        let height = 256;
        let width = 256;
        let input = Tensor::random(Shape::new([batch_size, 8, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));

        // Check that the output values are within the expected range (0 to 1, due to sigmoid)

    }



}
