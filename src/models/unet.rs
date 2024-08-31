use burn::{
    module::Module, nn::{
        conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig}, pool::{MaxPool2d, MaxPool2dConfig}, BatchNorm, BatchNormConfig, PaddingConfig2d, Relu
    }, prelude::*, tensor::{backend::Backend, module::interpolate, ops::{InterpolateMode, InterpolateOptions}
    },
};


#[derive(Module, Debug)]
pub struct DoubleConv<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    relu: Relu,
}

impl<B: Backend> DoubleConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        self.relu.forward(x)
    }
}

#[derive(Debug)]
pub struct DoubleConvConfig {
    in_channels: usize,
    out_channels: usize,
}

impl DoubleConvConfig {
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self { in_channels, out_channels }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> DoubleConv<B> {
        let conv1 = Conv2dConfig::new([self.in_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(self.out_channels).init(device);
        let conv2 = Conv2dConfig::new([self.out_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(self.out_channels).init(device);
        let relu = Relu::new();

        DoubleConv {
            conv1,
            bn1,
            conv2,
            bn2,
            relu,
        }
    }
}


#[derive(Module, Debug)]
pub struct NestedConv<B: Backend> {
    conv: DoubleConv<B>,
    skips: Vec<DoubleConv<B>>,
    channel_adjust: Option<Conv2d<B>>,
    relu: Relu,
}
impl<B: Backend> NestedConv<B> {
    pub fn forward(&self, x: Tensor<B, 4>, skips: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        let mut out = x;


        for (i, skip) in skips.iter().enumerate() {
            let skip_out = self.skips[i].forward(skip.clone());


            let shape = out.shape().dims;
            let output_size = [shape[2], shape[3]];

            let upsampled_skip = interpolate(
                skip_out,
                output_size,
                InterpolateOptions::new(InterpolateMode::Nearest),
            );


            let adjusted_skip = if let Some(channel_adjust) = &self.channel_adjust {
                let adjusted = channel_adjust.forward(upsampled_skip);
               
                adjusted
            } else {

                upsampled_skip
            };

            out = out + adjusted_skip;

        }

        // Final convolution inside NestedConv

        let out = self.conv.forward(out);

        out
    }
}

#[derive(Debug)]
pub struct NestedConvConfig {
    in_channels: usize,
    out_channels: usize,
    num_skips: usize,
}

impl NestedConvConfig {
    pub fn new(in_channels: usize, out_channels: usize, num_skips: usize) -> Self {
        Self { in_channels, out_channels, num_skips }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> NestedConv<B> {
        let conv = DoubleConvConfig::new(self.in_channels, self.out_channels).init(device);

        let mut skips = Vec::with_capacity(self.num_skips);
        for _ in 0..self.num_skips {
            skips.push(DoubleConvConfig::new(self.out_channels, self.out_channels).init(device));
        }

        let relu = Relu::new();

        // If the in_channels and out_channels differ, we need a 1x1 convolution to adjust
        let channel_adjust = if self.in_channels != self.out_channels {
            Some(
                Conv2dConfig::new([self.out_channels, self.in_channels], [1, 1])
                    .with_stride([1, 1])
                    .init(device),
            )
        } else {
            None
        };

        NestedConv {
            conv,
            skips,
            channel_adjust,
            relu,
        }
    }
}

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

    #[test]
    fn test_unetplusplus_forward_pass_ndarray() {
        // Initialize device (using CPU for testing)
        type MyBackend = burn::backend::ndarray::NdArray;


        let device = burn::backend::ndarray::NdArrayDevice::default();

        // Create a UNet++ model with 3 input channels and 1 output channel
        let unet_pp: UNetPlusPlus<MyBackend> = UNetPlusPlus::init(8, 1, &device);

        // Create a random input tensor with shape (batch_size, channels, height, width)
        let batch_size = 1 ;
        let height = 64;
        let width = 64;
        let input = Tensor::random(Shape::new([batch_size, 8, height, width]), burn::tensor::Distribution::Default, &device);

        // Perform a forward pass
        let output = unet_pp.forward(input);

        // Check that the output has the correct shape
        assert_eq!(output.shape(), Shape::new([batch_size, 1, height, width]));
    }

}
