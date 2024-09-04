use burn::{
    module::Module, nn::{
        conv::{Conv2d, Conv2dConfig}, BatchNorm, BatchNormConfig, PaddingConfig2d, Relu
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
