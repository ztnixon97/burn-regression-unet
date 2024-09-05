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
pub struct NestedConvOld<B: Backend> {
    conv: DoubleConv<B>,
    skips: Vec<DoubleConv<B>>,
    relu: Relu,
}

impl<B: Backend> NestedConvOld<B> {
    // Now the function accepts borrowed references to tensors
    pub fn forward(&self, x: Tensor<B, 4>, skips: &[&Tensor<B, 4>]) -> Tensor<B, 4> {
        let mut out = x;

        for (i, &skip) in skips.iter().enumerate() {
            let skip_out = self.skips[i].forward(skip.clone());

            // Get the current shape of 'out'
            let shape = out.shape().dims;
            let output_size = [shape[2], shape[3]];  // height and width of 'out'

            // Upsample the skip connection to match the dimensions of 'out'
            let upsampled_skip = interpolate(
                skip_out,
                output_size,  // Match the height and width
                InterpolateOptions::new(InterpolateMode::Nearest),
            );
            // Now 'out' and 'upsampled_skip' should have the same spatial dimensions
            out = Tensor::cat((&[out, upsampled_skip]).to_vec(), 1);  // Add skip connection
        }

        // Final convolution inside NestedConvOld
        self.conv.forward(out)
    }
}



#[derive(Debug)]
pub struct NestedConvOldConfig {
    in_channels: usize,
    out_channels: usize,
    num_skips: usize,
}

impl NestedConvOldConfig {
    pub fn new(in_channels: usize, out_channels: usize, num_skips: usize) -> Self {
        Self { in_channels, out_channels, num_skips }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> NestedConvOld<B> {
        let conv = DoubleConvConfig::new(self.in_channels, self.out_channels).init(device);

        let mut skips = Vec::with_capacity(self.num_skips);
        for _ in 0..self.num_skips {
            skips.push(DoubleConvConfig::new(self.out_channels, self.out_channels).init(device));
        }

        let relu = Relu::new();


        NestedConvOld {
            conv,
            skips,
            relu,
        }
    }
}

#[derive(Module, Debug)]
pub struct NestedConv<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    activation: Relu,
}

impl<B: Backend> NestedConv<B> {
    // Now the function accepts borrowed references to tensors
    pub fn forward(&self, x: Tensor<B, 4>, skips: &[&Tensor<B, 4>]) -> Tensor<B, 4> {
        let shape = x.shape().dims;
        let output_size = [shape[2], shape[3]];  // Get height and width of `x`
    
        let mut all_skips = Vec::with_capacity(skips.len());
    
        // Iterate over all skip connections
        for skip in skips.iter() {
            let skip_shape = skip.shape().dims;
    
            // If the spatial dimensions of the skip don't match `x`, upsample it
            let skip: Tensor<B, 4> = if skip_shape[2] != shape[2] || skip_shape[3] != shape[3] {
                interpolate(
                    (*skip).clone(),
                    output_size,  // Match height and width of `x`
                    InterpolateOptions::new(InterpolateMode::Nearest),
                )
            } else {
                (*skip).clone()
            };
    
            all_skips.push(skip);
        }
    
        // Concatenate all skip connections with the input `x`
        let x = Tensor::cat((&[x, Tensor::cat(all_skips, 1)]).to_vec(), 1);  // Concatenate along channel dimension
        // Forward pass through the convolutional layers
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.activation.forward(x);
    
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let output = self.activation.forward(x);
    
        output
    }
    
    
}



#[derive(Debug)]
pub struct NestedConvConfig {
    in_channels: usize,
    mid_channels: usize,
    out_channels: usize,
}

impl NestedConvConfig {
    pub fn new (in_channels: usize, mid_channels: usize, out_channels: usize) -> Self {
        Self { in_channels, mid_channels, out_channels }
    }
    pub fn init<B:Backend>(&self, device: &B::Device) -> NestedConv<B> {
        let conv1 = Conv2dConfig::new([self.in_channels, self.mid_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn1 = BatchNormConfig::new(self.mid_channels).init(device);

        let conv2 = Conv2dConfig::new([self.mid_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn2 = BatchNormConfig::new(self.out_channels).init(device);

        let activation = Relu::new();

        NestedConv {
            conv1,
            bn1,
            conv2,
            bn2,
            activation,
        }
    }
}
