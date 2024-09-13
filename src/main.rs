use burn::{
    backend::{libtorch::{LibTorch, LibTorchDevice}, wgpu::WgpuDevice, Autodiff}, data::dataloader::DataLoaderBuilder,
     optim::AdamConfig, prelude::*, record::CompactRecorder,
      tensor::{backend::{AutodiffBackend, Backend}, Tensor},
       train::{
        metric::{Adaptor, LossInput, LossMetric}, LearnerBuilder,
    }
};




use burn::train::TrainOutput;

const INPUT_CHANNELS: usize = 8;
#[cfg(feature="ndarray")]
use burn::backend::ndarray::NdArray;
#[cfg(not(feature="ndarray"))]
use burn::backend::Wgpu;


pub mod models;
pub mod dataset;
use models::unetplusplus::UNetPlusPlus;
use models::simple_unet::SimpleUNet as UNet;
use dataset::data::{TerrainBatch, TerrainBatcher, TerrainDataset};
use burn::nn::loss::MseLoss;
use burn::train::TrainStep;
use burn::train::ValidStep;




pub struct RegressionOutput4d<B: Backend> {
    /// The loss.
    pub loss: Tensor<B, 1>,

    /// The output.
    pub output: Tensor<B, 4>,

    /// The targets.
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> RegressionOutput4d<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        Self { loss, output, targets }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for RegressionOutput4d<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Config, Debug)]
pub struct UnetConfig {
    in_channels: usize,
    out_channels: usize,
}

impl UnetConfig {
    /// Initializes the UNet model based on the provided configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNet<B> {
        UNet::init(self.in_channels, self.out_channels, device)
    }
}

#[derive(Config, Debug)]
pub struct UnetPlusPlusConfig {
    in_channels: usize,
    out_channels: usize,
}

impl UnetPlusPlusConfig {
    /// Initializes the UNetPlusPlus model based on the provided configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> UNetPlusPlus<B> {
        UNetPlusPlus::init(self.in_channels, self.out_channels, device)
    }
}


pub struct RegressionBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,  // Input images (batch_size, 8, height, width)
    pub targets: Tensor<B, 4>, // Target images (batch_size, 1, height, width)
}

impl<B: Backend> UNet<B> {
    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>, // Input shape: (batch_size, channels, height, width)
        targets: Tensor<B, 4>, // Target shape: (batch_size, channels, height, width)
    ) -> RegressionOutput4d<B> {
        // Forward pass through the model
        let output = self.forward(images); 

        //println!("Output: {:?}", output.clone());
        
        // Flatten spatial dimensions (height and width) into a single dimension
        let output = output; // Resulting shape: (batch_size, channels * height * width)
        let targets = targets; // Ensure targets have the same shape

        // Calculate MSE loss
        let loss = MseLoss::new()
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);
        //println!("Loss: {:?}", loss.clone());
        // Return the regression output struct
        RegressionOutput4d::new(loss, output, targets)
    }
}



impl<B: Backend> UNetPlusPlus<B> {
    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>, // Input shape (batch_size, channels, height, width)
        targets: Tensor<B, 4>, // Target shape (batch_size, channels, height, width)
    ) -> RegressionOutput4d<B> {
        let output = self.forward(images); // Flatten spatial dimensions to match the target shape

        let targets = targets; // Flatten targets in the same way

        let loss = MseLoss::new()
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);

        RegressionOutput4d::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TerrainBatch<B>, RegressionOutput4d<B>> for UNetPlusPlus<B> {
    fn step(&self, batch: TerrainBatch<B>) -> TrainOutput<RegressionOutput4d<B>> {
        // Perform the forward pass and compute the output
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TerrainBatch<B>, RegressionOutput4d<B>> for UNetPlusPlus<B> {
    fn step(&self, batch: TerrainBatch<B>) -> RegressionOutput4d<B> {
        // Perform the forward pass
        self.forward_regression(batch.inputs, batch.targets)
    }
}

impl<B: AutodiffBackend> TrainStep<TerrainBatch<B>, RegressionOutput4d<B>> for UNet<B> {
    fn step(&self, batch: TerrainBatch<B>) -> TrainOutput<RegressionOutput4d<B>> {
        // Perform the forward pass and compute the output

        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TerrainBatch<B>, RegressionOutput4d<B>> for UNet<B> {
    fn step(&self, batch: TerrainBatch<B>) -> RegressionOutput4d<B> {
        // Perform the forward pass
        self.forward_regression(batch.inputs, batch.targets)
    }
}


#[derive(Config)]
pub struct TrainingConfig {
    pub model: UnetPlusPlusConfig,
    pub optimizer: AdamConfig,
    #[config(default = 2)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

#[derive(Config)]
pub struct TrainingConfigUnet {
    pub model: UnetConfig,
    pub optimizer: AdamConfig,
    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 1)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);

    let model = config.model.init(&device);
    B::seed(config.seed);

    // Use the outer backend type B for training
    let batcher_train = TerrainBatcher::<B>::new(device.clone());

    // Use the inner backend type for validation
    let batcher_valid = TerrainBatcher::<B::InnerBackend>::new(device.clone());

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(TerrainDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(TerrainDataset::test());

    

    // Initialize learner
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            model,
            config.optimizer.init(),
            config.learning_rate,
        );

    // Train model
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    // Save the trained model
    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}




fn main() {
    #[cfg(feature="ndarray")]
    {
        type MyBackend = NdArray;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        // Create a UNet++ configuration
        let model_config = UnetPlusPlusConfig {
            in_channels: INPUT_CHANNELS,
            out_channels: 1,
        };

        // Create a training configuration
        let training_config = TrainingConfig::new(model_config, AdamConfig::new());

        // Start training
        train::<MyAutodiffBackend>(
            "G:/burn-unet/learner",
            training_config,
            device,
        );
    }
    #[cfg(feature="torch")]
    {
        type TchBackend = LibTorch;
        type MyAutodiffBackend = Autodiff<TchBackend>;

        //let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
        let device = LibTorchDevice::Cpu;
        // Create a UNet++ configuration
        let model_config = UnetPlusPlusConfig {
            in_channels: INPUT_CHANNELS,
            out_channels: 1,
        };

        // Create a training configuration
        let training_config = TrainingConfig::new(model_config, AdamConfig::new());

        // Start training
        train::<MyAutodiffBackend>(
            "G:/burn-unet/learner",
            training_config,
            device,
        );
    }
    #[cfg(feature="wgpu")]
    {
        type MyBackend = Wgpu<f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        //let device = burn::backend::wgpu::WgpuDevice::BestAvailable;
        let device = WgpuDevice::BestAvailable;
        // Create a UNet++ configuration
        let model_config = UnetPlusPlusConfig {
            in_channels: INPUT_CHANNELS,
            out_channels: 1,
        };

        // Create a training configuration
        let training_config = TrainingConfig::new(model_config, AdamConfig::new());

        // Start training
        train::<MyAutodiffBackend>(
            "G:/burn-unet/learner",
            training_config,
            device,
        );
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{backend::Backend, Tensor, Shape};
    use burn::backend::Autodiff;
    use serial_test::serial;


    // Function to test the model for a given input size
    fn test_model<B: Backend>(size: usize) -> bool {
        let device = B::Device::default();
        let model_config = UnetPlusPlusConfig {
            in_channels: 8,
            out_channels: 1,
        };

        let model = model_config.init(&device);
        // Perform the forward pass, ensuring that the input tensor is on the same device
        let _output = model.forward(Tensor::<B, 4>::zeros(Shape::new([1, 8, size, size]), &device));
        true // Forward pass succeeded
    }
    

    #[test]
    #[serial]
    #[ignore = "This test will break other tests and should be run alone"]
    fn test_largest_input_size() {
        // Initialize the device and the model configuration
        type MyBackend = burn::backend::wgpu::Wgpu<f32, i32>; 
        type MyAutodiffBackend= Autodiff<MyBackend>; // Change backend as necessary
        

        let mut max_size = 16; // Start with 32x32

        // Test for increasing powers of 2 until failure
        while max_size <= 512 { // Test up to 512x512
            println!("Testing model with input size {}x{}", max_size, max_size);

            if test_model::<MyAutodiffBackend>(max_size) {
                println!("Model succeeded with input size {}x{}", max_size, max_size);
                max_size *= 2;  // Increase input size to the next power of 2
            } else {
                println!("Model failed with input size {}x{}", max_size, max_size);
                break;
            }
        }

        println!("Largest successful input size: {}x{}", max_size / 2, max_size / 2);

        // Optionally, assert that the model can handle at least a certain size (e.g., 256x256)
        assert!(max_size / 2 >= 128, "Model should handle at least 128x128 input size.");
    }

}
