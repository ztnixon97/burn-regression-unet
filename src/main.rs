use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::{backend::{AutodiffBackend, Backend}, Tensor},
    train::{
        metric::{Adaptor, LossInput, LossMetric,}, LearnerBuilder,
    },
    backend::Autodiff,
};


use burn::train::TrainOutput;

const INPUT_CHANNELS: usize = 1;
#[cfg(feature="ndarray")]
use burn::backend::ndarray::NdArray;
#[cfg(not(feature="ndarray"))]
use burn::backend::Wgpu;

pub mod models;
pub mod dataset;
use models::unet::UNetPlusPlus;
use models::simple_unet::SimpleUNet as UNet;
use dataset::dataset::{GeoTiffDataset,GeoTiffBatch, GeoTiffBatcher };
use burn::nn::loss::MseLoss;
use burn::train::TrainStep;
use burn::train::ValidStep;
use burn::train::RegressionOutput;


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
    ) -> RegressionOutput<B> {
        // Forward pass through the model
        let output = self.forward(images); 
        
        // Flatten spatial dimensions (height and width) into a single dimension
        let output = output.flatten(1, 3); // Resulting shape: (batch_size, channels * height * width)
        let targets = targets.flatten(1, 3); // Ensure targets have the same shape

        // Calculate MSE loss
        let loss = MseLoss::new()
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);
        println!("Loss: {:?}", loss.clone());
        // Return the regression output struct
        RegressionOutput::new(loss, output, targets)
    }
}



impl<B: Backend> UNetPlusPlus<B> {
    pub fn forward_regression(
        &self,
        images: Tensor<B, 4>, // Input shape (batch_size, channels, height, width)
        targets: Tensor<B, 4>, // Target shape (batch_size, channels, height, width)
    ) -> RegressionOutput<B> {
        let output = self.forward(images)
            .flatten(2, 3); // Flatten spatial dimensions to match the target shape

        let targets = targets.flatten(2, 3); // Flatten targets in the same way

        let loss = MseLoss::new()
            .forward(output.clone(), targets.clone(), nn::loss::Reduction::Mean);

        RegressionOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<GeoTiffBatch<B>, RegressionOutput<B>> for UNetPlusPlus<B> {
    fn step(&self, batch: GeoTiffBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        // Perform the forward pass and compute the output
        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GeoTiffBatch<B>, RegressionOutput<B>> for UNetPlusPlus<B> {
    fn step(&self, batch: GeoTiffBatch<B>) -> RegressionOutput<B> {
        // Perform the forward pass
        self.forward_regression(batch.inputs, batch.targets)
    }
}

impl<B: AutodiffBackend> TrainStep<GeoTiffBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: GeoTiffBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        // Perform the forward pass and compute the output

        let item = self.forward_regression(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<GeoTiffBatch<B>, RegressionOutput<B>> for UNet<B> {
    fn step(&self, batch: GeoTiffBatch<B>) -> RegressionOutput<B> {
        // Perform the forward pass
        self.forward_regression(batch.inputs, batch.targets)
    }
}


#[derive(Config)]
pub struct TrainingConfig {
    pub model: UnetPlusPlusConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 2)]
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
    #[config(default = 10)]
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

fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfigUnet, device: B::Device) {
    create_artifact_dir(artifact_dir);

    let model = config.model.init(&device);
    B::seed(config.seed);

    // Use the outer backend type B for training
    let batcher_train = GeoTiffBatcher::<B>::new(device.clone());

    // Use the inner backend type for validation
    let batcher_valid = GeoTiffBatcher::<B::InnerBackend>::new(device.clone());

    // Create data loaders
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(GeoTiffDataset::<B>::from_folder("F:/test_data/val", vec![1]));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(1)
        .shuffle(config.seed)
        .num_workers(1)
        .build(GeoTiffDataset::<B::InnerBackend>::from_folder("F:/test_data/val", vec![1]));

    

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
        let model_config = UnetConfig {
            in_channels: INPUT_CHANNELS,
            out_channels: 1,
        };

        // Create a training configuration
        let training_config = TrainingConfigUnet::new(model_config, AdamConfig::new());

        // Start training
        train::<MyAutodiffBackend>(
            "G:/burn-unet/learner",
            training_config,
            device,
        );
    }
    #[cfg(not(feature="ndarray"))]
    {
        type MyBackend = Wgpu<f32, i32>;
        type MyAutodiffBackend = Autodiff<MyBackend>;

        let device = burn::backend::wgpu::WgpuDevice::DiscreteGpu(0);
        // Create a UNet++ configuration
        let model_config = UnetConfig {
            in_channels: INPUT_CHANNELS,
            out_channels: 1,
        };

        // Create a training configuration
        let training_config = TrainingConfigUnet::new(model_config, AdamConfig::new());

        // Start training
        train::<MyAutodiffBackend>(
            "G:/burn-unet/learner",
            training_config,
            device,
        );
    }
}
