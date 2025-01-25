use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{transform::{Mapper, MapperDataset}, Dataset, InMemDataset, SqliteDataset},
    },
    prelude::*,
    tensor::{backend::Backend, module::interpolate, ops::{InterpolateMode, InterpolateOptions}},
};
use std::{
    fs::File,
    io::{BufReader, Read},
};
use bytemuck;
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

const HEIGHT: usize = 256;
const WIDTH: usize = 256;
const INPUT_CHANNELS: usize = 8;

#[derive(Deserialize, Clone, Debug)]
pub struct TerrainDataItemRaw {
    pub image_bytes: Vec<f32>,
    pub target_bytes: Vec<f32>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct TerrainDataItem {
    pub input: TensorData,
    pub target: TensorData,
}

struct BytesToTensorData;

impl Mapper<TerrainDataItemRaw, TerrainDataItem> for BytesToTensorData {
    fn map(&self, item: &TerrainDataItemRaw) -> TerrainDataItem {

        let input = TensorData::new(item.image_bytes.clone(), Shape::new([INPUT_CHANNELS, HEIGHT, WIDTH]));
        let target = TensorData::new(item.target_bytes.clone(), Shape::new([1, HEIGHT, WIDTH]));

        TerrainDataItem {input, target }
    }
}

// Type alias for the generic MappedDataset
type MappedDataset<D> = MapperDataset<D, BytesToTensorData, TerrainDataItemRaw>;

pub struct TerrainDataset<D> {
    dataset: MappedDataset<D>,
}

impl TerrainDataset<InMemDataset<TerrainDataItemRaw>> {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        let folder = match split {
            "train" => "F:/test_data/train",
            "test" => "F:/test_data/val",
            _ => panic!("Invalid split"),
        };

        let raw_items = Self::load_folder(folder, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let in_mem_dataset = InMemDataset::new(raw_items);
        let dataset = MapperDataset::new(in_mem_dataset, BytesToTensorData);

        TerrainDataset { dataset }
    }

    fn load_folder(folder: &str, input_bands: Vec<usize>) -> Vec<TerrainDataItemRaw> {
        WalkDir::new(folder)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && e.path().extension().unwrap_or_default() == "bin")
            .map(|e| {
                let file = File::open(e.path()).unwrap();
                let mut reader = BufReader::new(file);

                let height = HEIGHT;
                let width = WIDTH;
                let num_bands = 9; // Total number of bands in the binary file

                // The total number of elements in the file, assuming 9 bands
                let total_num_elements = num_bands * width * height;

                // Read the entire binary data into a buffer of floats
                let mut buffer = vec![0.0f32; total_num_elements];
                reader.read_exact(bytemuck::cast_slice_mut(&mut buffer)).expect("Failed to read binary data");

                // Extract only the selected input bands for the input (image_bytes)
                let mut image_bytes = Vec::with_capacity(input_bands.len() * width * height);
                for &band in &input_bands {
                    let start_index = (band - 1) * width * height;
                    let end_index = start_index + width * height;
                    image_bytes.extend_from_slice(&bytemuck::cast_slice(&buffer[start_index..end_index]));
                }

                // Extract band 9 for the target (target_bytes)
                let target_start_index = (9 - 1) * width * height;
                let target_bytes: Vec<f32> = buffer[target_start_index..(target_start_index + width * height)].to_vec();
                TerrainDataItemRaw {
                    image_bytes,
                    target_bytes: bytemuck::cast_slice(&target_bytes).to_vec(),
                }
            })
            .collect()
    }
}

impl TerrainDataset<SqliteDataset<TerrainDataItemRaw>> {
    pub fn train_sqlite() -> Self {
        Self::new_sqlite("train")
    }

    pub fn test_sqlite() -> Self {
        Self::new_sqlite("test")
    }

    fn new_sqlite(split: &str) -> Self {
        let sqlite_dataset: SqliteDataset<TerrainDataItemRaw> = SqliteDataset::from_db_file("F:/2m_dsm_dtm_256px_py.db", split).unwrap();
        let dataset = MapperDataset::new(sqlite_dataset, BytesToTensorData);
        TerrainDataset { dataset }
    }
}

impl<D> Dataset<TerrainDataItem> for TerrainDataset<D>
where
    D: Dataset<TerrainDataItemRaw>,
{
    fn get(&self, index: usize) -> Option<TerrainDataItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

#[derive(Clone)]
pub struct TerrainBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TerrainBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    fn downsample(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let output_size = [128, 128];
        let run = true;

        if run {
            // Get the dimensions of the input tensor (channels, height, width)
            let dims = tensor.dims();
            let height = dims[1];
            let width = dims[2];

            // Calculate the starting coordinates for cropping to the center
            let crop_height = output_size[0];
            let crop_width = output_size[1];
            let start_height = (height - crop_height) / 2;
            let start_width = (width - crop_width) / 2;

            // Crop the tensor by passing the ranges for all dimensions at once
            tensor.slice([
                None, // Keep all channels (no cropping on the first dimension)
                Some((start_height as i64, (start_height + crop_height) as i64)), // Crop the height
                Some((start_width as i64, (start_width + crop_width) as i64)),   // Crop the width
            ])
        } else {
            tensor
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerrainBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> Batcher<TerrainDataItem, TerrainBatch<B>> for TerrainBatcher<B> {
    fn batch(&self, items: Vec<TerrainDataItem>) -> TerrainBatch<B> {
        // Collect and downsample the inputs
        let inputs: Vec<_> = items
            .iter()
            .map(|item| {
                let tensor = Tensor::<B, 3>::from_floats(item.input.clone(), &self.device);
                let downsampled = self.downsample(tensor).unsqueeze(); // Downsample input
                downsampled
            })
            .collect();

        // Collect and downsample the targets
        let targets: Vec<_> = items
            .iter()
            .map(|item| {
                let tensor = Tensor::<B, 3>::from_floats(item.target.clone(), &self.device);
                let downsampled = self.downsample(tensor).unsqueeze(); // Downsample target
                downsampled
            })
            .collect();

        TerrainBatch {
            inputs: Tensor::cat(inputs, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    // Test that the folder loading works correctly.
    #[test]
    fn test_load_folder() {
        let input_bands = vec![1, 2];
        let data_items = TerrainDataset::<InMemDataset<TerrainDataItemRaw>>::load_folder("F:/test_data/train", input_bands);

        assert!(!data_items.is_empty(), "The dataset should not be empty");
        println!("Loaded {} items", data_items.len());
        for item in data_items {
            assert_eq!(item.image_bytes.len(), HEIGHT * WIDTH * 2, "Image bytes should match input bands size");
            assert_eq!(item.target_bytes.len(), HEIGHT * WIDTH, "Target bytes should match the target size");
        }
    }

    // Test that the dataset conversion works correctly.
    #[test]
    fn test_dataset_mapping() {
        type MyBackend = NdArray;

        let device = burn::backend::ndarray::NdArrayDevice::default();
        let input_bands = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let raw_items = TerrainDataset::<InMemDataset<TerrainDataItemRaw>>::load_folder("F:/test_data/train", input_bands);
        let in_mem_dataset = InMemDataset::new(raw_items);
        let mapped_dataset = MapperDataset::new(in_mem_dataset, BytesToTensorData);

        let terrain_dataset = TerrainDataset { dataset: mapped_dataset };

        assert!(terrain_dataset.len() > 0, "The dataset should have items");

        if let Some(item) = terrain_dataset.get(0) {
            // Check that the tensor has the right shape and content
            let input: Tensor<MyBackend, 3> = Tensor::from_floats(item.input.clone(), &device);
            let target: Tensor<MyBackend, 3> = Tensor::from_floats(item.target.clone(), &device);

            assert_eq!(input.shape(), Shape::new([INPUT_CHANNELS, HEIGHT, WIDTH]), "Input tensor shape mismatch");
            assert_eq!(target.shape(), Shape::new([1, HEIGHT, WIDTH]), "Target tensor shape mismatch");
        } else {
            panic!("Failed to retrieve item from dataset");
        }
    }

    #[test]
    fn test_batcher() {
        type MyBackend = NdArray;
        let device = burn::backend::ndarray::NdArrayDevice::default();
        let input_bands = vec![1, 2, 3, 4, 5, 6, 7, 8];

        let raw_items = TerrainDataset::<InMemDataset<TerrainDataItemRaw>>::load_folder("F:/test_data/train", input_bands);
        let in_mem_dataset = InMemDataset::new(raw_items);
        let mapped_dataset = MapperDataset::new(in_mem_dataset, BytesToTensorData);

        let terrain_dataset = TerrainDataset { dataset: mapped_dataset };

        let batcher = TerrainBatcher::<MyBackend>::new(device);

        let items: Vec<_> = (0..terrain_dataset.len()).filter_map(|i| terrain_dataset.get(i)).collect();
        let batch = batcher.batch(items.clone());

        assert_eq!(batch.inputs.shape(), Shape::new([items.len(), INPUT_CHANNELS, HEIGHT, WIDTH]), "Input batch shape mismatch");
        assert_eq!(batch.targets.shape(), Shape::new([items.len(), 1, HEIGHT, WIDTH]), "Target batch shape mismatch");
    }
}
