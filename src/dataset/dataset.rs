use burn::tensor::{backend::Backend, Shape};
use burn::data::dataset::Dataset;
use burn::data::dataloader::batcher::Batcher;

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use serde::{Deserialize, Serialize};
use burn::prelude::*;
use bytemuck;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;

#[cfg(feature="gdal")]
use gdal::Dataset as GdalDataset;
/// Represents a single dataset item loaded from a GeoTIFF file.
#[derive(Clone, Debug)]
pub struct GeoTiffDatasetItem {
    pub input: TensorData,  // 8 bands of input data
    pub target: TensorData, // 1 band of target data
}

/// Represents the raw path to a GeoTIFF file.
#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct GeoTiffDatasetItemRaw {
    image_path: PathBuf, // Path to the GeoTIFF file
}

impl GeoTiffDatasetItemRaw {
    pub fn new<P: AsRef<Path>>(image_path: P) -> GeoTiffDatasetItemRaw {
        GeoTiffDatasetItemRaw {
            image_path: image_path.as_ref().to_path_buf(),
        }
    }
}

/// A dataset structure for managing a collection of GeoTIFF files.
#[derive(Debug)]
pub struct GeoTiffDataset<B: Backend> {
    items: Vec<GeoTiffDatasetItemRaw>,
    input_bands: Vec<usize>,
    _marker: PhantomData<B>,
}

impl<B: Backend> GeoTiffDataset<B> {

    #[cfg(not(feature="gdal"))]
    pub fn from_folder<P: AsRef<Path>>(root: P, input_bands: Vec<usize>) -> Self {
        let files = WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && e.path().extension().unwrap_or_default() == "bin")
            .map(|e| GeoTiffDatasetItemRaw::new(e.path()))
            .collect();

        Self {
            items: files,
            input_bands,
            _marker: PhantomData,
        }
    }
    
    #[cfg(feature="gdal")]
    pub fn from_folder<P: AsRef<Path>>(root: P, input_bands: Vec<usize>) -> Self {
        let files = WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file() && e.path().extension().unwrap_or_default() == "tif")
            .map(|e| GeoTiffDatasetItemRaw::new(e.path()))
            .collect();

        Self {
            items: files,
            input_bands,
            _marker: PhantomData,
        }
    }
    #[cfg(not(feature="gdal"))]
    pub fn load_item(&self, item: &GeoTiffDatasetItemRaw) -> GeoTiffDatasetItem {
        let file = File::open(&item.image_path).expect("Failed to open Binary file");
        let mut reader = BufReader::new(file);
        let height = 512;
        let width = 512;
    
        // Calculate the number of elements for the selected input bands and target (1 band)
        let num_elements_input = self.input_bands.len() * width * height;

        // The total number of elements in the file, assuming all 9 bands are present
        let total_num_elements = 9 * width * height;
    
        // Read the entire binary data into a buffer
        let mut buffer = vec![0.0f32; total_num_elements];
        reader.read_exact(bytemuck::cast_slice_mut(&mut buffer)).expect("Failed to read binary data");
    
        // Create a new buffer for the selected input bands
        let mut input_data = Vec::with_capacity(num_elements_input);
    
        // Read only the selected bands into the input_data buffer
        for &band in &self.input_bands {
            let start_index = (band - 1) * width * height;
            let end_index = start_index + width * height;
            input_data.extend_from_slice(&buffer[start_index..end_index]);
        }
    
        // Read the last band (band 9) as the target data
        let target_start_index = (9 - 1) * width * height;
        let target_data = &buffer[target_start_index..];
    
        // Convert input_data and target_data into Data<f32, 3>
        let input_data = TensorData::new(input_data, Shape::new([self.input_bands.len(), height, width]));
        let target_data = TensorData::new(target_data.to_vec(), Shape::new([1, height, width]));

        GeoTiffDatasetItem { input: input_data, target: target_data }
    }

    #[cfg(feature="gdal")]
    pub fn load_item(&self, item: &GeoTiffDatasetItemRaw) -> GeoTiffDatasetItem {
        let dataset = GdalDataset::open(&item.image_path).expect("Failed to open GeoTIFF file");
        let (width, height) = dataset.raster_size();

        // Load selected input bands
        let mut input_data = Vec::with_capacity(self.input_bands.len() * height * width);
        for &i in &self.input_bands {
            let band = dataset.rasterband(i).unwrap();
            let mut buffer = vec![0.0f32; (width * height) as usize];
            band.read_into_slice(
                (0, 0),
                band.size(),
                (width, height),
                &mut buffer,
                None,
            ).expect("Failed to read band data");
            input_data.extend(buffer);
        }

        // Load target band (9)
        let mut target_data = vec![0.0f32; (width * height) as usize];
        let band = dataset.rasterband(9).unwrap();
        band.read_into_slice(
            (0, 0),
            band.size(),
            (width, height),
            &mut target_data,
            None,
        ).expect("Failed to read band data");

        // Convert input_data and target_data into Data<f32, 3>
        let input_data = TensorData::new(input_data, Shape::new([self.input_bands.len(), height, width]));
        let target_data = TensorData::new(target_data.to_vec(), Shape::new([1, height, width]));

        GeoTiffDatasetItem { input: input_data, target: target_data }
    }
}

impl<B: Backend> Dataset<GeoTiffDatasetItem> for GeoTiffDataset<B> {
    fn get(&self, index: usize) -> Option<GeoTiffDatasetItem> {
        self.items.get(index).map(|item| self.load_item(item))
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

/// A batch structure for holding a batch of GeoTIFF data.
#[derive(Clone, Debug)]
pub struct GeoTiffBatch<B: Backend> {
    pub inputs: Tensor<B, 4>,  // Batch of inputs (batch_size, 8, height, width)
    pub targets: Tensor<B, 4>, // Batch of targets (batch_size, 1, height, width)
}

/// A batcher structure for creating batches of GeoTIFF data.
#[derive(Clone, Debug)]
pub struct GeoTiffBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> GeoTiffBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<GeoTiffDatasetItem, GeoTiffBatch<B>> for GeoTiffBatcher<B> {
    fn batch(&self, items: Vec<GeoTiffDatasetItem>) -> GeoTiffBatch<B> {
        let inputs: Vec<_> = items
            .iter()
            .map(|item| {
                let tensor = Tensor::<B,3>::from_floats(item.input.clone(), &self.device).unsqueeze();
                tensor
            })
            .collect();
        
        let targets: Vec<_> = items
            .iter()
            .map(|item| {
                let tensor = Tensor::<B,3>::from_floats(item.target.clone(), &self.device).unsqueeze();
                tensor
            })
            .collect();

        GeoTiffBatch {
            inputs: Tensor::cat(inputs, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}
