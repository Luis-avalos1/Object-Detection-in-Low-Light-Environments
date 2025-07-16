# Object Detection in Low-Light Environments

A comprehensive research project that evaluates and compares object detection performance in low-light conditions using various image enhancement techniques and the YOLOv5 model.

 --> WORK IN PROGRESS <--
## 🔍 Overview

This project investigates how different image enhancement methods can improve object detection accuracy in challenging low-light environments. We compare the performance of several enhancement techniques including histogram equalization, CLAHE, Retinex, gamma correction, and brightness/contrast adjustment on the ExDark dataset.

## 🚀 Features

- **Multiple Enhancement Techniques**: Implements histogram equalization, CLAHE, Single Scale Retinex, gamma correction, and brightness/contrast adjustment
- **YOLOv5 Integration**: Uses pre-trained YOLOv5 models for robust object detection
- **Comprehensive Evaluation**: Calculates precision, recall, and average precision metrics
- **Visual Results**: Generates annotated images showing detection results
- **ExDark Dataset Support**: Full compatibility with the ExDark low-light dataset

## 📁 Project Structure

```
├── src/                     # Core source code
│   ├── object_detection.py  # Main detection script
│   └── enhance.py           # Image enhancement algorithms
├── utils/                   # Utility scripts
│   ├── enhanced_labeling.py
│   ├── enhance_labels_batch.py
│   ├── fix_labels.py
│   └── rename_annotation_script.py
├── notebooks/               # Experimental notebooks and test scripts
├── data/                    # Dataset and model files
│   ├── ExDark_Dataset/      # Low-light image dataset
│   ├── ground_truths/       # Annotation files
│   └── models/              # Pre-trained models
├── results/                 # Output results
│   ├── detection_results/   # Annotated detection images
│   └── enhanced_images/     # Enhanced versions of input images
└── README.md
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Object-Detection-in-Low-Light-Environments.git
   cd Object-Detection-in-Low-Light-Environments
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision ultralytics opencv-python numpy scikit-learn matplotlib tqdm
   ```

## 📊 Dataset

This project uses the [ExDark dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset), which contains:
- **7,363** low-light images across 12 object classes
- Multiple lighting conditions (ambient, object, single, weak, strong, screen, window)
- PASCAL VOC format annotations

### Supported Classes:
- Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Dog, Motorbike, People, Table, Others

## 🎯 Usage

### Basic Object Detection
```bash
python src/object_detection.py
```

### Image Enhancement Only
```bash
python src/enhance.py
```

## 📈 Enhancement Techniques

1. **Histogram Equalization**: Redistributes pixel intensities for better contrast
2. **CLAHE**: Contrast Limited Adaptive Histogram Equalization for local enhancement
3. **Single Scale Retinex**: Logarithmic image processing for illumination estimation
4. **Gamma Correction**: Power-law transformation for brightness adjustment
5. **Brightness/Contrast**: Linear adjustments to overall image appearance

## 🔬 Evaluation Metrics

- **Precision**: Proportion of true positive detections among all positive detections
- **Recall**: Proportion of true positive detections among all ground truth objects
- **Average Precision (AP)**: Area under the precision-recall curve
- **IoU Threshold**: 0.5 for determining true/false positives

## 📊 Results

The project generates comprehensive evaluation reports comparing:
- Detection performance across different enhancement methods
- Precision-recall curves for each technique
- Visual comparisons with annotated bounding boxes
- Statistical analysis of detection improvements

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [ExDark Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) by Loh et al.
- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics
- OpenCV community for image processing tools

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact [avaloseluis2@gmail.com].

---

*This project was developed as part of research into improving computer vision performance in challenging lighting conditions.*
