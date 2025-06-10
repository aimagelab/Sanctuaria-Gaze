# Sanctuaria-Gaze

**Sanctuaria-Gaze** is a multimodal dataset of egocentric recordings from visits to four sanctuaries in Northern Italy. Alongside the data, we release an open-source framework
for automatic detection and analysis of Areas of Interest (AOIs), enabling gaze-based research in dynamic, real-world settings without manual annotation.

> **Note:** The dataset will be released soon. Please stay tuned for updates!

## Features

- Batch processing of folders with gaze/video pairs
- Single-file processing for custom workflows
- IDT scanpath generation
- Frame extraction from videos
- Automated annotation prediction
- Creation of annotated output videos
- Command-line interface with flexible options

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/aimagelab/Sanctuaria-Gaze.git
cd Sanctuaria-Gaze
pip install -r requirements.txt
```

## Usage

### Single File

```bash
python annotate.py --idt --verbose path/to/subject_gaze.csv path/to/subject.mp4
```

### Batch Folder

```bash
python annotate.py --idt --verbose path/to/folder/
```

### Options

- `--idt` : Run IDT scanpath generation
- `--no-extract` : Skip frame extraction
- `--no-annotate` : Skip annotation
- `--no-video` : Skip video creation
- `--idt-dis-threshold FLOAT` : Set IDT dispersion threshold (default: 0.05)
- `--idt-dur-threshold INT` : Set IDT duration threshold (default: 100)
- `--stop-frame INT` : Stop after this frame number
- `--verbose` : Enable verbose logging

## Input Format

The input `_gaze.csv` file **must** include the following columns:

- `gaze_timestamp`
- `world_index`
- `confidence`
- `norm_pos_x`
- `norm_pos_y`

each row represents a single gaze sample, with normalized gaze positions (`norm_pos_x`, `norm_pos_y`) ranging from 0 to 1, and timestamps (`gaze_timestamp`) in seconds. The `confidence` column indicates the reliability of each gaze point, and `world_index` corresponds to the frame index in the associated video.

## Object Classes

The list of object names (classes) used for detection must be specified in a plain text file named `object_classes.txt`.  
Each line should contain a single object name. Lines starting with `#` are treated as comments and ignored.

Example `object_classes.txt`:
```
# Religious and architectural objects
altar
crucifix
pews
pulpit
chalice
...
```

This file should be placed in the project directory.  
The detection models (e.g., YOLOv8_World, OWLv2) will automatically load object names from this file at runtime.

### Example Demo

Suppose you have the following files in `data/`:

- `subject1_gaze.csv`
- `subject1.mp4`
- `subject2_gaze.csv`
- `subject2.mp4`

To process all pairs in the folder:

```bash
python annotate.py --idt data/
```

To process a single pair with verbose output and custom IDT thresholds:

```bash
python annotate.py data/subject1_gaze.csv data/subject1.mp4 --idt --idt-dis-threshold 0.07 --idt-dur-threshold 120 --verbose
```

## Output

- Annotated CSV files and processed videos are saved in the output directory.
- Temporary extracted frames and intermediate files are cleaned up automatically.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## Citation

If you use this tool in your research, please cite:

```bibtex
```

---

*For more details, see the code documentation and comments in `annotate.py`.*
