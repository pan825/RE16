# RE16 (R-EPG16 Model)

This repository contains the implementation of the R-EPG16 model, a sophisticated model inspired by the research on neural mechanisms of visual motion processing in insects. The model is designed to simulate and analyze the neural responses to visual motion stimuli, particularly focusing on the processing of optic flow and motion detection.

## Project Structure

```
RE16/
├── 1D/           # 1D model implementations
├── 2D/           # 2D model implementations
└── .gitignore    # Git ignore rules
```

## Requirements

- Python 3.7 or higher
- Brian2 (>= 2.5.0)
- NumPy
- SciPy
- Matplotlib (for visualization)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/RE16.git
cd RE16
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install the required packages:
```bash
pip install brian2 numpy scipy matplotlib
```
