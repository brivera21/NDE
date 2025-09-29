# Numerical Distance Effect Experiment

A Python implementation of the classic numerical distance effect paradigm based on Moyer and Landauer (1973).

## Background

The **numerical distance effect** is a robust cognitive phenomenon demonstrating that people are faster and more accurate at comparing two numbers when they are farther apart in magnitude. For example, comparing 2 vs 9 is easier than comparing 7 vs 9. This effect suggests that numbers are represented along an internal mental number line with analog properties.

### The Moyer & Landauer (1973) Study

Moyer, R. S., & Landauer, T. K. (1973). Time required for judgements of numerical inequality. *Nature*, 215(5109), 1519-1520.

This seminal paper provided key evidence for analog magnitude representation by showing that reaction times decrease systematically as the numerical distance between compared digits increases.

## Experiment Design

Participants are presented with pairs of single-digit numbers and must quickly judge which number is larger (or smaller, depending on task instructions). 

### Key Predictions

1. **Distance Effect**: Reaction time decreases as numerical distance increases
2. **Problem Size Effect**: Comparisons involving larger numbers may show longer reaction times
3. **High Accuracy**: Performance should be highly accurate (>90%) as the task involves simple numerical comparisons

## Features

- Randomized stimulus presentation
- Reaction time measurement
- Trial-by-trial data logging
- Analysis scripts for distance effect visualization
- Statistical analysis of RT by numerical distance

## Requirements

```
python>=3.7
numpy
pandas
matplotlib
scipy
psychopy  # or other experimental software used
```

## Installation

```bash
git clone https://github.com/yourusername/numerical-distance-effect.git
cd numerical-distance-effect
pip install -r requirements.txt
```

## Usage

### Running the Experiment

```bash
python experiment.py
```

### Analyzing Data

```bash
python analyze_data.py --input data/subject_01.csv
```

## Data Output

The experiment generates CSV files containing:
- Subject ID
- Trial number
- Number 1
- Number 2
- Numerical distance
- Correct response
- Participant response
- Reaction time (ms)
- Accuracy

## Expected Results

A typical result shows:
- Linear decrease in RT as numerical distance increases
- Mean RT difference of ~100-200ms between distance 1 and distance 8
- Overall accuracy >95%

## Repository Structure

```
numerical-distance-effect/
├── NDEexperiment.py       # Main experiment script
├── analyze_data.py        # Data analysis script
└── README.md
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- Moyer, R. S., & Landauer, T. K. (1973). Time required for judgements of numerical inequality. *Nature*, 215(5109), 1519-1520.
