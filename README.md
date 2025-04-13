# LLM Context Length Test for Ollama
This tool is made to determine open source (ollama) models context length
Runs different kind of tests and then evaluates them into the results folder by default
Some example test result were posted with modified num_ctx parameter

There is a "model generator" folder where you can easily generate models with modified num_ctx count

## Installation
```
git clone https://github.com/cride9/LLM-Benchmark.git
cd LLM-Benchmark
```

### Install packages
```
python -m venv venv

windows:
venv\scripts\activate

# unix:
# source venv\bin\activate

pip install -r requirements.txt
```

## Usage
```
python benchmark.py --context-length 10000,20000,40000 --model granite3.249152:latest
```

## Available tests
- all
- basic_retrieval
- memory_association
- multi_document
- topic_switching

### Switches
| Switch   | Meaning | Default |
| -------- | ------- | ------- |
| --model  | Name of the Ollama model to benchmark    | - |
| --output-dir | Directory to save benchmark results     | results |
| --context-lengths    | Comma-separated list of context lengths to test    | 1000,5000,10000,20000 |
| --test    | Which test to run    | all |
| --test-mode    | Run in test mode without requiring Ollama (for testing the script)    | - |

## Model generator
```
generator.bat modelname contextlength
# generator.bat llama3.2:latest 65536

# output: llama3.2:latest65536
```
