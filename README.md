## Quick Start

1. **Setup Environment**

Use Python verison 3.10.x

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download MovieLens-100K**
```bash
mkdir -p data/raw
cd data/raw
wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
cd ../..
```

3. **Run the Project**
```bash
python main.py
```