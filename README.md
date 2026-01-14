# Zomato Restaurant Intelligence Recommendation Platform

A data-driven platform that leverages machine learning to deliver personalized restaurant recommendations using Zomato dataset features like cuisine, location, ratings, and reviews. Built for users seeking intelligent dining suggestions and restaurant owners analyzing trends.

## Features

- Content-based filtering with TF-IDF and cosine similarity for precise matches.
- Data cleaning, preprocessing, and analysis of Zomato datasets (cuisines, prices, ratings).
- Personalized top-10 restaurant suggestions based on user preferences.
- Insights for new restaurants on themes, menus, and optimal locations.


## Tech Stack

| Category | Technologies |
| :-- | :-- |
| Language | Python, Pandas, NumPy |
| ML/Recommenders | Scikit-learn (TF-IDF, Cosine Similarity), possibly Neural Networks |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebooks, VS Code |
| Version Control | Git/GitHub |

## Prerequisites

- Python 3.8+ installed.
- Git for cloning the repo.
- Zomato dataset (e.g., from Kaggle: restaurants, reviews, metadata).

## Installation

1. Clone the repository:

```
git clone https://github.com/KaustubhMestri/Zomato-Restaurant-Intelligence-Recommendation-Platform.git
cd Zomato-Restaurant-Intelligence-Recommendation-Platform
```

2. Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

Core packages: `pandas scikit-learn numpy matplotlib seaborn jupyter`.

## Usage

1. Launch Jupyter Notebook:

```
jupyter notebook
```

2. Open `main.ipynb` or equivalent analysis notebook.
3. Run data preprocessing cells to clean dataset (remove duplicates, NaNs, preprocess text).
4. Input a restaurant name or user preferences; generate recommendations:
Example output: Top similar restaurants sorted by aggregate rating.

For a quick demo:

```python
recommend_restaurants('example_restaurant_name', top_n=10)
```


## Project Structure

```
Zomato-Restaurant-Intelligence-Recommendation-Platform/
├── data/
│   ├── raw/          # Zomato CSV/JSON datasets
│   └── processed/    # Cleaned data
├── notebooks/        # Analysis and ML notebooks
├── src/
│   ├── preprocess.py # Data cleaning
│   └── recommender.py # Core recommendation logic
├── requirements.txt
└── README.md
```


## Contributing

Fork the repo, create a feature branch, add changes, and submit a PR. Focus on model improvements or new features like collaborative filtering.

Follow PEP8 standards. Test with provided notebooks.

## License

MIT License - feel free to use and modify.

## Acknowledgments

Inspired by Zomato datasets on Kaggle and standard recommender systems. Thanks to open-source tools like Scikit-learn.
<span style="display:none"></span>

<div align="center">⁂</div>
