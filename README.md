# Sports Insight Ranking Project

This project is designed to generate, process, and rank sports insights from a variety of articles. The system uses a pipeline of data processing, model training, and evaluation to produce a ranked list of insights for different sports events.

## Project Structure

The project is organized into the following directories:

-   `data_generation/`: Scripts for creating the dataset.
-   `Dataset/`: Contains the raw and processed datasets.
-   `training_code/`: Scripts for training the ranking models.
-   `evaluation_code/`: Scripts and notebooks for evaluating the trained models.
-   `rank/`: Notebook for the final ranking approach.
-   `supplementary_files/`: Additional data files used in the project.

## Step-by-Step Workflow

### 1. Data Generation

The data generation process involves several steps to validate articles, generate insights, and score them for factual consistency.

-   **Article Validation**: The `data_generation/article_validation_save.py` script is used to filter articles and ensure their relevance to specific sports matches.
-   **Insight Generation**: `data_generation/insight_generation.py` generates insights from the validated articles using a large language model.
-   **Factual Scoring**: The generated insights are scored for factual consistency using `data_generation/factScore.py` (with GPT-4o) and `data_generation/summacConv.py` (with SummaCConv).

### 2. Model Training

The core of this project is the training of reward models to rank the generated insights. The training scripts are located in the `training_code/` directory.

-   **Models**: The project uses Llama 3.2 models with 1B and 3B parameters.
-   **Training Objectives**: The models are trained to optimize for either NDCG (`*_ndcg_only.py`) or recall (`*_recall_only.py`).
-   **Supplementary Data**: The training process utilizes data from `supplementary_files/`, which includes `processed_persons.csv`, `sports_keywords.csv`, and `sports_sentiment.csv`.

To run the training for the 1B model with NDCG optimization, you would execute:
```bash
python training_code/Llama-3.2-1B-ndcg_only.py
```

### 3. Evaluation

The performance of the trained models is evaluated using the scripts in the `evaluation_code/` directory. This includes calculating metrics to assess the quality of the insight rankings. The `improvised_Evaluation_code.ipynb` notebook provides a detailed evaluation framework.

### 4. Ranking

The final ranking of insights is performed using the trained models. The `rank/approach_final_ranking.ipynb` notebook contains the implementation of the final ranking logic, applying the reward models to produce a sorted list of insights.

## How to Run the Project

1.  **Set up the Environment**: Ensure you have Python and the necessary libraries installed. The specific requirements can be inferred from the import statements in the Python scripts.
2.  **Data Generation**: Run the scripts in the `data_generation/` directory in the following order:
    1.  `article_validation_save.py`
    2.  `insight_generation.py`
    3.  `factScore.py` and/or `summacConv.py`
3.  **Train the Models**: Run the desired training script from the `training_code/` directory. For example:
    ```bash
    python training_code/Llama-3.2-3B-recall_only.py
    ```
4.  **Evaluate**: Use the notebooks and scripts in `evaluation_code/` to evaluate the model performance.
5.  **Generate Final Rankings**: Run the `rank/approach_final_ranking.ipynb` notebook to get the final ranked insights.

