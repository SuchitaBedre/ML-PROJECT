## Dataset

The merged and cleaned dataset (`merged_food_dataset_cleaned.csv`, ~268MB) is too large for GitHub 
and is hosted on Google Drive instead:

📂 **Download link:** [merged_food_dataset_cleaned.csv](https://drive.google.com/file/d/1wzqEfBtVdo2qeicGiZixVOwdfb026_2W/view?usp=drive_link)

### How to use in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/[your-folder]/merged_food_dataset_cleaned.csv")
```
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

New Preprocessed dataset - 

https://drive.google.com/file/d/1O_W2gLcYEMiqvvIzZka5DrgV0CYZpo29/view?usp=drive_link

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

### folder structure of this project:

https://drive.google.com/file/d/1Kga3hlasWqepgtYkKoFVEMFENNvUj7_I/view?usp=drive_link

## data folder zip file:
https://drive.google.com/file/d/1UcSxORIf8b6TRkb06X6RRlHaRVVhn8kd/view?usp=drive_link

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


## Dataset

The preprocessed recipe dataset (`recipe_final_preprocessed.csv`, ~1.6GB, 1,132,366 rows, 20 columns) 
is too large for GitHub and is hosted on Google Drive instead:

📂 **Download link:** https://drive.google.com/file/d/1X18do7geyLQK7je4rzReBGERGmh6shPm/view?usp=drive_link
## Dataset

The preprocessed recipe dataset (`recipe_final_preprocessed.csv`, ~1.6GB, 1,132,366 rows, 20 columns) 
is too large for GitHub and is hosted on Google Drive instead:

📂 **Download link:** https://drive.google.com/file/d/1X18do7geyLQK7je4rzReBGERGmh6shPm/view?usp=drive_link

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### How to use in Colab
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv("/content/drive/MyDrive/[your-folder]/recipe_final_preprocessed.csv")
print(df.shape)  # should show (1132366, 20)
```

### Columns
rating, review, name, minutes, n_steps, description, n_ingredients, calories, 
total_fat, sugar, sodium, protein, saturated_fat, carbohydrates, tags_text, 
ingredients_text, steps_text, review_year, recipe_age_days, rating_binary

**Target column:** `rating_binary` (1 = rating ≥4, 0 = rating <4)
