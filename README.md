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
