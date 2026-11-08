# Dataset location

The dataset lives here as:

```
housing.csv
```

Full path:

```
/home/jabs101/Documents/DEATHON INTERNSHIP/Task-02-California-Housing-Regression/data/raw/housing.csv
```

It is the Kaggle "California Housing Prices" CSV (20,640 rows, 10 columns):

```
longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
population, households, median_income, median_house_value, ocean_proximity
```

Two columns matter for the task requirements:

- `total_bedrooms` - 207 missing values, needs imputation
- `ocean_proximity` - categorical (5 classes), needs encoding

The notebook reads this exact filename. If the file is renamed, update
`DATA_PATH` in the first code cell.
