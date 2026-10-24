# Data files

These files contain comma-separated numeric rows **without header lines**. The files are kept in this archive to make the practice notebooks reproducible.

- `house_prices.csv`: four input features (house size, bedrooms, floors, age) followed by the house-price target. The notebook examples describe the target in thousands of dollars.
- `student_admission.csv`: two exam-score features followed by a binary admission label (`0` or `1`).
- `microchip_test_results.csv`: two test-result features followed by a binary class label (`0` or `1`).

The loaders in `src/` resolve these files relative to the repository rather than relying on the notebook's current working directory.
