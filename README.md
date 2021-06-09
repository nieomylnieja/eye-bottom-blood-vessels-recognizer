# Retinal blood vessels recognizer

- dataset: [HRF (High-Resolution Fundus)](https://www5.cs.fau.de/research/data/fundus-images/)
- [Frangi](https://scikit-image.org/docs/0.14.x/auto_examples/filters/plot_frangi.html) filter was used alongside some
  adjustments to enhance the readability of the resulting image
- the GUI runs on `streamlit` to run locally:
    ```shell
    pipenv install && pipenv shell && streamlit run main.py
    ```
- you can adjust the following parameters:
    - HSV V channel threshold, which is used to reduce the bright spots created during the image creation
    - Visibility enhancement multiplier, which is applied on the resulting image to intensify the whiteness of the
      elements

- finally, a statistics report is generated with accuracy, sensitivity and specificity for four different hit/miss
  thresholds