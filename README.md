# A Biologically Constrained Model of Semantic Memory Search
This is a provisory repository for the paper submission: "A Biologically Constrained Model of Semantic Memory Search". 

These instructions explain how to install the necessary software, run the model and reproduce paper results. Before doing this, it is helpful to read the paper. Also, since this is a provisory repository some parts of it might be unstable.

1. Install all Python package requirements with `pip install -r requirements.txt` (this might take some time). To run simulations faster (used here) install `pyopencl` and `nengo_ocl` (with pip). Installing them might require some manual intervention on some machines.

2. Install this project as package: `python setup.py develop`

3. Fetch FAN data from the web by running `python fetch_external.py` in the
   `scripts` folder. This will get Free Norms from the University of South
   Florida webiste, which hosts the data. Ngram data is also needed, [here](https://figshare.com/articles/Processed_Ngram_data/4733674) we provide bi-gram matrices. They need to be manually downloaded into the `association_data` directory.

4. In the same directory, run `python categorize_animals.py`. This step will
   create pickled files in `animal_data` that contain dictionary with
   animal to category mappings.

5. In the `cogsci17-semflu` directory run `python create_database.py` to
   create multiple pickled databases of association data and association matrices (FAN,
   Beagle and Ngrams) that are loaded by the model.

## Run the model

To run the model, it should suffice to run the script `run_models.py` in the `./cogsci17_semflu/models/` directory. Model simulations will be generated in the sub-directory `data`.

## Reproducing data

The `results-plot` notebook in the `notebook` directory can be used to reproduce Fig2 from the paper.
