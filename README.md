# Janus
An interactive interface for GPT-X, built with Streamlit.

## Contributing

If contributing to this repository, please make sure to do the following:

+ On installing new dependencies (via `pip` or `conda`), please make sure to update the `environment.yml` files via the
  following command (note that you need to separately create the `environment-osx.yml` file by exporting from Mac OS!):
  
  `rm environment.yml; conda env export --no-builds | grep -v "^prefix: " > environment.yml`

---


## Quickstart on MacOS (CPU-Only)
Clone the repository, and create a conda environment with `environment-osx.yaml`.

```shell script
git clone https://github.com/stanford-mercury/janus.git
cd janus
conda env create -f environment-osx.yml
```

## Running Janus
Run the streamlit application and you're good to go!

```shell script
streamlit run main.py
```
