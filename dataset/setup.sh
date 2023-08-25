kaggle datasets download -d coldfir3/aslfs-dataset-compression-complete-4c
kaggle datasets download -d coldfir3/aslfs-dataset-compression-complete-4caux
mkdir .data
unzip aslfs-dataset-compression-complete-4c.zip -d .data/parquet
unzip aslfs-dataset-compression-complete-4caux.zip -d .data/parquet
rm aslfs-dataset-compression-complete-4c.zip
rm aslfs-dataset-compression-complete-4caux.zip
