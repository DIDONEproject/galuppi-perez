# Didone Aria Analysis

This is a repo based on [python-dust](https://github.com/00sapo/python-dust)

Install dependencies: `./install.sh`

Uninstall: remove this directory.

## Reproducible research

When the corpus will be released, download it and put it in the "data/corpus" folder.
Otherwise, the extraction will be based on the "data/cache" folder.

Extract the features: `./dust -m modeling features`

Run the experiments: `./run.sh`

## Usage example

### CLI

The cli interface help can be seen with `./dust -m modeling`. Examples of usage for the
cli are in the `run.sh` file.

### API

```python
from modeling import easy_tools
from modeling.data import load_features
from modeling import automl

# load the dataframes and sets 'Composer' as y column
data, X, y = load_features('Composer')

# 1. select only Galuppi and Perez (no needed if using corpus from this repo)
# 2. pickle the aria tested at the end (questionable arias) and removes them
# 3. removes columns containing NaN
X, y = easy_tools.select_galuppi_perez(easy_tools, self.prehook)(data, X, y)

# run automl process (time in seconds)
auotml(data_x_y=(X, y), automl_time=4*3600)
```

Documentation of each function can be found into the code.

For more advanced examples, see `modeling/__main__.py`. Just keep in mind that any
method of the `Model` class can be used as a command for the cli interface (thanks to
the `fire` package).

### Re-using the corpus

#### Features

For extracting features, we use the [musif](https://musif.didone.eu) python package.

The list and definition of all the features is avalable [here](https://musif.didone.eu/Feature_definition.html).

The source scores will be released in the context of the [Didone ERC
project](https://didone.eu). For later reference, in this project we used the dataset as
of 21 November 2023, git hash `a3e30ac20f6df909065f75425da36e8f7819bcfe`.

#### Dataframe music scores

To provide an alternative method to deal with the corpus, we also provide some dataframes in
the `data/dfs` folder. Note that we extract features starting from MusicXML and
MuseScore files and these dataframes are not used in this code. However, they may be
useful for working on this same dataset using different approaches (neural networks,
HMM, etc.). To load these files, you can unpickle them or
use the function `musif.cache.load_score_df`. The object contained is a dictionary where
the keys are the names of the parts and the values are dataframes with the following
columns:

- "Type": A string identifying the type of object. Possible values: `"Note"`,
  `"Rest"`, `"Measure"`, `"Time Signature"`
- "Name": A string with the name of the note in Common Western Notation or with
  the time signature string for time signatures; for measures and rests, the value
  `"-"` is used.
- "Value": The midi pitch for notes, -1 for others
- "Measure Onset": The beat position of the object in reference to the beginning
  of the measure, -1 for measures
- "Part Onset": The onset position of the object in reference to the beginning
  of the part
- "Duration": The duration of the object, -1 for time signatures
- "Tie": If a tie is applied to the note, its type is there (one of `"start"`,
  `"continue"`, `"stop"`), otherwise `"-"` is used

This Corpus is a subset of the corpus created by the [Didone](https://didone.eu)
project. As such, the original scores will be released all together to the rest of the corpus.

# Credits

Didone Project — https://didone.eu

# Cite us

Work under publication
