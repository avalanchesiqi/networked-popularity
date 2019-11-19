# Code and Data for YouTube Networked Popularity Study

We release the code and data for the following paper.
If you use these datasets, or refer to its results, please cite:
> [Siqi Wu](https://avalanchesiqi.github.io/), [Marian-Andrei Rizoiu](http://www.rizoiu.eu/), and [Lexing Xie](http://users.cecs.anu.edu.au/~xlx/). Estimating Attention Flow in Online Video Networks. *ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW)*, 2019. \[[paper](https://avalanchesiqi.github.io/files/cscw2019network.pdf)\|[slides](https://avalanchesiqi.github.io/files/cscw2019slides.pdf)\|[blog](https://medium.com/acm-cscw/how-does-the-network-of-youtube-music-videos-drive-attention-42130144b59b)\]

## Code usage
We provide three quickstart bash scripts:
1. [run_all_wrangling.sh](/wrangling/run_all_wrangling.sh)
2. [run_all_measures.sh](/measures/run_all_measures.sh)
3. [run_all_models.sh](/models/run_all_models.sh)

Download and place data in the [data](/data) directory, then uncompress them.
First run `run_all_wrangling.sh` to create formatted data, then run `run_all_temporal_analysis.sh` to conduct the temporal analysis or `run_all_predictors.sh` to reproduce the results of prediction tasks.
Detailed usage and running time are documented in the corresponding python scripts.

Note the datasets are large, so the quickstart scripts will take up to 24 hours to finish.
Check the estimated running time in each python script before you run the quickstart scripts.

## Python packages version
All codes are developed and tested in Python 3.6.7, along with NumPy 1.14.5, matplotlib 3.0.3 and SciPy 1.2.1.

## Data
The data is hosted on [Google Drive](https://drive.google.com/drive/folders/19R3_2hRMVqlMGELZm47ruk8D9kqJvAmL?usp=sharing) and [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TORICY).
See more details in this [data description](/data/README.md).
