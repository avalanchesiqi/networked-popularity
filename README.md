
# Code and Data for YouTube Networked Popularity Study

We release the code and data for the following paper.
If you use these datasets, or refer to its results, please cite:
> [Siqi Wu](https://avalanchesiqi.github.io/), [Marian-Andrei Rizoiu](http://www.rizoiu.eu/), and [Lexing Xie](http://users.cecs.anu.edu.au/~xlx/). Estimating Attention Flow in Online Video Networks. *ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW)*, 2019. \[[paper](https://avalanchesiqi.github.io/files/icwsm2018engagement.pdf)\]

## Code usage
We provide three quickstart bash scripts:
1. [run_all_wrangling.sh](/wrangling/run_all_wrangling.sh)
2. [run_all_measures.sh](/measures/run_all_measures.sh)
3. [run_all_models.sh](/models/run_all_models.sh)

Download and place data in the [data](/data) directory, then uncompress them.
First run `run_all_wrangling.sh` to create formatted data, then run `run_all_temporal_analysis.sh` to conduct the temporal analysis or `run_all_predictors.sh` to reproduce the results of prediction tasks.
Detailed usage and running time are documented in the corresponding python scripts.
Plotting scripts to generate figures in the paper are in the [plots](/plots) directory.

Note the datasets are large, so the quickstart scripts will take up to 24 hours to finish.
Check the estimated running time in each python script before you run the quickstart scripts.

## Python packages version
All codes are developed and tested in Python 3.6, along with NumPy 1.13, matplotlib 2.1 and SciPy 0.19.

## Data
The data is hosted on [Google Drive](https://drive.google.com/drive/folders/19R3_2hRMVqlMGELZm47ruk8D9kqJvAmL?usp=sharing) and [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TORICY).
  
  ```
  data
  │   README.md
  └───vevo_en_videos_60k.tar.bz2
  │   │   vevo_en_videos_60k.json
  └───vevo_en_embeds_60k.txt
  └───vevo_forecast_data_60k.tsv
  └───recsys1.tar.bz2
  │   │   recsys_2018-09-01.p
  │   │   ...
  │   │   recsys_2018-10-02.p
  └───recsys2.tar.bz2
  │   │   recsys_2018-10-03.p
  │   │   ...
  │   │   recsys_2018-11-02.p
  └───network_pickle.tar.bz2
  │   │   network_2018-09-01.p
  │   │   ...
  │   │   network_2018-11-02.p
  ```

### File Description
Data are compressed in `tar.bz2`.
Uncompress by command `find -name "*.tar.bz2" -exec tar -jxvf {} \;`.

File | Uncompressed | Compressed
--- | --- | ---
vevo_en_videos_60k.json | 1.4GB | 295MB
network_pickle/ | 3GB | 952MB

<!---
### Tweeted videos dataset
This dataset contains YouTube videos published between July 1st and August 31st, 2016.
To be collected, the video needs
(a) be mentioned on Twitter during aforementioned collection period;
(b) have insight statistics available;
(c) have at least 100 views within the first 30 days after upload.

### Quality videos datasets
These datasets contain videos deemed of high quality by domain experts.
* Vevo videos: Videos of verified Vevo artists, as of August 31st, 2016.
* Billboard16 videos: Videos of [2016 Billboard Hot 100 chart](http://www.billboard.com/charts/year-end/2016/hot-100-songs).
* Top news videos: Videos of [top 100 most viewed News channels](https://vidstatsx.com/youtube-top-100-most-viewed-news-politics).

## Video Data Fields
Each line is a YouTube video in `json` format, an example is shown below.
```json
{
   "id": "pFMj8KL8nJA",
   "snippet": {
      "description": "For more on India's goods and services tax and the future of the economy under Prime Minister Narendra Modi, CCTV America\u2019s Rachelle Akuffo interviewed Peter Kohli, the chief investment officer at D-M-S Funds.",
      "title": "Peter Kohli on the importance of the goods and services tax",
      "channelId": "UCj7wKsOBhRD9Jy4yahkMRMw",
      "channelTitle": "CCTV America",
      "publishedAt": "2016-08-10T00:34:01.000Z",
      "categoryId": "25",
      "detectLang": "en"
   },
   "contentDetails": {
      "duration": "PT5M27S",
      "definition": "hd",
      "dimension": "2d",
      "caption": "false"
   },
   "topicDetails": {
      "topicIds": ["/m/0546cd"],
      "relevantTopicIds": ["/m/03rk0", "/m/0gfps3", "/m/0296q2", "/m/05qt0", "/m/0dgrhmk", "/m/09x0r", "/m/05qt0", "/m/098wr"]
   },
   "insights": {
      "startDate": "2016-08-10",
      "days": "0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,18,19,23,26,29,30,44,45,62,69,114,118,122,149,154,159,160,182,188,189,199,204,226,253",
      "dailyView": "70,11,15,7,7,8,11,4,7,2,2,1,6,6,3,2,2,2,1,1,4,1,1,1,1,2,3,1,1,1,1,3,1,2,2,1,1,1,1",
      "totalView": "281",
      "dailyWatch": "171.966666667,22.35,42.95,24.6333333333,26.05,25.3833333333,34.25,9.63333333333,6.31666666667,0.7,7.13333333333,0.0333333333333,15.2333333333,16.7,2.2,0.116666666667,0.966666666667,1.1,5.43333333333,5.43333333333,10.7666666667,1.2,5.43333333333,1.8,5.43333333333,5.45,3.15,0.2,1.68333333333,0.733333333333,0.483333333333,3.21666666667,5.43333333333,0.383333333333,5.6,0.0666666666667,0.533333333333,5.43333333333,1.06666666667",
      "avgWatch": "2.3290628707",
      "dailyShare": "2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
      "totalShare": "2",
      "dailySubscriber": "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
      "totalSubscriber": "0"
   }
}
```

### detectLang field
`detectLang` is the result from [langdetect 1.0.7](https://pypi.python.org/pypi/langdetect?), 'NA' if no result returns.
Note in the latest version of [youtube-insight](https://github.com/avalanchesiqi/youtube-insight), we changed to [googletrans 2.3.0](https://pypi.org/project/googletrans/).

### topicDetails field
`topicIds` and `relevantTopicIds` are resolved to entity name via the latest [Freebase data dump](https://developers.google.com/freebase/).
We provide extracted mapping results in `freebase_mid_type_name.csv`.
Our parser is inspired by the [Freebase-to-Wikipedia](https://github.com/saleiro/Freebase-to-Wikipedia) project.
--->