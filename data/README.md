# Vevo Music Graph Dataset

These datasets are first used in the following paper.
If you use these datasets, or refer to its results, please cite:
> [Siqi Wu](https://avalanchesiqi.github.io/), [Marian-Andrei Rizoiu](http://www.rizoiu.eu/), and [Lexing Xie](http://users.cecs.anu.edu.au/~xlx/). Estimating Attention Flow in Online Video Network. *ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW)*, 2019. \[[paper](https://avalanchesiqi.github.io/files/cscw2019network.pdf)\]

## Data
The data is hosted on [Google Drive](https://drive.google.com/drive/folders/19R3_2hRMVqlMGELZm47ruk8D9kqJvAmL?usp=sharing) and [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TORICY).

  ```
  data
  │   README.md
  └───artist_details.json
  └───network_pickle.tar.bz2
  │   │   network_2018-09-01.p
  │   │   ...
  │   │   network_2018-11-02.p
  └───persistent_network.csv
  └───recsys1.tar.bz2
  │   │   recsys_2018-09-01.p
  │   │   ...
  │   │   recsys_2018-10-02.p
  └───recsys2.tar.bz2
  │   │   recsys_2018-10-03.p
  │   │   ...
  │   │   recsys_2018-11-02.p
  └───teaser.json
  └───vevo_en_embeds_60k.txt
  └───vevo_en_videos_60k.tar.bz2
  │   │   vevo_en_videos_60k.json
  └───vevo_forecast_data_60k.tsv
  ```

### File Description
Data are compressed in `tar.bz2`.
Uncompress by command `find -name "*.tar.bz2" -exec tar -jxvf {} \;`.

File | Uncompressed | Compressed
--- | --- | ---
artist_details.json | 1.2MB | -
network_pickle/ | 3GB | 952MB
persistent_network.csv | 652KB | -
recsys/ | 7.7GB | 1.8+1.7GB
teaser.json | 212KB | -
vevo_en_embeds_60k.txt | 3.6MB | -
vevo_en_videos_60k.json | 1.4GB | 295MB
vevo_forecast_data_60k.tsv | 15MB | -

### artist_details.json
4435 artist metadata crawled from MusicBrainz, complemented with artist Twitter Id.
Each line contains data for an artist.
```json
{"query": "justin bieber",
 "channel_title": "JustinBieberVEVO",
 "channel_id": "UCHkj014U2CQ2Nv0UZeYpE_A",
 "twitter_id": "justinbieber",
 "artist_id": "e0140a67-e4d1-4f13-8a01-364355bee46e",
 "artist_name": "justin bieber",
 "country": "CA",
 "tag-dict": {"pop": 5, "electropop": 1, "dance-pop": 1, "teen pop": 2, "contemporary r&b": 1, "tropical house": 1}
}
```

### network_pickle/
63 daily snapshots of relevant network in the Vevo dataset, in the format of video embed.
Each file, e.g., `network_2018-09-01.p`, contains a daily snapshot for the videos in Vevo network in `pickle` format.
We use the embed of target video as key, the list of videos pointing towards to it as value, associated by its rank on relevant list and views on that day.

An example is `{0: [(13028, 36, 194), (54133, 0, 160), (42948, 36, 300), (20438, 29, 162)]}`.
It means target video `--5boVTkNSo` ("Erick Sermon - Feel It ft. Sy Scott", embed 0) is linked by source video `CqMDF5NLi_M` ("Erick Sermon - Hittin" Switches', embed 13028),
where `--5boVTkNSo` appears at position 36 of `CqMDF5NLi_M`'s relevant list.
`CqMDF5NLi_M` has 194 views on that day (2018-09-01).

### persistent_network.csv
52758 extracted persistent network in the format of (source embed, target embed) pair, delimited by comma.
The first line is header.

### recsys/
63 daily snapshots of both the relevant and recommended networks, in the format of raw text.
Each file, e.g., `recsys_2018-09-01.json`, contains the target vid with its recommended_list, recommended_views, and relevant_list.
recommended_list is collected by crawling the video list on right-hand panel.
recommended_views is the cumulative view count for the videos in recommended_list at the time of crawling.
relevant_list is collected by querying YouTube Data API.

### teaser.json
6 videos from Adele to plot the teaser figure.

### vevo_en_embeds_60k.txt
60740 video embeds in the format of (embed, vid, video title) pair, delimited by comma.
```text
0,--5boVTkNSo,Erick Sermon - Feel It ft. Sy Scott
```

### vevo_en_videos_60k.json
60740 Vevo video in `json` format.
```json
{"id": "uJz3sRJ1fTQ",
 "snippet": {"publishedAt": "2018-07-12T23:01:00.000Z",
             "channelId": "UCKonA-DOxJbL0VHUDlc3vsA",
             "title": "Bury Tomorrow - Adrenaline (Audio)",
             "description": "collapsed long text description...",
             "thumbnails": "https://i.ytimg.com/vi/uJz3sRJ1fTQ/default.jpg",
             "channelTitle": "BuryTomorrowVEVO",
             "tags": ["tag 1", "tag 2", "tag n"],
             "categoryId": "10"
             },
 "contentDetails": {"duration": "PT2M44S",
                    "definition": "hd",
                    "caption": "false",
                    "licensedContent": true,
                    "regionRestriction": {"allowed": ["country 1", "country 2", "country n"]}
                    },
 "insights": {"dailyView": [1163, 3407, 1100, 850, 1131, 1156, 1197, 969, 929, 666, 529, 626, 697, 693, 721, 700, 542, 524, 614, 615, 572, 494, 515, 425, 343, 489, 447, 403, 435, 424, 347, 284, 321, 298, 342, 365, 358, 296, 199, 325, 288, 356, 355, 335, 273, 225, 286, 276, 321, 322, 305, 263, 236, 265, 263, 309, 287, 310, 213, 209, 253, 284, 230, 241, 221, 193, 180, 222, 241, 242, 226, 237, 200, 166, 219, 242, 179, 226, 299, 227, 226, 299, 266, 196, 207, 185, 143, 140, 187, 196, 168, 182, 208, 148, 150, 215, 197, 204, 189, 193, 179, 148, 174, 169, 172, 197, 209, 147, 102, 162, 132, 191, 167, 151],
              "startDate": "2018-07-12",
              "totalView": 42935,
              "dailyShare": [38, 77, 28, 12, 19, 19, 10, 10, 7, 9, 6, 1, 6, 1, 7, 4, 4, 3, 7, 3, 3, 5, 1, 3, 4, 1, 6, 1, 2, 3, 2, 2, 0, 3, 4, 0, 2, 0, 0, 3, 2, 2, 0, 1, 0, 4, 1, 2, 2, 7, 1, 2, 1, 4, 1, 8, 2, 3, 0, 4, 4, 0, 0, 1, 0, 0, 2, 0, 4, 5, 3, 2, 1, 3, 2, 5, 0, 2, 1, 1, 0, 0, 0, 1, 5, 4, 2, 2, 1, 0, 2, 2, 1, 1, 2, 2, 1, 1, 0, 2, 2, 0, 1, 1, 0, 0, 4, 0, 1, 1, 1, 3, 1, 1],
              "totalShare": 444,
              "dailyWatch": [1898.31666667, 6293.01666667, 2077.51666667, 1608.01666667, 2222.45, 2398.23333333, 2487.4, 2170.83333333, 1885.93333333, 1338.36666667, 1048.16666667, 1320.76666667, 1578.21666667, 1585.98333333, 1631.46666667, 1567.1, 1214.91666667, 1172.81666667, 1390.43333333, 1385.86666667, 1231.33333333, 1110.85, 1134.91666667, 913.533333333, 723.916666667, 1130.81666667, 1010.56666667, 912.466666667, 1035.6, 946.15, 754.583333333, 608.316666667, 727.616666667, 656.533333333, 792.433333333, 823.966666667, 790.466666667, 644.35, 430.3, 711.7, 652.6, 805.55, 781.383333333, 716.933333333, 592.65, 447.166666667, 603.6, 607.616666667, 709.566666667, 718.266666667, 683.466666667, 592.65, 524.35, 591.1, 584.416666667, 669.016666667, 627.733333333, 694.783333333, 474.683333333, 451.616666667, 547.45, 605.266666667, 491.416666667, 483.933333333, 498.966666667, 424.4, 365.8, 481.5, 547.3, 510.45, 492.216666667, 505.716666667, 417.716666667, 366.766666667, 521.383333333, 499.616666667, 372.016666667, 498.55, 675.35, 473.1, 439.916666667, 632.05, 582.916666667, 422.15, 447.4, 404.933333333, 312.133333333, 277.3, 411.4, 426.55, 346.8, 418.716666667, 424.516666667, 312.883333333, 332.633333333, 478.633333333, 457.266666667, 453.95, 415.966666667, 423.9, 389.6, 314.6, 352.3, 364.666666667, 391.75, 452.816666667, 450.4, 314.033333333, 224.166666667, 369.966666667, 294.166666667, 402.133333333, 357.85, 305.416666667],
              "avgWatch": 2.121350491053197,
              "endDate": "2018-11-02"
              },
 "topics": ["Music", "Rock_music"]
 }
```

### vevo_forecast_data_60k.tsv
60740 Vevo video in the format of (embed, vid, daily views in 63 days, total views in 63 days) pair, delimited by tab.
```text
0       --5boVTkNSo     105,92,77,90,94,82,93,83,69,84,82,98,73,102,97,93,92,98,86,71,107,107,47,74,78,66,68,87,86,74,73,67,50,57,88,75,72,74,71,53,66,90,92,73,65,70,59,62,81,81,44,63,69,76,72,84,70,60,82,69,84,77,90        4914
```
