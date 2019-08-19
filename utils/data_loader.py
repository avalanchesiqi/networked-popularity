import json
import numpy as np


class DataLoader:
    TARGET_LISTS = ['Pop_music', 'Rock_music', 'Hip_hop_music', 'Independent_music',
                    'Country_music', 'Electronic_music', 'Soul_music', 'Others']

    def __init__(self):
        self.vevo_en_vid_list = None
        self.vid_embed_dict = None
        self.embed_view_dict = None
        self.embed_avg_view_dict = None
        self.num_videos = 0

        self.embed_cid_dict = None
        self.embed_title_dict = None
        self.embed_uploadtime_dict = None
        self.embed_genre_dict = None

    def load_video_views(self):
        self.vid_embed_dict = {}
        self.embed_view_dict = {}
        self.embed_avg_view_dict = {}
        with open('../data/vevo_forecast_data_60k.tsv', 'r') as fin:
            for line in fin:
                embed, vid, ts_view, total_view = line.rstrip().split('\t')
                embed = int(embed)
                self.embed_view_dict[embed] = [int(x) for x in ts_view.split(',')]
                self.vid_embed_dict[vid] = embed
                self.embed_avg_view_dict[embed] = np.mean(self.embed_view_dict[embed])
        self.vevo_en_vid_list = sorted(self.vid_embed_dict.keys())
        self.num_videos = len(self.vevo_en_vid_list)
        print('>>> Daily view data has been loaded!')

    def load_embed_content_dict(self):
        if self.vid_embed_dict is None:
            self.load_video_views()
            self.load_embed_content_dict()
        else:
            self.embed_cid_dict = {}
            self.embed_title_dict = {}
            self.embed_uploadtime_dict = {}
            self.embed_genre_dict = {}
            with open('../data/vevo_en_videos_60k.json', 'r') as fin:
                for line in fin:
                    video_json = json.loads(line.rstrip())
                    vid = video_json['id']
                    embed = self.vid_embed_dict[vid]

                    cid = video_json['snippet']['channelId']
                    self.embed_cid_dict[embed] = cid

                    title = video_json['snippet']['title']
                    self.embed_title_dict[embed] = title

                    uploadtime = video_json['snippet']['publishedAt'][:10]
                    self.embed_uploadtime_dict[embed] = uploadtime

                    if 'topics' in video_json:
                        topics = [x for x in video_json['topics'] if x in self.TARGET_LISTS]
                    else:
                        topics = []
                    self.embed_genre_dict[embed] = topics
