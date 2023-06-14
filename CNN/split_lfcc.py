import pandas as pd
import numpy as np
import torch


feats = pd.read_hdf('out/feats.h5')
print(feats)
new_feats = pd.DataFrame(columns=['fid', 'features', 'class', 'utt'])
feat_len = 30
print(feats.columns)
for i in feats.index:
    feat = feats['features'][i]
    # feat shape: (time, 60)
    if feat.shape[0] > feat_len:
        # devide into segments
        for j in range(0, feat.shape[0], feat_len):
            if j + feat_len > feat.shape[0]:
                continue
            new_feats = new_feats.append({'fid': feats['utterance-id'][i] + str(j), 'features': feat[j:j+feat_len, :], 'class': feats['speaker-id'][i], 'utt': feats['recording-id'][i]}, ignore_index=True)

    else:
        new_feats = new_feats.append({'fid': feats['utterance-id'][i], 'features': feat, 'class': feats['speaker-id'][i], 'utt': feats['recording-id'][i]}, ignore_index=True)
        

print(new_feats)
new_feats.to_hdf('out/feats_split.h5', key='feats', mode='w')