from math import ceil


def resample(ts, max_gap=3, max_len=30):
    for s in ts:
        if s['end'] - s['start'] > max_len:
            print(f'Error: max_len ({max_len}) is smaller than one of the segments ({s})!')
            return None

    gts = []
    g = [ts[0]]
    for s in ts[1:]:
        if s['start'] - g[-1]['end'] < max_gap:
            g.append(s)
        else:
            gts.append(g)
            g = [s]
    gts.append(g)

    ret = []
    for g in gts:
        l = g[-1]['end'] - g[0]['start']
        split_num = ceil(l / max_len)
        if split_num > 1:
            min_len = l / split_num
            start = g[0]['start']
            end = g[0]['end']
            for s in g[1:]:
                if s['end'] - start > max_len or end - start > min_len:
                    ret.append({'start': start, 'end': end})
                    start = s['start']
                    end = s['end']
                else:
                    end = s['end']
            ret.append({'start': start, 'end': end})
        else:
            ret.append({'start': g[0]['start'], 'end': g[-1]['end']})

    return ret
