uid_item_hist = {}


cnt = 0
for line in open('/Users/yukun/Desktop/dev_hist.tsv'):
    cnt += 1
    if cnt % 1000 == 0:
        pass
        # print cnt
    uid, query, nid, click, _, _, _, pv_time, _, _, hist = line.split('\t')

    hist = hist.strip()
    if uid not in uid_item_hist:
        uid_item_hist[uid] = {}
    rn = pv_time + '-' + nid
    if rn not in uid_item_hist[uid]:
        uid_item_hist[uid][rn] = {}
    if hist == '':
        continue
    for hist_i in hist.split('\x03'):
        uid_item_hist[uid][rn][hist_i] = 0

print len(uid_item_hist)

cnt = 0
for line in open('/Users/yukun/Desktop/new_dev.tsv'):
    cnt += 1
    if cnt % 1000 == 0:
        pass
        # print cnt
    uid, query, nid, click, _, _, _, pv_time, hist = line.split('\t')
    hist = hist.strip()
    if uid not in uid_item_hist:
        print line[:-1]
        print uid
        assert uid in uid_item_hist
    rn = pv_time + '-' + nid
    if rn not in uid_item_hist[uid]:
        print line[:-1]
        print rn
        assert rn in uid_item_hist[uid]
    if hist == '':
        continue
    for hist_i in hist.split('\x03'):
        if hist_i not in uid_item_hist[uid][rn]:
            print line[:-1]
            print hist_i
            assert hist_i in uid_item_hist[uid][rn]